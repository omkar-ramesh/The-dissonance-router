"""Hope: the team lead who reviews a failure and proposes a fix, without calling a model.

A failure already comes with a reason code. Hope looks the code up in a playbook, writes down why it
happened and what fixes it, and checks the fix by re-reading the failing message with the reader's rules
(plain regex, so it costs 0 tokens). Fixes are either a rule added to the signal reader (kept, so the
router gets better) or removing an unverified link from the answer. Failures that need a human decision
(tier thresholds) get a diagnosis and no automatic fix.

What Hope can't do for free is write a better answer: that needs the model again and is a separate,
priced step. Lessons are stored as codes and patterns, never message text.
"""
from contextlib import closing

from src.agent import signals
from src.agent.honesty import find_unbacked_sources
from src.db.database import get_connection

PLAYBOOK = {
    "MISSED_SIGNAL": {
        "why": "The reader only knows phrasings about the answer (\"make it shorter\"). A complaint with no style "
               "words slipped through, so no habit was learned.",
        "fix": "Teach the reader that complaints about length ('wall of text', 'not reading all that') mean shorter answers.",
        "kind": "rule",
        "rules": [("length", "concise", r"wall of text|too much text|not reading (all|any) (of )?(that|this|it)|so (long|wordy)")],
    },
    "MISREAD_NEGATION": {
        "why": "\"keep it short\" matched inside \"don't keep it short\", so the request was read backwards.",
        "fix": "Add negated length phrasings ahead of the plain ones, so \"don't keep it short\" reads as wanting detail.",
        "kind": "rule",
        "rules": [("length", "detailed", r"(don'?t|do not|never|not) (keep|make|be)( \w+)? (it |this |that )?(short|brief|concise)"
                                          r"|not (so |too )?(short|brief)")],
    },
    "UNVERIFIED_SOURCE": {
        "why": "The answer cites a link the agent never fetched. It comes from the model's memory and may be invented.",
        "fix": "Remove the unverified link from the answer and say it was removed.",
        "kind": "redact",
    },
    "UNDER_ROUTED": {
        "why": "A real contradiction didn't move the answer up a tier.",
        "fix": "No automatic fix: this needs a decision about the escalation thresholds (0.4 / 0.7).",
        "kind": "manual",
    },
    "OVER_ROUTED": {
        "why": "An aligned user was sent to a stronger model than needed.",
        "fix": "No automatic fix: this needs a decision about the escalation thresholds (0.4 / 0.7).",
        "kind": "manual",
    },
}


def redact(answer: str, unverified: list) -> str:
    for link in unverified:
        answer = answer.replace(link, "[unverified link removed]")
    return answer


def review(code: str, say: str = "", answer: str = "", unverified: list | None = None) -> dict:
    """Why it failed, the proposed fix, and a zero-token check that the fix works."""
    pb = PLAYBOOK.get(code)
    if not pb:
        return {"code": code, "kind": "manual", "why": "Hope has no playbook entry for this code.",
                "fix": "Needs a human look.", "tokens": 0, "verified": None, "check": "", "applied": False}
    out = {"code": code, "kind": pb["kind"], "why": pb["why"], "fix": pb["fix"], "tokens": 0,
           "verified": None, "check": "", "applied": False}
    if pb["kind"] == "rule" and say:
        before = signals.SignalExtractor().extract(say)
        saved = {d: list(r) for d, r in signals.PATTERNS.items()}     # test the rules on a scratch copy
        try:
            for dim, value, pattern in pb["rules"]:
                signals.add_rule(dim, value, pattern)
            after = signals.SignalExtractor().extract(say)
        finally:
            signals.PATTERNS.clear()
            signals.PATTERNS.update(saved)
        out["verified"] = all(after.get(dim) == value for dim, value, _ in pb["rules"])
        out["check"] = f"Re-reading the message: before {before or 'no signal'}, with the fix {after or 'no signal'}."
    elif pb["kind"] == "redact":
        cleaned = redact(answer, unverified or [])
        left = find_unbacked_sources(cleaned)
        out["verified"] = bool(unverified) and not left
        out["check"] = f"Removed {len(unverified or [])} unverified link(s); {len(left)} left."
        out["fixed_answer"] = cleaned
    return out


def apply(db_path: str, code: str, persona: str = "") -> int:
    """Keep the rule fix: the signal reader uses it from now on, and it survives restarts. Returns rules added."""
    pb = PLAYBOOK.get(code)
    if not pb or pb["kind"] != "rule":
        return 0
    added = 0
    with closing(get_connection(db_path)) as conn:
        with conn:
            for dim, value, pattern in pb["rules"]:
                if conn.execute("SELECT 1 FROM lessons WHERE kind = 'rule' AND pattern = ?", (pattern,)).fetchone():
                    continue
                conn.execute("INSERT INTO lessons (code, kind, dimension, value, pattern, persona) "
                             "VALUES (?, 'rule', ?, ?, ?, ?)", (code, dim, value, pattern, persona))
                added += 1
    load_rules(db_path)
    return added


def record(db_path: str, persona: str, code: str, passed: bool) -> None:
    """One lesson per demo run: what happened to this persona, as a code. No message text is kept."""
    with closing(get_connection(db_path)) as conn:
        with conn:
            conn.execute("INSERT INTO lessons (code, kind, persona) VALUES (?, ?, ?)",
                         (code, "pass" if passed else "fail", persona))


def ledger(db_path: str, limit: int = 40) -> dict:
    """The shared lessons: per persona how often they passed and failed, the rules kept, and the latest entries."""
    with closing(get_connection(db_path)) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, code, kind, dimension, value, persona, created_at FROM lessons ORDER BY id DESC LIMIT ?", (limit,))]
        counts = conn.execute("SELECT persona, kind, COUNT(*) AS n FROM lessons WHERE persona != '' GROUP BY persona, kind").fetchall()
        rules = conn.execute("SELECT COUNT(*) FROM lessons WHERE kind = 'rule'").fetchone()[0]
    by = {}
    for c in counts:
        by.setdefault(c["persona"], {"pass": 0, "fail": 0, "rule": 0})[c["kind"]] = c["n"]
    return {"lessons": rows, "personas": by, "rules": rules}


def load_rules(db_path: str) -> int:
    """Put every kept rule into the reader (run at startup and after apply())."""
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute("SELECT dimension, value, pattern FROM lessons WHERE kind = 'rule' ORDER BY id").fetchall()
    for r in rows:
        signals.add_rule(r["dimension"], r["value"], r["pattern"])
    return len(rows)

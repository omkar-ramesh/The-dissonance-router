"""The note bot: plain-language notes on what the agent learned about you, and why.

Zero tokens by design. Notes are written from records the agent already keeps (an event
log of the moments that matter, plus feedback, routing decisions and saved conversations),
with fixed sentences. No model is called, so a note can't invent anything: every sentence
points at something that happened. Quotes from a conversation are erased when Memory
deletes that conversation.
"""
from contextlib import closing

from src.db.database import get_connection

LABEL = {
    ("length", "concise"): "short answers", ("length", "detailed"): "detailed answers",
    ("tone", "formal"): "a formal tone", ("tone", "casual"): "a casual tone",
    ("sourcing", "minimal"): "no sources", ("sourcing", "full"): "sources for claims",
    ("structure", "bullet"): "bullet points", ("structure", "paragraph"): "paragraphs",
    ("revision", "rejects"): "a second draft (first ones rarely fit)", ("revision", "accepts"): "ready-to-use answers",
}
REASON_WORDS = {"too_long": "too long", "too_short": "too short", "too_formal": "too formal", "too_casual": "too casual",
                "needs_sources": "missing sources", "too_many_sources": "too many sources", "use_bullets": "better as bullets",
                "use_paragraphs": "better as paragraphs", "wrong": "missing the point"}


def label(dim: str, value: str) -> str:
    return "no preference" if value == "balanced" else LABEL.get((dim, value), value)


class NoteBook:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def log(self, user_id: str, kind: str, dimension: str = "", value: str = "", detail: str = "",
            interaction_id=None) -> None:
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                conn.execute("INSERT INTO events (user_id, kind, dimension, value, detail, interaction_id) "
                             "VALUES (?, ?, ?, ?, ?, ?)", (user_id, kind, dimension, value, detail[:120], interaction_id))

    def forgotten_after(self, user_id: str) -> int:
        """Answers up to this id no longer count as evidence (the user asked to forget what was learned)."""
        with closing(get_connection(self.db_path)) as conn:
            row = conn.execute("SELECT MAX(interaction_id) FROM events WHERE user_id = ? AND kind = 'forgot'",
                               (user_id,)).fetchone()
        return row[0] or 0

    def render(self, user_id: str, state: dict) -> dict:
        """Notes, newest first, plus a summary of where things stand now (`state` = Agent.inspect)."""
        with closing(get_connection(self.db_path)) as conn:
            events = [dict(r) for r in conn.execute(
                "SELECT * FROM events WHERE user_id = ? ORDER BY id DESC LIMIT 60", (user_id,)).fetchall()]
            downs = conn.execute("SELECT reason, COUNT(*) AS n, MAX(created_at) AS at FROM feedback "
                                 "WHERE user_id = ? AND kind = 'down' AND reason != '' GROUP BY reason", (user_id,)).fetchall()
            tiers = conn.execute("SELECT model_used, COUNT(*) AS n, MAX(timestamp) AS at FROM interactions "
                                 "WHERE user_id = ? GROUP BY model_used", (user_id,)).fetchall()
            budget = conn.execute("SELECT COUNT(*) AS n, MAX(timestamp) AS at FROM interactions WHERE user_id = ? "
                                  "AND routing_reason LIKE 'Budget limit: the router chose%'", (user_id,)).fetchone()
            saved = conn.execute("SELECT COUNT(*) AS n, SUM(saved) AS kept, MAX(archived_at) AS at FROM conversations "
                                 "WHERE user_id = ?", (user_id,)).fetchone()

        notes = [self._sentence(e) for e in events]
        # Running totals: one note each, dated by the latest thing they count.
        for row in downs:
            notes.append({"at": row["at"], "kind": "feedback",
                          "text": f"You marked {_times(row['n'], 'answer')} as {REASON_WORDS.get(row['reason'], row['reason'])}."})
        counts = {r["model_used"]: (r["n"], r["at"]) for r in tiers}
        up = sum(counts.get(t, (0, None))[0] for t in ("super", "ultra"))
        if up:
            latest = max(counts[t][1] for t in ("super", "ultra") if t in counts)
            notes.append({"at": latest, "kind": "routing",
                          "text": f"{_times(up, 'answer')} moved up to Super or Ultra because of a contradiction; "
                                  f"{_times(counts.get('nano', (0, None))[0], 'answer')} stayed on Nano."})
        if budget["n"]:
            notes.append({"at": budget["at"], "kind": "budget",
                          "text": f"The budget stepped down a tier {_times(budget['n'], 'time')} to make the money last."})
        if saved["n"]:
            notes.append({"at": saved["at"], "kind": "memory",
                          "text": f"{_times(saved['n'], 'conversation')} in Memory, {saved['kept'] or 0} kept for good."})
        notes.sort(key=lambda n: _sortable(n["at"]), reverse=True)
        return {"summary": _summary(state), "notes": notes, "count": len(events) + len(downs) + bool(up)
                + bool(budget["n"]) + bool(saved["n"]), "tokens_used": 0}

    @staticmethod
    def _sentence(e: dict) -> dict:
        d, v, kind = e["dimension"], e["value"], e["kind"]
        quote = f" Latest sign: “{e['detail']}”." if e["detail"] else ""
        if kind == "said":
            text = (f"You said you prefer {label(d, v)}." if v != "balanced"
                    else f"You cleared your {d} preference.")
        elif kind == "learned":
            text = f"Learned: you keep wanting {label(d, v)}.{quote}"
        elif kind == "faded":
            text = f"Your {d} habit faded: no recent signs, so I stopped assuming {label(d, v)}."
        elif kind == "contradiction":
            said, did = (v.split(">") + [""])[:2]
            text = (f"Contradiction on {d}: you said {label(d, said)}, but you keep wanting {label(d, did)}. "
                    f"Answers moved up to {e['detail'] or 'a stronger model'} to serve both.")
        elif kind == "accepted":
            text = f"You agreed: your {d} preference is now {label(d, v)}. The contradiction is closed."
        elif kind == "forgot":
            text = "You asked me to forget what I'd learned about how you answer. Starting from what you say."
        else:
            text = e["detail"] or kind
        return {"at": e["created_at"], "kind": kind, "text": text}


def _summary(state: dict) -> list[str]:
    said, did = state["stated_profile"], state["observed_profile"]
    lines = []
    stated = [label(d, v) for d, v in said.items() if v != "balanced"]
    learned = [label(d, v) for d, v in did.items() if v != "balanced"]
    lines.append("You said you prefer " + _join(stated) + "." if stated else "You haven't said how you like answers yet.")
    lines.append("From what you do, I've learned you want " + _join(learned) + "." if learned
                 else "I haven't learned any habits yet: they need a few recent signs.")
    clash = [d for d, s in state["conflicts_detected"].items() if s > 0]
    if clash:
        lines.append(f"Open contradiction{'s' if len(clash) > 1 else ''}: {', '.join(clash)}. "
                     f"Next answer goes to {state['model_chosen'].title()}.")
    else:
        lines.append("What you say and what you do agree, so the next answer goes to Nano.")
    return lines


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _times(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _sortable(ts) -> str:
    """SQLite timestamps ('2026-10-28 10:00:00') and ISO ones ('2026-10-28T10:00:00+00:00') sort together."""
    return (ts or "").replace("T", " ")[:19]

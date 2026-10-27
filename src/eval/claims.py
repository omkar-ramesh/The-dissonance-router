"""False-claim checks: does what the agent says match what it recorded?

Two kinds of claim are checked automatically:
  - the agent's own explanations (routing reason, suggestion): every value they quote
    is compared with the database, read fresh with SQL rather than through the agent;
  - sources in a model answer: a link or DOI that nothing supplied is unverifiable.
    The agent has no search tool, so today any such link is unbacked.

What it can't check: whether a model's prose is factually true. That needs a reference
answer; see "Limits" in docs/EVAL.md.
"""
import re
import sqlite3

from src.agent.honesty import find_unbacked_sources  # noqa: F401  (re-exported: the eval and the agent share one rule)

_REASON = re.compile(
    r"Strongest dissonance: (?P<dim>\w+) \(you said '(?P<stated>\w+)', you behave '(?P<observed>\w+)', "
    r"seen (?P<times>\d+)x, score (?P<score>[\d.]+)\)")
_SUGGEST_SET = re.compile(r"You said you prefer (?P<dim>\w+) = '(?P<stated>\w+)', but you consistently behave as "
                          r"'(?P<observed>\w+)'")
_SUGGEST_NEW = re.compile(r"You never set a (?P<dim>\w+) preference, but you keep asking for '(?P<observed>\w+)'")
_TIER_PHRASES = {"Escalating to Ultra": "ultra", "Using Super": "super",
                 "Preferences aligned": "nano", "Quick request": "nano"}


def verify_explanations(db_path: str, user_id: str, routing_reason: str, suggestion) -> list:
    """Problems found in an explanation, as plain sentences. Empty = every claim checks out."""
    problems = []
    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        profile = con.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        last = con.execute("SELECT model_used, conflict_score FROM interactions WHERE user_id = ? "
                           "ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()

        def profile_value(side, dim):
            return profile[f"{side}_{dim}"] if profile and f"{side}_{dim}" in profile.keys() else None

        for phrase, tier in _TIER_PHRASES.items():
            if phrase in routing_reason and last and last["model_used"] != tier:
                problems.append(f"says '{phrase}' but the model used was {last['model_used']}")

        match = _REASON.search(routing_reason)
        if match:
            dim = match["dim"]
            for side in ("stated", "observed"):
                actual = profile_value(side, dim)
                if match[side] != actual:
                    problems.append(f"says {side} {dim} is '{match[side]}' but the profile has '{actual}'")
            seen = con.execute(
                "SELECT COUNT(*) FROM conflicts WHERE user_id = ? AND conflict_type = ? AND stated_value = ? "
                "AND observed_value = ? AND resolved = 0", (user_id, dim, match["stated"], match["observed"])
            ).fetchone()[0]
            if int(match["times"]) != seen:
                problems.append(f"says seen {match['times']}x but {seen} unresolved conflicts are recorded")
            if last and abs(float(match["score"]) - last["conflict_score"]) > 0.005:
                problems.append(f"says score {match['score']} but {last['conflict_score']:.2f} was recorded")

        if suggestion:
            new = _SUGGEST_NEW.search(suggestion)
            old = _SUGGEST_SET.search(suggestion)
            if new:
                dim = new["dim"]
                if profile_value("stated", dim) != "balanced":
                    problems.append(f"says no {dim} preference was set but it is '{profile_value('stated', dim)}'")
                if new["observed"] != profile_value("observed", dim):
                    problems.append(f"says observed {dim} is '{new['observed']}' but the profile differs")
            elif old:
                dim = old["dim"]
                for side in ("stated", "observed"):
                    if old[side] != profile_value(side, dim):
                        problems.append(f"suggestion says {side} {dim} is '{old[side]}' but the profile differs")
            else:
                problems.append("suggestion has a form the checker does not recognise")
    finally:
        con.close()
    return problems


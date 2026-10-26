"""Persists routing decisions (`interactions`) and contradiction history (`conflicts`)."""
from contextlib import closing

from src.db.database import get_connection


class DecisionLogger:
    """Writes the audit trail that the decision log UI shows."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def log_decision(
        self,
        user_id: str,
        request: str,
        model_used: str,
        conflict_score: float,
        reasoning: str,
        agent_response: str = "",
    ) -> None:
        """Record one routing decision. `agent_response` stays empty until Phase 2."""
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO interactions (user_id, user_message, agent_response, model_used, "
                    "conflict_detected, conflict_score, routing_reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, request, agent_response, model_used,
                     conflict_score > 0.0, conflict_score, reasoning),
                )

    def get_recent_decisions(self, user_id: str, limit: int = 5) -> list[dict]:
        """Return the user's latest decisions, newest first."""
        with closing(get_connection(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row, conflict_detected=bool(row["conflict_detected"])) for row in rows]

    def log_conflict(
        self, user_id: str, dimension: str, stated_value: str, observed_value: str, confidence: float
    ) -> None:
        """Record one sighting of a contradiction on a dimension."""
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO conflicts (user_id, stated_value, observed_value, conflict_type, "
                    "confidence) VALUES (?, ?, ?, ?, ?)",
                    (user_id, stated_value, observed_value, dimension, confidence),
                )

    def get_conflict_repeats(self, user_id: str, stated: dict, observed: dict) -> dict[str, int]:
        """Count unresolved prior sightings of each dimension's *current* contradiction.

        Only rows matching today's exact stated/observed pair count, so a contradiction
        that changes shape starts its escalation over.
        """
        with closing(get_connection(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT conflict_type, stated_value, observed_value, COUNT(*) AS n FROM conflicts "
                "WHERE user_id = ? AND resolved = 0 "
                "GROUP BY conflict_type, stated_value, observed_value",
                (user_id,),
            ).fetchall()
        return {
            row["conflict_type"]: row["n"]
            for row in rows
            if stated.get(row["conflict_type"]) == row["stated_value"]
            and observed.get(row["conflict_type"]) == row["observed_value"]
        }

    def resolve_conflicts(self, user_id: str, dimension: str) -> int:
        """Mark a dimension's open contradictions resolved. Returns rows updated."""
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                cursor = conn.execute(
                    "UPDATE conflicts SET resolved = 1 "
                    "WHERE user_id = ? AND conflict_type = ? AND resolved = 0",
                    (user_id, dimension),
                )
        return cursor.rowcount

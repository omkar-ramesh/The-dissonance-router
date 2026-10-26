"""Scores the gap between stated and observed preferences.

Per dimension:
  - stated == observed                  -> 0.0
  - opposite poles (concise vs detailed) -> 0.70, rising 0.80 -> 0.90 -> 0.95 as the
    same contradiction keeps reappearing
  - observed "balanced"                  -> 0.0: no behavioral evidence yet, so
    nothing can contradict the stated preference
  - stated "balanced", observed a pole   -> 0.40, rising to a cap of 0.65: a
    preference is emerging that the user never stated; Super, never Ultra
"""
from typing import Optional

from src.agent.profiles import DIMENSIONS, validate

# Indexed by how many times this exact contradiction was seen before (0 = first time).
OPPOSED_SCHEDULE = (0.70, 0.80, 0.90, 0.95)
PARTIAL_SCHEDULE = (0.40, 0.50, 0.60, 0.65)


class ConflictDetector:
    """Pure scoring logic; no database access."""

    def detect_conflicts(
        self, stated: dict, observed: dict, repeats: Optional[dict] = None
    ) -> dict:
        """Score every dimension and add `overall_score` (the average of all five).

        `repeats` maps dimension -> prior sightings of the same contradiction.
        """
        repeats = repeats or {}
        scores = {
            dim: self.calculate_conflict_score(stated[dim], observed[dim], dim, repeats.get(dim, 0))
            for dim in DIMENSIONS
        }
        scores["overall_score"] = round(sum(scores.values()) / len(DIMENSIONS), 4)
        return scores

    def calculate_conflict_score(
        self, stated_val: str, observed_val: str, dimension: str, repeats: int = 0
    ) -> float:
        """Score one dimension in [0, 1]. See module docstring for the rules."""
        validate(dimension, stated_val)
        validate(dimension, observed_val)
        if stated_val == observed_val or observed_val == "balanced":
            return 0.0
        schedule = PARTIAL_SCHEDULE if stated_val == "balanced" else OPPOSED_SCHEDULE
        return schedule[min(max(repeats, 0), len(schedule) - 1)]

    def get_highest_conflict(self, conflicts: dict) -> tuple[str, float]:
        """Return (dimension, score) of the strongest contradiction. Ties go to DIMENSIONS order."""
        dimension = max(DIMENSIONS, key=lambda dim: conflicts.get(dim, 0.0))
        return dimension, conflicts.get(dimension, 0.0)

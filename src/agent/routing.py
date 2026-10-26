"""Maps a dissonance score to a model tier."""
from typing import Optional

ULTRA_THRESHOLD = 0.7
SUPER_THRESHOLD = 0.4

REASONS = {
    "ultra": "High conflict detected. Escalating to Ultra for better reasoning.",
    "super": "Moderate conflict. Using Super for balanced approach.",
    "nano": "No significant conflict. Preferences aligned.",
}


class ModelRouter:
    """score >= 0.7 -> ultra, 0.4 <= score < 0.7 -> super, score < 0.4 -> nano."""

    def route(self, conflict_score: float) -> str:
        """Return "nano", "super" or "ultra"."""
        if not 0.0 <= conflict_score <= 1.0:
            raise ValueError(f"conflict_score must be in [0, 1], got {conflict_score}")
        if conflict_score >= ULTRA_THRESHOLD:
            return "ultra"
        if conflict_score >= SUPER_THRESHOLD:
            return "super"
        return "nano"

    def get_routing_reason(self, conflict_score: float, detail: Optional[str] = None) -> str:
        """Explain the routing choice, optionally followed by a specific detail."""
        reason = REASONS[self.route(conflict_score)]
        return f"{reason} {detail}" if detail else reason

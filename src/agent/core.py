"""Agent: ties profiles, conflict detection, routing and logging together."""
from typing import Optional

from src.agent.conflict_detector import ConflictDetector
from src.agent.llm_client import LLMClient, LLMError
from src.agent.logger import DecisionLogger
from src.agent.profiles import DIMENSIONS, ProfileManager
from src.agent.prompts import build_messages
from src.agent.routing import ULTRA_THRESHOLD, ModelRouter
from src.agent.signals import SignalExtractor
from src.db.database import init_db

# How many recent user messages vote on the observed profile.
EVIDENCE_WINDOW = 10
# How many prior turns are replayed to the model as conversation context.
HISTORY_TURNS = 4


class Agent:
    """Decides which model tier should answer a request, answers it, and records why."""

    def __init__(self, db_path: str, llm: Optional[LLMClient] = None):
        init_db(db_path)
        self.profiles = ProfileManager(db_path)
        self.detector = ConflictDetector()
        self.router = ModelRouter()
        self.logger = DecisionLogger(db_path)
        self.signals = SignalExtractor()
        self.llm = llm

    def analyze_request(self, user_id: str, user_request: str) -> dict:
        """Score, route and log a request without calling a model."""
        analysis = self._analyze(user_id, user_request)
        self._record(analysis)
        return analysis

    def respond(self, user_id: str, message: str) -> dict:
        """Full loop: learn from the message, route, answer with the chosen tier, log.

        Returns the analysis plus `response`, `model_id`, `latency_ms`, `usage` and
        `observed_updates` (dimensions this message changed).
        """
        if self.llm is None:
            raise RuntimeError("Agent was created without an LLM client")

        observed_updates = self.observe(user_id, message)
        analysis = self._analyze(user_id, message)
        history = self.logger.get_recent_decisions(user_id, limit=HISTORY_TURNS)
        try:
            result = self.llm.chat(analysis["model_chosen"], build_messages(analysis, history, message))
        except LLMError:
            # The observed profile may already have changed because of this message, so
            # keep it as evidence even though the model failed (empty response = not replayed).
            self._record(analysis)
            raise
        self._record(analysis, result.text)

        return {
            **analysis,
            "response": result.text,
            "model_id": result.model,
            "latency_ms": result.latency_ms,
            "usage": result.usage,
            "stats": result.stats(),
            "observed_updates": observed_updates,
        }

    def observe(self, user_id: str, message: str) -> dict:
        """Update the observed profile from recent messages plus this one.

        Returns {dimension: new_value} for every dimension that changed.
        """
        recent = [d["user_message"] for d in self.logger.get_recent_decisions(user_id, EVIDENCE_WINDOW - 1)]
        inferred = self.signals.infer_profile(recent + [message])
        current = self.profiles.get_observed_profile(user_id)

        changed = {}
        for dim, value in inferred.items():
            if current[dim] != value:
                self.profiles.update_observed(user_id, dim, value)
                changed[dim] = value
        return changed

    def accept_observed(self, user_id: str, dimension: str) -> None:
        """User agreed: adopt observed behavior as the stated preference and close the conflict."""
        observed = self.profiles.get_observed_profile(user_id)[dimension]
        self.profiles.update_stated(user_id, dimension, observed)
        self.logger.resolve_conflicts(user_id, dimension)

    def _analyze(self, user_id: str, user_request: str) -> dict:
        """Score the user's dissonance and pick a tier. No side effects.

        Routing uses the strongest single dimension, not the average: one strong
        contradiction (e.g. "concise" vs "detailed") averages to 0.14 across five
        dimensions and would never escalate. `overall_score` is still reported.
        """
        stated, observed = self.profiles.get_all_profiles(user_id)
        repeats = self.logger.get_conflict_repeats(user_id, stated, observed)
        conflicts = self.detector.detect_conflicts(stated, observed, repeats)
        dimension, score = self.detector.get_highest_conflict(conflicts)

        model = self.router.route(score)
        highest = None
        detail = None
        if score > 0:
            highest = {
                "dimension": dimension,
                "stated": stated[dimension],
                "observed": observed[dimension],
                "score": score,
                "times_seen": repeats.get(dimension, 0) + 1,
            }
            detail = (
                f"Strongest dissonance: {dimension} (you said '{highest['stated']}', "
                f"you behave '{highest['observed']}', seen {highest['times_seen']}x, "
                f"score {score:.2f})."
            )

        suggestion = None
        if score >= ULTRA_THRESHOLD:
            suggestion = (
                f"You said you prefer {dimension} = '{stated[dimension]}', but you consistently "
                f"behave as '{observed[dimension]}'. Update your stated preference?"
            )

        return {
            "user_id": user_id,
            "request": user_request,
            "model_chosen": model,
            "conflict_score": score,
            "overall_score": conflicts["overall_score"],
            "routing_reason": self.router.get_routing_reason(score, detail),
            "stated_profile": stated,
            "observed_profile": observed,
            "conflicts_detected": {dim: conflicts[dim] for dim in DIMENSIONS},
            "highest_conflict": highest,
            "suggestion": suggestion,
        }

    def _record(self, analysis: dict, agent_response: str = "") -> None:
        """Persist the decision and every contradiction seen in it."""
        user_id = analysis["user_id"]
        stated, observed = analysis["stated_profile"], analysis["observed_profile"]
        for dim, score in analysis["conflicts_detected"].items():
            if score > 0:
                self.logger.log_conflict(user_id, dim, stated[dim], observed[dim], score)
        self.logger.log_decision(
            user_id,
            analysis["request"],
            analysis["model_chosen"],
            analysis["conflict_score"],
            analysis["routing_reason"],
            agent_response,
        )

"""Phase 1: core logic with mock profiles and a throwaway SQLite file. No LLM calls."""
import pytest

from src.agent.conflict_detector import ConflictDetector
from src.agent.core import Agent
from src.agent.logger import DecisionLogger
from src.agent.profiles import DIMENSIONS, ProfileManager
from src.agent.routing import ModelRouter
from src.db.database import init_db

STATED = {"tone": "formal", "length": "concise", "sourcing": "minimal",
          "structure": "bullet", "revision": "rejects"}
OBSERVED = {"tone": "casual", "length": "detailed", "sourcing": "full",
            "structure": "paragraph", "revision": "accepts"}
BALANCED = {dim: "balanced" for dim in DIMENSIONS}


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def set_profile(profiles, user_id, stated=None, observed=None):
    for dim, value in (stated or {}).items():
        profiles.update_stated(user_id, dim, value)
    for dim, value in (observed or {}).items():
        profiles.update_observed(user_id, dim, value)


class TestModelRouter:
    @pytest.mark.parametrize("score, expected", [
        (0.85, "ultra"), (0.55, "super"), (0.20, "nano"),
        (0.70, "ultra"), (0.40, "super"),  # boundaries
        (0.6999, "super"), (0.3999, "nano"), (0.0, "nano"), (1.0, "ultra"),
    ])
    def test_route(self, score, expected):
        assert ModelRouter().route(score) == expected

    def test_reason_matches_tier(self):
        router = ModelRouter()
        assert "Ultra" in router.get_routing_reason(0.8)
        assert "Super" in router.get_routing_reason(0.5)
        assert "aligned" in router.get_routing_reason(0.1)

    def test_reason_appends_detail(self):
        assert ModelRouter().get_routing_reason(0.8, "Because X.").endswith("Because X.")

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_rejects_out_of_range(self, score):
        with pytest.raises(ValueError):
            ModelRouter().route(score)


class TestConflictDetector:
    def test_match_scores_zero(self):
        assert ConflictDetector().calculate_conflict_score("detailed", "detailed", "length") == 0.0

    def test_opposite_poles_score_point_seven(self):
        assert ConflictDetector().calculate_conflict_score("formal", "casual", "tone") == 0.7

    def test_unstated_preference_with_behavior_is_partial_gap(self):
        assert ConflictDetector().calculate_conflict_score("balanced", "detailed", "length") == 0.4

    def test_no_behavioral_evidence_is_no_conflict(self):
        assert ConflictDetector().calculate_conflict_score("concise", "balanced", "length") == 0.0

    def test_repeats_escalate_and_cap(self):
        detector = ConflictDetector()
        scores = [detector.calculate_conflict_score("concise", "detailed", "length", n) for n in range(6)]
        assert scores == [0.7, 0.8, 0.9, 0.95, 0.95, 0.95]

    def test_partial_gap_never_reaches_ultra(self):
        score = ConflictDetector().calculate_conflict_score("balanced", "detailed", "length", 50)
        assert score == 0.65
        assert ModelRouter().route(score) == "super"

    def test_overall_is_average_of_five(self):
        observed = dict(STATED, tone="casual", length="detailed")
        result = ConflictDetector().detect_conflicts(STATED, observed, {"length": 1})
        # (0.70 + 0.80 + 0 + 0 + 0) / 5
        assert result["overall_score"] == pytest.approx(0.30)
        assert ConflictDetector().get_highest_conflict(result) == ("length", 0.8)

    def test_full_mock_mismatch(self):
        result = ConflictDetector().detect_conflicts(STATED, OBSERVED)
        assert all(result[dim] == 0.7 for dim in DIMENSIONS)
        assert result["overall_score"] == pytest.approx(0.7)

    def test_single_strong_conflict_still_escalates(self):
        """The headline case: average is low, but the strongest dimension drives routing."""
        detector = ConflictDetector()
        result = detector.detect_conflicts(STATED, dict(STATED, length="detailed"))
        assert result["overall_score"] == pytest.approx(0.14)
        dimension, score = detector.get_highest_conflict(result)
        assert (dimension, score) == ("length", 0.7)
        assert ModelRouter().route(score) == "ultra"

    def test_no_conflict_highest_is_zero(self):
        result = ConflictDetector().detect_conflicts(STATED, STATED)
        assert ConflictDetector().get_highest_conflict(result)[1] == 0.0

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            ConflictDetector().calculate_conflict_score("concise", "verbose", "length")


class TestProfileManager:
    def test_new_user_defaults_to_balanced(self, db_path):
        assert ProfileManager(db_path).get_all_profiles("nobody") == (BALANCED, BALANCED)

    def test_store_and_retrieve_stated(self, db_path):
        profiles = ProfileManager(db_path)
        set_profile(profiles, "u1", stated=STATED)
        assert profiles.get_stated_profile("u1") == STATED
        assert profiles.get_observed_profile("u1") == BALANCED

    def test_update_observed(self, db_path):
        profiles = ProfileManager(db_path)
        set_profile(profiles, "u1", stated=STATED)
        assert profiles.update_observed("u1", "length", "detailed") is True
        assert profiles.get_observed_profile("u1")["length"] == "detailed"
        assert profiles.get_stated_profile("u1") == STATED

    def test_users_are_isolated(self, db_path):
        profiles = ProfileManager(db_path)
        profiles.update_stated("u1", "tone", "formal")
        assert profiles.get_stated_profile("u2")["tone"] == "balanced"

    @pytest.mark.parametrize("dimension, value", [("mood", "happy"), ("length", "verbose")])
    def test_invalid_input_raises(self, db_path, dimension, value):
        with pytest.raises(ValueError):
            ProfileManager(db_path).update_stated("u1", dimension, value)


class TestDecisionLogger:
    def test_log_and_read_back(self, db_path):
        logger = DecisionLogger(db_path)
        logger.log_decision("u1", "Draft an email", "ultra", 0.8, "because")
        logger.log_decision("u1", "Hi", "nano", 0.0, "aligned")
        newest, oldest = logger.get_recent_decisions("u1")
        assert (newest["user_message"], newest["model_used"], newest["conflict_detected"]) == ("Hi", "nano", False)
        assert (oldest["model_used"], oldest["conflict_score"], oldest["conflict_detected"]) == ("ultra", 0.8, True)
        assert oldest["agent_response"] == ""
        assert oldest["timestamp"]

    def test_recent_respects_limit(self, db_path):
        logger = DecisionLogger(db_path)
        for i in range(7):
            logger.log_decision("u1", f"m{i}", "nano", 0.0, "r")
        recent = logger.get_recent_decisions("u1", limit=5)
        assert [d["user_message"] for d in recent] == ["m6", "m5", "m4", "m3", "m2"]

    def test_repeats_count_only_current_contradiction(self, db_path):
        logger = DecisionLogger(db_path)
        logger.log_conflict("u1", "length", "concise", "detailed", 0.7)
        logger.log_conflict("u1", "length", "concise", "detailed", 0.8)
        logger.log_conflict("u1", "tone", "formal", "balanced", 0.4)
        stated = dict(BALANCED, length="concise", tone="formal")
        observed = dict(BALANCED, length="detailed", tone="casual")  # tone pair changed
        assert logger.get_conflict_repeats("u1", stated, observed) == {"length": 2}

    def test_resolve_resets_repeats(self, db_path):
        logger = DecisionLogger(db_path)
        logger.log_conflict("u1", "length", "concise", "detailed", 0.7)
        assert logger.resolve_conflicts("u1", "length") == 1
        stated = dict(BALANCED, length="concise")
        observed = dict(BALANCED, length="detailed")
        assert logger.get_conflict_repeats("u1", stated, observed) == {}


class TestAgent:
    def test_aligned_user_routes_nano(self, db_path):
        result = Agent(db_path).analyze_request("new", "Hello")
        assert result["model_chosen"] == "nano"
        assert result["conflict_score"] == 0.0
        assert result["highest_conflict"] is None
        assert result["suggestion"] is None

    def test_result_shape(self, db_path):
        result = Agent(db_path).analyze_request("u1", "Draft an email")
        assert set(result) >= {"user_id", "request", "model_chosen", "conflict_score",
                               "routing_reason", "stated_profile", "observed_profile",
                               "conflicts_detected"}
        assert set(result["conflicts_detected"]) == set(DIMENSIONS)

    def test_full_mock_profile_routes_ultra(self, db_path):
        agent = Agent(db_path)
        set_profile(agent.profiles, "u1", STATED, OBSERVED)
        result = agent.analyze_request("u1", "Draft an email")
        assert result["model_chosen"] == "ultra"
        assert result["overall_score"] == pytest.approx(0.7)

    def test_headline_scenario_escalates_over_time(self, db_path):
        """Says concise, behaves detailed: Ultra immediately, and confidence grows."""
        agent = Agent(db_path)
        set_profile(agent.profiles, "u1", {"length": "concise"}, {"length": "detailed"})
        scores = [agent.analyze_request("u1", f"request {i}")["conflict_score"] for i in range(4)]
        assert scores == [0.7, 0.8, 0.9, 0.95]

        result = agent.analyze_request("u1", "one more")
        assert result["model_chosen"] == "ultra"
        assert result["highest_conflict"]["times_seen"] == 5
        assert "length" in result["routing_reason"] and "concise" in result["routing_reason"]
        assert "Update your stated preference" in result["suggestion"]

    def test_partial_gap_routes_super(self, db_path):
        agent = Agent(db_path)
        set_profile(agent.profiles, "u1", observed={"length": "detailed"})
        result = agent.analyze_request("u1", "hi")
        assert result["model_chosen"] == "super"
        assert "Set it as your preference?" in result["suggestion"]   # guideline 5: a way out at Super too

    def test_accepting_observed_resolves_conflict(self, db_path):
        agent = Agent(db_path)
        set_profile(agent.profiles, "u1", {"length": "concise"}, {"length": "detailed"})
        agent.analyze_request("u1", "first")
        agent.accept_observed("u1", "length")
        assert agent.profiles.get_stated_profile("u1")["length"] == "detailed"
        assert agent.analyze_request("u1", "second")["model_chosen"] == "nano"

    def test_every_request_is_logged(self, db_path):
        agent = Agent(db_path)
        set_profile(agent.profiles, "u1", {"length": "concise"}, {"length": "detailed"})
        agent.analyze_request("u1", "first")
        agent.analyze_request("u1", "second")
        decisions = agent.logger.get_recent_decisions("u1")
        assert [d["user_message"] for d in decisions] == ["second", "first"]
        assert decisions[0]["model_used"] == "ultra"

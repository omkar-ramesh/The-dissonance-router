"""One test group per guideline in docs/GUIDELINES.md. No network."""
import pytest

from src.agent.core import Agent
from src.agent.llm_client import LLMResult
from src.agent.prompts import build_messages
from src.agent.relevance import is_light
from src.agent.signals import SignalExtractor
from src.db.database import init_db


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, tier, messages, temperature=0.7):
        self.calls.append((tier, messages))
        return LLMResult(text=f"answer from {tier}", tier=tier, model=f"{tier}-model", latency_ms=5)


@pytest.fixture
def agent(tmp_path):
    path = str(tmp_path / "g.db")
    init_db(path)
    return Agent(path, FakeLLM())


def concise_but_detailed(agent, user="u1"):
    """Stated concise, then two requests for detail: an open contradiction on length."""
    agent.profiles.update_stated(user, "length", "concise")
    agent.respond(user, "Explain attention, in more detail please")
    return agent.respond(user, "Can you elaborate on the softmax?")


class TestG1WordsAreClaimsBehaviorIsEvidence:
    def test_answer_follows_behavior(self, agent):
        concise_but_detailed(agent)
        system = agent.llm.calls[-1][1][0]["content"]
        assert "Be thorough and detailed." in system      # behavior shapes the answer

    def test_words_are_never_overwritten_by_behavior(self, agent):
        concise_but_detailed(agent)
        assert agent.profiles.get_stated_profile("u1")["length"] == "concise"   # only the user changes this


class TestG2WeakSignalsTeachNothingAndOldOnesFade:
    def test_two_signals_far_apart_are_not_a_habit(self):
        messages = ["elaborate please"] + ["what time is it?"] * 8 + ["go deeper"]
        assert SignalExtractor().infer_profile(messages) == {}

    def test_two_recent_signals_are_a_habit(self):
        assert SignalExtractor().infer_profile(["elaborate please", "go deeper"]) == {"length": "detailed"}

    def test_recent_behavior_outweighs_older(self):
        messages = ["elaborate", "tell me more", "make it shorter", "too long"]
        assert SignalExtractor().infer_profile(messages) == {"length": "concise"}

    def test_a_habit_fades_when_it_stops_showing_up(self, agent):
        agent.respond("u1", "elaborate on tokenizers")
        assert agent.respond("u1", "go deeper on BPE")["observed_updates"] == {"length": "detailed"}
        faded = {}
        for i in range(12):
            faded.update(agent.respond("u1", f"Explain topic number {i} to me")["observed_updates"])
        assert faded == {"length": "balanced"}
        assert agent.profiles.get_observed_profile("u1")["length"] == "balanced"


class TestG3ThisTurnsInstructionWinsForThisTurn:
    def test_one_off_instruction_is_obeyed_now(self, agent):
        concise_but_detailed(agent)                                  # profile says: detailed
        agent.respond("u1", "Explain recursion, keep it short")
        system = agent.llm.calls[-1][1][0]["content"]
        assert "Keep the answer short." in system and "Be thorough" not in system
        assert "Follow any explicit instruction" in system

    def test_one_off_instruction_is_not_learned(self, agent):
        agent.respond("u1", "Explain recursion, make it less formal")
        assert agent.profiles.get_observed_profile("u1")["tone"] == "balanced"

    def test_no_reconciling_when_this_message_sides_with_what_they_said(self, agent):
        concise_but_detailed(agent)
        result = agent.respond("u1", "Explain dropout, keep it short")
        assert result["model_chosen"] == "ultra"     # the contradiction is real...
        last_user_turn = agent.llm.calls[-1][1][-1]["content"]
        assert "TL;DR" not in last_user_turn         # ...but right now they're doing what they said

    def test_still_reconciles_when_this_message_asks_for_the_observed_side(self, agent):
        """Regression caught by the live run: 'elaborate' is the escalating message itself.
        TL;DR-then-depth still gives the depth asked for, so the reconciliation stays."""
        concise_but_detailed(agent)
        last_user_turn = agent.llm.calls[-1][1][-1]["content"]
        assert "TL;DR" in last_user_turn


class TestG4SpendPowerWhereItMatters:
    @pytest.mark.parametrize("message", ["hi", "thanks!", "ok got it", "Thank you so much", "What is 2+2?",
                                         "Name one primary color", "bye"])
    def test_light_messages(self, message):
        assert is_light(message)

    @pytest.mark.parametrize("message", ["Explain recursion", "Why is the sky blue?", "How do I reverse a list?",
                                         "Write a cover letter for a data analyst role",
                                         "ok and what about positional encoding?"])
    def test_substantive_messages(self, message):
        assert not is_light(message)

    def test_quick_message_stays_on_nano_despite_open_conflict(self, agent):
        concise_but_detailed(agent)
        result = agent.respond("u1", "thanks!")
        assert result["model_chosen"] == "nano"
        assert "Quick" in result["routing_reason"]
        assert agent.inspect("u1")["model_chosen"] == "ultra"     # the conflict is still open

    def test_quick_message_does_not_escalate_confidence(self, agent):
        concise_but_detailed(agent)
        agent.respond("u1", "thanks!")
        assert agent.respond("u1", "Why does layer norm help?")["conflict_score"] == 0.8

    def test_short_style_request_is_never_skipped(self, agent):
        agent.profiles.update_stated("u1", "length", "concise")
        agent.respond("u1", "elaborate")
        assert agent.respond("u1", "go deeper")["model_chosen"] == "ultra"


class TestG5ExplainEveryChoiceNeverChangeSilently:
    def test_every_answer_has_a_reason(self, agent):
        for message in ["hi", "Explain recursion", "elaborate", "go deeper"]:
            assert agent.respond("u1", message)["routing_reason"]

    def test_suggestion_offered_at_super_too(self, agent):
        agent.respond("u1", "elaborate on tokenizers")
        result = agent.respond("u1", "go deeper on BPE")
        assert result["model_chosen"] == "super"
        assert "never set a length preference" in result["suggestion"]

    def test_accept_can_be_undone(self, agent):
        concise_but_detailed(agent)
        previous = agent.accept_observed("u1", "length")
        assert previous == "concise"
        assert agent.profiles.get_stated_profile("u1")["length"] == "detailed"
        agent.profiles.update_stated("u1", "length", previous)          # what the Undo button does
        assert agent.profiles.get_stated_profile("u1")["length"] == "concise"


class TestG6HonestAndPrivate:
    def test_every_prompt_asks_for_honesty(self, agent):
        agent.respond("u1", "hi")
        system = agent.llm.calls[-1][1][0]["content"]
        assert "say so" in system and "Never invent sources" in system

    def test_honesty_rule_present_on_every_tier(self):
        base = {"stated_profile": {d: "balanced" for d in ("tone", "length", "sourcing", "structure", "revision")},
                "conflicts_detected": {d: 0.0 for d in ("tone", "length", "sourcing", "structure", "revision")}}
        base["observed_profile"] = dict(base["stated_profile"])
        for tier in ("nano", "super", "ultra"):
            system = build_messages({**base, "model_chosen": tier}, [], "q")[0]["content"]
            assert "Never invent sources" in system

"""Behavior signals (feedback, regenerate), the Nano signal reader, reasoning on/off and the
public demo guard. No network."""
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import config
from src.agent.core import Agent
from src.agent.extractor import NanoExtractor, parse
from src.agent.llm_client import LLMClient, LLMError, LLMResult
from src.api import limits
from src.api.dependencies import get_agent
from src.main import app


class FakeLLM:
    def __init__(self, replies=None):
        self.calls, self.replies = [], list(replies or [])

    def chat(self, tier, messages, temperature=0.7, **kwargs):
        self.calls.append({"tier": tier, "messages": messages, **kwargs})
        text = self.replies.pop(0) if self.replies else f"answer {len(self.calls)} from {tier}"
        return LLMResult(text=text, tier=tier, model=f"{tier}-model", latency_ms=5)


@pytest.fixture
def agent(tmp_path):
    return Agent(str(tmp_path / "b.db"), FakeLLM())


# ---------------------------------------------------------------- the Nano reader

class TestParse:
    def test_reads_first_key_with_evidence(self):
        assert parse('{"length": ["detailed", "I still don\'t get it"]}', "I still don't get it") == {"length": "detailed"}

    def test_tokenizer_space_after_apostrophe_still_matches(self):
        assert parse('{"length": ["detailed", "I still don\' t get it"]}', "I still don't get it") == {"length": "detailed"}

    def test_only_the_first_key_counts(self):
        reply = '{"length": ["detailed", "more detail"], "tone": ["formal", "Explain"]}'
        assert parse(reply, "Explain X, more detail please") == {"length": "detailed"}

    def test_quote_must_be_in_the_message(self):
        assert parse('{"tone": ["casual", "talk to me like a friend"]}', "explain like ur my buddy") == {}

    def test_quoted_words_are_someone_elses(self):
        msg = 'My manager says "keep it short". Explain Kafka.'
        assert parse('{"length": ["concise", "keep it short"]}', msg) == {}
        assert parse('{"length": ["concise", "My manager says  "keep it short""]}', msg) == {}   # broken JSON too

    def test_value_must_be_a_pole(self):
        assert parse('{"length": ["balanced", "explain"]}', "explain") == {}
        assert parse('{"length": ["verbose", "explain"]}', "explain") == {}

    def test_no_evidence_no_signal(self):
        assert parse('{"length": "detailed"}', "more detail") == {}
        assert parse("{}", "What is DNS?") == {}

    def test_no_json_raises_so_the_caller_falls_back(self):
        with pytest.raises(ValueError):
            parse("I think they want detail", "more detail")


class TestNanoExtractor:
    def test_reads_once_then_caches(self, tmp_path):
        llm = FakeLLM(['{"length": ["detailed", "lost"]}'])
        x = NanoExtractor(llm, str(tmp_path / "c.db"))
        Agent(str(tmp_path / "c.db"))                    # creates the tables
        assert x.extract("I'm lost") == {"length": "detailed"}
        assert x.extract("I'm lost") == {"length": "detailed"}
        assert len(llm.calls) == 1
        fresh = NanoExtractor(FakeLLM(), str(tmp_path / "c.db"))   # cache survives a restart
        assert fresh.extract("I'm lost") == {"length": "detailed"} and not fresh.llm.calls

    def test_asks_with_reasoning_off_and_a_small_cap(self):
        llm = FakeLLM(["{}"])
        NanoExtractor(llm).extract("What is DNS?")
        assert llm.calls[0]["reasoning"] == "off" and llm.calls[0]["max_tokens"] <= 100
        assert llm.calls[0]["tier"] == "nano"

    def test_falls_back_to_keywords_and_retries_later(self):
        class Down:
            def chat(self, *a, **k):
                raise LLMError("ollama is not running")
        x = NanoExtractor(Down())
        assert x.extract("please elaborate") == {"length": "detailed"}    # keyword reader answered
        assert x.fallbacks == 1 and not x._memo                           # not cached: Nano is asked next time

    def test_empty_message_costs_nothing(self):
        llm = FakeLLM()
        assert NanoExtractor(llm).extract("  ") == {} and not llm.calls

    def test_reader_calls_are_reported_per_turn(self, tmp_path):
        db = str(tmp_path / "d.db")
        llm = FakeLLM(["{}", "the answer"])
        a = Agent(db, llm, NanoExtractor(llm, db))
        result = a.respond("u", "What is DNS?")
        assert len(result["reader_stats"]) == 1 and result["response"] == "the answer"
        assert a.respond("u", "What is DNS?")["reader_stats"] == []          # cached: no new reading


# ---------------------------------------------------------------- behavior signals

class TestFeedback:
    def test_two_too_long_thumbs_down_make_a_concise_habit(self, agent):
        a = agent.respond("u", "Explain DNS")["interaction_id"]
        b = agent.respond("u", "Explain TCP")["interaction_id"]
        agent.feedback("u", a, "down", "too_long")
        result = agent.feedback("u", b, "down", "too_long")
        assert result["observed_updates"]["length"] == "concise"
        assert agent.profiles.get_observed_profile("u")["revision"] == "rejects"

    def test_thumbs_up_and_copy_mean_accepts(self, agent):
        a = agent.respond("u", "Explain DNS")["interaction_id"]
        b = agent.respond("u", "Explain TCP")["interaction_id"]
        agent.feedback("u", a, "up")
        agent.feedback("u", b, "copy")
        assert agent.profiles.get_observed_profile("u")["revision"] == "accepts"

    def test_behavior_beats_words_on_the_same_answer(self, agent):
        a = agent.respond("u", "Explain DNS in more detail")["interaction_id"]
        b = agent.respond("u", "Explain TCP in more detail")["interaction_id"]
        assert agent.profiles.get_observed_profile("u")["length"] == "detailed"
        agent.feedback("u", a, "down", "too_long")
        agent.feedback("u", b, "down", "too_long")
        assert agent.profiles.get_observed_profile("u")["length"] == "concise"

    def test_one_click_is_not_a_habit(self, agent):
        a = agent.respond("u", "Explain DNS")["interaction_id"]
        assert agent.feedback("u", a, "down", "too_long")["observed_updates"] == {}

    def test_cannot_touch_someone_elses_answer(self, agent):
        a = agent.respond("alice", "Explain DNS")["interaction_id"]
        with pytest.raises(LookupError):
            agent.feedback("mallory", a, "down", "too_long")

    @pytest.mark.parametrize("kind, reason", [("meh", ""), ("down", "boring"), ("up", "too_long")])
    def test_rejects_unknown_kinds_and_reasons(self, agent, kind, reason):
        a = agent.respond("u", "Explain DNS")["interaction_id"]
        with pytest.raises(ValueError):
            agent.feedback("u", a, kind, reason)


class TestRegenerate:
    def test_replaces_the_answer_in_place(self, agent):
        agent.respond("u", "Explain DNS")
        before = agent.logger.get_recent_decisions("u", 10)
        result = agent.regenerate("u")
        after = agent.logger.get_recent_decisions("u", 10)
        assert len(after) == len(before) == 1                    # same row: the words aren't counted twice
        assert after[0]["agent_response"] == result["response"] != before[0]["agent_response"]
        assert result["interaction_id"] == before[0]["id"]

    def test_asks_for_a_different_approach(self, agent):
        agent.respond("u", "Explain DNS")
        agent.regenerate("u")
        assert "different answer" in agent.llm.calls[-1]["messages"][-1]["content"]

    def test_two_regenerates_on_two_answers_mean_rejects(self, agent):
        agent.respond("u", "Explain DNS")
        agent.regenerate("u")
        agent.respond("u", "Explain TCP")
        assert agent.regenerate("u")["observed_updates"] == {"revision": "rejects"}

    def test_nothing_to_regenerate(self, agent):
        with pytest.raises(ValueError):
            agent.regenerate("nobody")


# ---------------------------------------------------------------- reasoning on/off

def _capture():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
    return seen, httpx.MockTransport(handler)


class TestReasoningSwitch:
    MODELS = {"nano": "n", "super": "s", "ultra": "u"}

    def client(self, transport, providers, max_tokens=None):
        return LLMClient("http://t/v1", self.MODELS, transport=transport, providers=providers,
                         reasoning={"nano": "off", "super": "off", "ultra": "on"}, max_tokens=max_tokens)

    def test_ollama_off_uses_reasoning_effort_none(self):
        seen, t = _capture()
        result = self.client(t, {"nano": "ollama"}).chat("nano", [{"role": "user", "content": "hi"}])
        assert seen[0]["reasoning_effort"] == "none" and result.reasoning_on is False

    def test_nebius_off_uses_the_chat_template_switch(self):
        seen, t = _capture()
        self.client(t, {"nano": "nebius"}).chat("nano", [{"role": "user", "content": "hi"}])
        assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_ultra_keeps_thinking(self):
        seen, t = _capture()
        result = self.client(t, {"ultra": "ollama"}).chat("ultra", [{"role": "user", "content": "hi"}])
        assert "reasoning_effort" not in seen[0] and result.reasoning_on is True

    def test_per_call_override_and_output_cap(self):
        seen, t = _capture()
        c = self.client(t, {"ultra": "ollama"}, max_tokens={"ultra": 4000})
        c.chat("ultra", [{"role": "user", "content": "hi"}], reasoning="off", max_tokens=50)
        c.chat("ultra", [{"role": "user", "content": "hi"}])
        assert seen[0]["reasoning_effort"] == "none" and seen[0]["max_tokens"] == 50
        assert seen[1]["max_tokens"] == 4000


# ---------------------------------------------------------------- public demo guard

class TestGuard:
    def test_rate_limit_per_ip(self):
        g = limits.Guard(per_minute=2, daily_budget_usd=10)
        g.admit("1.1.1.1"); g.admit("1.1.1.1")
        with pytest.raises(HTTPException) as e:
            g.admit("1.1.1.1")
        assert e.value.status_code == 429
        g.admit("2.2.2.2")                          # other visitors are unaffected

    def test_daily_budget(self):
        g = limits.Guard(per_minute=100, daily_budget_usd=0.01)
        g.admit("ip"); g.spend(0.02)
        with pytest.raises(HTTPException) as e:
            g.admit("ip")
        assert "budget" in e.value.detail

    def test_cost_counts_the_reader_too(self):
        assert limits.cost_of({"stats": {"est_cost_usd": 0.01}, "reader_stats": [{"est_cost_usd": 0.002}]}) == pytest.approx(0.012)


@pytest.fixture
def client(tmp_path):
    a = Agent(str(tmp_path / "api.db"), FakeLLM())
    app.dependency_overrides[get_agent] = lambda: a
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestApi:
    def test_feedback_and_regenerate_endpoints(self, client):
        r = client.post("/chat", json={"user_id": "u1", "message": "Explain DNS"}).json()
        fb = client.post("/feedback", json={"user_id": "u1", "interaction_id": r["interaction_id"], "kind": "up"})
        assert fb.status_code == 200 and "observed_profile" in fb.json()
        regen = client.post("/regenerate", json={"user_id": "u1"})
        assert regen.status_code == 200 and regen.json()["interaction_id"] == r["interaction_id"]
        log = client.get("/decision-log/u1").json()["decisions"]
        assert [f["kind"] for f in log[0]["feedback"]] == ["up", "regenerate"]

    def test_feedback_errors(self, client):
        r = client.post("/chat", json={"user_id": "u1", "message": "Explain DNS"}).json()
        assert client.post("/feedback", json={"user_id": "u2", "interaction_id": r["interaction_id"],
                                              "kind": "up"}).status_code == 404
        assert client.post("/feedback", json={"user_id": "u1", "interaction_id": r["interaction_id"],
                                              "kind": "down", "reason": "boring"}).status_code == 422
        assert client.post("/regenerate", json={"user_id": "nobody"}).status_code == 409

    def test_public_demo_refuses_guessable_ids(self, client, monkeypatch):
        monkeypatch.setattr(config, "PUBLIC_DEMO", True)
        monkeypatch.setattr(limits, "GUARD", limits.Guard(100, 100))
        assert client.get("/profiles/demo").status_code == 422
        assert client.get("/profiles/abcdefghijkl3456").status_code == 200
        assert client.post("/update-style", json={"user_id": "abcdefghijkl3456", "dimension": "length",
                                                  "value": "concise", "kind": "observed"}).status_code == 403
        assert client.get("/health").json()["public_demo"] is True

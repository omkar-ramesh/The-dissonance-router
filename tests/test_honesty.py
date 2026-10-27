"""Guideline 6 in the product: links the agent can't vouch for are flagged, not passed along. No network."""
import pytest
from fastapi.testclient import TestClient

from src.agent.core import Agent
from src.agent.honesty import find_unbacked_sources
from src.agent.llm_client import LLMResult
from src.api.dependencies import get_agent
from src.main import app


class Linky:
    def __init__(self, text):
        self.text = text

    def chat(self, tier, messages, temperature=0.7, **kw):
        return LLMResult(text=self.text, tier=tier, model="m", latency_ms=1)


def test_agent_flags_links_it_cannot_vouch_for(tmp_path):
    agent = Agent(str(tmp_path / "h.db"), Linky("See https://kafka.apache.org/documentation/ for more."))
    result = agent.respond("u", "Explain Kafka partitions")
    assert result["unverified_sources"] == ["https://kafka.apache.org/documentation/"]


def test_answers_without_links_are_clean(tmp_path):
    assert Agent(str(tmp_path / "h.db"), Linky("A partition is an ordered log.")).respond("u", "hi")["unverified_sources"] == []


def test_illustrations_are_not_flagged():
    assert find_unbacked_sources("Open https://example.com, then run `curl https://`.") == []


def test_replayed_history_is_flagged_too(tmp_path):
    agent = Agent(str(tmp_path / "h.db"), Linky("Read https://arxiv.org/abs/1706.03762"))
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        client = TestClient(app)
        client.post("/chat", json={"user_id": "u1", "message": "Who wrote the transformer paper?"})
        log = client.get("/decision-log/u1").json()["decisions"]
        assert log[0]["unverified_sources"] == ["https://arxiv.org/abs/1706.03762"]
    finally:
        app.dependency_overrides.clear()

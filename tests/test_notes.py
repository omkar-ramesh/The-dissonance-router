"""The note bot: notes from records, no model calls. No network."""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.agent.core import Agent
from src.agent.llm_client import LLMResult
from src.agent.memory import MemoryStore
from src.api.dependencies import get_agent
from src.main import app


class Counting:
    def __init__(self):
        self.calls = 0

    def chat(self, tier, messages, temperature=0.7, **kw):
        self.calls += 1
        return LLMResult(text="an answer", tier=tier, model=tier, latency_ms=1, input_tokens=100, output_tokens=100)


@pytest.fixture
def agent(tmp_path):
    return Agent(str(tmp_path / "n.db"), Counting(), memory_dir=str(tmp_path / "memory"))


def texts(agent, user="u"):
    return [n["text"] for n in agent.notes.render(user, agent.inspect(user))["notes"]]


def test_the_story_of_a_contradiction(agent):
    agent.set_stated("u", "length", "concise")
    agent.respond("u", "Explain DNS in more detail please")
    agent.respond("u", "Can you elaborate on caching?")
    agent.accept_observed("u", "length")
    notes = " | ".join(texts(agent))
    assert "You said you prefer short answers." in notes
    assert "Learned: you keep wanting detailed answers. Latest sign: “Can you elaborate on caching?”" in notes
    assert "Contradiction on length: you said short answers, but you keep wanting detailed answers. Answers moved up to Ultra" in notes
    assert "You agreed: your length preference is now detailed answers." in notes


def test_notes_cost_nothing(agent):
    agent.respond("u", "Explain DNS")
    before = agent.llm.calls
    rendered = agent.notes.render("u", agent.inspect("u"))
    assert agent.llm.calls == before and rendered["tokens_used"] == 0


def test_contradiction_is_noted_once_not_every_turn(agent):
    agent.set_stated("u", "length", "concise")
    for m in ("More detail please on DNS", "Elaborate on TTL", "Go deeper on caching", "Elaborate on resolvers"):
        agent.respond("u", m)
    assert sum(t.startswith("Contradiction") for t in texts(agent)) == 1


def test_behavior_is_noted_with_what_you_did(agent):
    a = agent.respond("u", "Explain DNS")["interaction_id"]
    b = agent.respond("u", "Explain TCP")["interaction_id"]
    agent.feedback("u", a, "down", "too_long")
    agent.feedback("u", b, "down", "too_long")
    notes = " | ".join(texts(agent))
    assert "You marked 2 answers as too long." in notes
    assert "Learned: you keep wanting short answers. Latest sign: “👎 too long”" in notes


def test_summary_says_where_things_stand(agent):
    assert agent.notes.render("u", agent.inspect("u"))["summary"][0] == "You haven't said how you like answers yet."
    agent.set_stated("u", "tone", "formal")
    summary = agent.notes.render("u", agent.inspect("u"))["summary"]
    assert summary[0] == "You said you prefer a formal tone." and "agree" in summary[2]


def test_setting_the_same_preference_twice_is_one_note(agent):
    agent.set_stated("u", "length", "concise")
    agent.set_stated("u", "length", "concise")
    assert texts(agent).count("You said you prefer short answers.") == 1


def test_deleting_a_conversation_erases_its_quotes_from_the_notes(agent):
    agent.respond("u", "Explain my secret project in more detail")
    agent.respond("u", "Elaborate on the secret part")
    item = agent.memory.archive("u")
    eleven_days_later = datetime.fromisoformat(item["archived_at"]) + timedelta(days=11)
    MemoryStore(agent.memory.db_path, agent.memory.folder, 10, now=lambda: eleven_days_later).sweep()
    notes = " | ".join(texts(agent))
    assert "secret" not in notes and "Learned: you keep wanting detailed answers." in notes


def test_notes_endpoint(tmp_path):
    a = Agent(str(tmp_path / "api.db"), Counting(), memory_dir=str(tmp_path / "memory"))
    app.dependency_overrides[get_agent] = lambda: a
    try:
        client = TestClient(app)
        client.post("/update-style", json={"user_id": "u1", "dimension": "length", "value": "concise"})
        body = client.get("/notes/u1").json()
        assert body["tokens_used"] == 0 and body["notes"][0]["text"] == "You said you prefer short answers."
        assert body["count"] >= 1
    finally:
        app.dependency_overrides.clear()


def test_forget_observed_clears_learned_habit_and_keeps_stated(agent):
    agent.profiles.update_stated("u", "length", "concise")
    for m in ("Explain attention, in more detail please", "Can you elaborate on the softmax?", "Go deeper on gradients"):
        agent.respond("u", m)
    assert agent.inspect("u")["observed_profile"]["length"] == "detailed"
    agent.forget_observed("u")
    state = agent.inspect("u")
    assert state["observed_profile"]["length"] == "balanced" and state["stated_profile"]["length"] == "concise"
    agent.respond("u", "What is a tensor?")                       # old turns no longer count
    assert agent.inspect("u")["observed_profile"]["length"] == "balanced"
    assert any("forget" in t for t in texts(agent))

"""Phase 3: HTTP API, with a fake model and a throwaway database."""
import pytest
from fastapi.testclient import TestClient

from src.agent.core import Agent
from src.agent.llm_client import LLMError, LLMResult
from src.api.dependencies import get_agent
from src.main import app


class FakeLLM:
    def __init__(self, fail=False):
        self.fail = fail

    def chat(self, tier, messages, temperature=0.7):
        if self.fail:
            raise LLMError("Could not reach the LLM backend for m: connection refused")
        return LLMResult(text=f"answer from {tier}", tier=tier, model=f"{tier}-model", latency_ms=5)


@pytest.fixture
def agent(tmp_path):
    return Agent(str(tmp_path / "api.db"), FakeLLM())


@pytest.fixture
def client(agent):
    app.dependency_overrides[get_agent] = lambda: agent
    yield TestClient(app)
    app.dependency_overrides.clear()


def chat(client, message, user="u1"):
    response = client.post("/chat", json={"user_id": user, "message": message})
    assert response.status_code == 200, response.text
    return response.json()


def test_home_serves_the_product_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "The Dissonance Router" in response.text


def test_proof_serves_the_eval_report_without_answers(client, monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"simulated": true, "scenarios": [{"id": "a", "turns": [{"say": "hi", "response": "long text"}]}]}')
    monkeypatch.setattr("src.api.routes.PROOF_SOURCES", [tmp_path / "missing.json", report])
    body = client.get("/proof").json()
    assert body["source"] == "report.json" and body["report"]["simulated"] is True
    assert "response" not in body["report"]["scenarios"][0]["turns"][0]      # prose stays out of the payload


def test_proof_says_so_when_there_is_no_report(client, monkeypatch, tmp_path):
    monkeypatch.setattr("src.api.routes.PROOF_SOURCES", [tmp_path / "nope.json"])
    monkeypatch.setattr("src.api.routes.NANO_SOURCES", [tmp_path / "nope2.json"])
    monkeypatch.setattr("src.api.routes.COMPARE_SOURCES", [tmp_path / "nope3.json"])
    body = client.get("/proof").json()
    assert body["report"] is None and body["source"] is None
    assert body["nano_report"] is None and body["compare"] is None


def test_the_committed_snapshot_is_valid():
    """The hosted demo falls back to docs/eval-report.json, so it must exist and parse."""
    import json
    from src.api.routes import PROOF_SOURCES
    report = json.loads(PROOF_SOURCES[-1].read_text(encoding="utf-8"))
    assert report["scenarios"] and report["strategies"] and report["self_test"]["total"]


def test_health_lists_tiers(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["tiers"]) == {"nano", "super", "ultra"}
    assert all({"model", "provider"} <= set(t) for t in body["tiers"].values())


def test_chat_returns_answer_tier_reason_and_stats(client):
    body = chat(client, "What is 2+2?")
    assert body["response"] == "answer from nano"
    assert body["model_chosen"] == "nano"
    assert body["routing_reason"]
    assert "stats" in body and body["stats"]["tier"] == "nano"


def test_demo_flow_over_http(client):
    """Say concise, keep asking for detail, get escalated, accept, back to Nano."""
    r = client.post("/update-style", json={"user_id": "u1", "dimension": "length", "value": "concise"})
    assert r.status_code == 200 and r.json()["stated_profile"]["length"] == "concise"

    assert chat(client, "Explain attention, in more detail please")["model_chosen"] == "nano"
    escalated = chat(client, "Can you elaborate on the softmax?")
    assert escalated["model_chosen"] == "ultra"
    assert "Update your stated preference" in escalated["suggestion"]

    accepted = client.post("/accept", json={"user_id": "u1", "dimension": "length"}).json()
    assert accepted["stated_profile"]["length"] == "detailed"
    assert accepted["model_chosen"] == "nano"
    assert chat(client, "thanks, next one")["model_chosen"] == "nano"


def test_profiles_are_read_only(client, agent):
    agent.profiles.update_stated("u1", "length", "concise")
    agent.profiles.update_observed("u1", "length", "detailed")
    first = client.get("/profiles/u1").json()
    second = client.get("/profiles/u1").json()
    assert first["model_chosen"] == "ultra"
    assert first["conflict_score"] == second["conflict_score"] == 0.7   # viewing doesn't escalate
    assert agent.logger.get_recent_decisions("u1") == []                # and isn't logged
    assert first["conflicts_detected"]["length"] == 0.7


def test_new_user_profile_is_balanced(client):
    body = client.get("/profiles/nobody").json()
    assert set(body["stated_profile"].values()) == {"balanced"}
    assert body["model_chosen"] == "nano" and body["suggestion"] is None


def test_decision_log_newest_first_with_utc_timestamps(client):
    for i in range(3):
        chat(client, f"message {i}")
    body = client.get("/decision-log/u1?limit=2").json()
    assert [d["user_message"] for d in body["decisions"]] == ["message 2", "message 1"]
    assert body["decisions"][0]["timestamp"].endswith("Z") and "T" in body["decisions"][0]["timestamp"]
    assert body["decisions"][0]["agent_response"] == "answer from nano"


@pytest.mark.parametrize("payload", [
    {"user_id": "u1", "dimension": "length", "value": "verbose"},     # bad value
    {"user_id": "u1", "dimension": "mood", "value": "happy"},          # bad dimension
    {"user_id": "u1", "dimension": "length", "value": "concise", "kind": "wished"},
    {"user_id": "../etc", "dimension": "length", "value": "concise"},  # unsafe id
])
def test_update_style_rejects_bad_input(client, payload):
    assert client.post("/update-style", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {"user_id": "u1", "message": ""},
    {"user_id": "", "message": "hi"},
    {"user_id": "u 1", "message": "hi"},
    {"user_id": "u1"},
])
def test_chat_rejects_bad_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_accept_rejects_unknown_dimension(client):
    assert client.post("/accept", json={"user_id": "u1", "dimension": "mood"}).status_code == 422


def test_decision_log_limit_bounds(client):
    assert client.get("/decision-log/u1?limit=0").status_code == 422
    assert client.get("/decision-log/u1?limit=101").status_code == 422


def test_model_outage_is_503_and_message_is_kept(tmp_path):
    down = Agent(str(tmp_path / "down.db"), FakeLLM(fail=True))
    app.dependency_overrides[get_agent] = lambda: down
    try:
        response = TestClient(app).post("/chat", json={"user_id": "u1", "message": "hello"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "Could not reach" in response.json()["detail"]
    assert [d["user_message"] for d in down.logger.get_recent_decisions("u1")] == ["hello"]

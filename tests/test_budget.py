"""The simulated wallet: the router plans within a $ budget. No network."""
import pytest
from fastapi.testclient import TestClient

from src import config
from src.agent import budget
from src.agent.budget import BudgetExceeded, cost, plan
from src.agent.core import Agent
from src.agent.llm_client import LLMResult
from src.api.dependencies import get_agent
from src.eval.claims import verify_explanations
from src.main import app

P = config.PRICES


class Spender:
    """Uses every token it is allowed: the worst case for the budget."""

    def __init__(self):
        self.calls = []

    def chat(self, tier, messages, temperature=0.7, **kw):
        self.calls.append({"tier": tier, **kw})
        out = kw.get("max_tokens") or 300
        return LLMResult(text="answer " * 10, tier=tier, model=tier, latency_ms=1, input_tokens=200, output_tokens=out)


class TestPlan:
    def test_no_limit_changes_nothing(self):
        p = plan("ultra", 200, 0, 0, P)
        assert p.tier == "ultra" and not p.limited and p.chat_kwargs == {} and p.reason is None

    def test_steps_down_when_a_tier_would_eat_too_much_of_what_is_left(self):
        p = plan("ultra", 200, limit=0.01, spent=0, prices=P)
        assert p.tier == "super"
        assert "Ultra" in p.reason and "Super answered" in p.reason

    def test_keeps_the_tier_when_affordable(self):
        p = plan("ultra", 200, limit=1.0, spent=0, prices=P)
        assert p.tier == "ultra" and p.reason is None and p.limited

    def test_tight_budget_turns_thinking_off_and_shortens(self):
        p = plan("nano", 200, limit=0.01, spent=0.0085, prices=P)
        assert p.chat_kwargs["reasoning"] == "off" and p.max_tokens <= budget.TIGHT_CAP["nano"]
        assert "thinking is off" in p.reason

    def test_answer_can_never_cost_more_than_what_is_left(self):
        remaining = 0.00005
        p = plan("nano", 100, limit=1.0, spent=1.0 - remaining, prices=P)
        assert cost(P, p.tier, 100, p.max_tokens) <= remaining + 1e-12

    def test_nothing_left_says_so(self):
        with pytest.raises(BudgetExceeded):
            plan("nano", 200, limit=0.01, spent=0.01, prices=P)
        with pytest.raises(BudgetExceeded):                         # too little for a useful answer
            plan("nano", 200, limit=1.0, spent=1.0 - 0.000001, prices=P)


def contradiction(agent, user="u"):
    agent.profiles.update_stated(user, "length", "concise")
    agent.respond(user, "Explain attention in more detail please")


class TestAgent:
    def test_downgrade_is_logged_honestly(self, tmp_path):
        db = str(tmp_path / "b.db")
        agent = Agent(db, Spender())
        contradiction(agent)
        agent.budgets.set_limit("u", 0.01)
        r = agent.respond("u", "Can you elaborate on the softmax?")
        assert r["budget"]["router_chose"] == "ultra" and r["model_chosen"] == "super"
        assert agent.logger.get_recent_decisions("u", 1)[0]["model_used"] == "super"
        assert "Escalating to Ultra" not in r["routing_reason"]
        # the false-claim checker reads the database: the explanation must match what happened
        assert verify_explanations(db, "u", r["routing_reason"], r["suggestion"]) == []

    def test_spending_never_passes_the_limit(self, tmp_path):
        agent = Agent(str(tmp_path / "b.db"), Spender())
        agent.budgets.set_limit("u", 0.002)
        answered = 0
        with pytest.raises(BudgetExceeded):
            for i in range(200):
                agent.respond("u", f"Explain topic {i}")
                answered += 1
        limit, spent = agent.budgets.get("u")
        assert answered > 3 and spent <= limit + 1e-12

    def test_reports_predicted_and_measured_cost(self, tmp_path):
        agent = Agent(str(tmp_path / "b.db"), Spender())
        r = agent.respond("u", "Explain DNS")
        assert r["budget"]["predicted_usd"] > 0 and r["budget"]["measured_usd"] > 0
        assert agent.budgets.get("u")[1] == pytest.approx(r["budget"]["measured_usd"])   # spend is tracked even with no limit

    def test_predictions_learn_real_answer_lengths(self, tmp_path):
        agent = Agent(str(tmp_path / "b.db"), Spender())        # answers are 300 tokens, the guess is 500
        misses = []
        for i in range(6):
            b = agent.respond("u", f"Explain topic {i}")["budget"]
            misses.append(abs(b["measured_usd"] - b["predicted_usd"]) / b["predicted_usd"])
        assert misses[-1] < misses[0] / 2                    # the miss shrinks as it learns
        assert 300 <= agent.typical.tokens["nano"] < 400

    def test_capped_answers_dont_teach_a_forced_length(self):
        t = budget.Typical()
        t.learn("nano", 100, cap=100)
        assert t.tokens["nano"] == budget.EXPECTED_OUT["nano"]

    def test_unanswered_message_is_still_evidence(self, tmp_path):
        agent = Agent(str(tmp_path / "b.db"), Spender())
        agent.budgets.set_limit("u", 0.000001)
        with pytest.raises(BudgetExceeded):
            agent.respond("u", "Explain DNS")
        assert agent.logger.get_recent_decisions("u", 1)[0]["agent_response"] == ""


class TestApi:
    @pytest.fixture
    def client(self, tmp_path):
        a = Agent(str(tmp_path / "api.db"), Spender())
        app.dependency_overrides[get_agent] = lambda: a
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_set_and_read_the_budget(self, client):
        body = client.post("/budget", json={"user_id": "u1", "limit_usd": 0.05}).json()
        assert body["limit_usd"] == 0.05 and body["remaining_usd"] == 0.05
        client.post("/chat", json={"user_id": "u1", "message": "Explain DNS"})
        after = client.get("/budget/u1").json()
        assert 0 < after["spent_usd"] < 0.05
        reset = client.post("/budget", json={"user_id": "u1", "limit_usd": 0.05, "reset_spent": True}).json()
        assert reset["spent_usd"] == 0

    def test_used_up_is_402_with_a_reason(self, client):
        client.post("/budget", json={"user_id": "u1", "limit_usd": 0.000001})
        r = client.post("/chat", json={"user_id": "u1", "message": "Explain DNS"})
        assert r.status_code == 402 and "Budget" in r.json()["detail"]

    def test_limit_is_validated(self, client):
        assert client.post("/budget", json={"user_id": "u1", "limit_usd": -1}).status_code == 422

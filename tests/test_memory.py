"""Past conversations: compressed archive, 10 days unless kept, reminders. No network; a fake clock."""
import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.agent.core import Agent
from src.agent.llm_client import LLMResult
from src.agent.memory import MemoryStore
from src.api.dependencies import get_agent
from src.main import app


class Echo:
    def __init__(self):
        self.calls = []

    def chat(self, tier, messages, temperature=0.7, **kw):
        self.calls.append(messages)
        return LLMResult(text=f"answer to: {messages[-1]['content'][:40]}", tier=tier, model=tier, latency_ms=1)


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 28, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def days(self, n):
        self.t += timedelta(days=n)


@pytest.fixture
def setup(tmp_path):
    agent = Agent(str(tmp_path / "m.db"), Echo(), memory_dir=str(tmp_path / "memory"))
    clock = Clock()
    agent.memory = MemoryStore(agent.memory.db_path, tmp_path / "memory", 10, now=clock)
    return agent, clock


def chat(agent, *messages, user="u"):
    for m in messages:
        agent.respond(user, m)


class TestArchive:
    def test_new_chat_compresses_the_conversation(self, setup):
        agent, _ = setup
        chat(agent, "Explain DNS", "And TCP?")
        item = agent.memory.archive("u")
        assert item["turns"] == 2 and item["title"] == "Explain DNS" and not item["kept"]
        path = agent.memory.file("u", item["id"])
        assert path.suffix == ".gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            saved = json.load(f)
        assert [t["user_message"] for t in saved["turns"]] == ["Explain DNS", "And TCP?"]

    def test_empty_conversation_is_not_archived(self, setup):
        assert setup[0].memory.archive("u") is None

    def test_new_chat_starts_clean_but_keeps_the_profile(self, setup):
        agent, _ = setup
        chat(agent, "Explain DNS in more detail", "Elaborate on caching")
        habit = agent.profiles.get_observed_profile("u")["length"]
        agent.memory.archive("u")
        agent.respond("u", "What is a mutex?")
        sent = agent.llm.calls[-1]
        assert not any("DNS" in m["content"] for m in sent[1:-1])        # the old chat isn't context any more
        assert agent.profiles.get_observed_profile("u")["length"] == habit == "detailed"

    def test_each_new_chat_archives_only_its_own_turns(self, setup):
        agent, _ = setup
        chat(agent, "one")
        first = agent.memory.archive("u")
        chat(agent, "two", "three")
        second = agent.memory.archive("u")
        assert first["turns"] == 1 and second["turns"] == 2

    def test_regenerate_cannot_reach_into_an_archived_chat(self, setup):
        agent, _ = setup
        chat(agent, "Explain DNS")
        agent.memory.archive("u")
        with pytest.raises(ValueError):
            agent.regenerate("u")


class TestRetention:
    def test_unkept_conversation_is_deleted_after_10_days(self, setup):
        agent, clock = setup
        chat(agent, "Explain DNS")
        item = agent.memory.archive("u")
        path = agent.memory.file("u", item["id"])
        clock.days(9.9)
        assert agent.memory.sweep() == 0 and path.exists()
        clock.days(0.2)
        assert agent.memory.sweep() == 1 and not path.exists()
        assert agent.memory.list("u") == []

    def test_deletion_erases_the_words_not_the_statistics(self, setup):
        agent, clock = setup
        chat(agent, "My secret plan")
        agent.memory.archive("u")
        clock.days(11)
        agent.memory.sweep()
        row = agent.logger.get_recent_decisions("u", 1)[0]
        assert row["user_message"] == "" and row["agent_response"] == "" and row["model_used"] == "nano"

    def test_kept_conversation_is_never_deleted(self, setup):
        agent, clock = setup
        chat(agent, "Explain DNS")
        item = agent.memory.archive("u")
        agent.memory.keep("u", item["id"])
        clock.days(400)
        assert agent.memory.sweep() == 0 and agent.memory.list("u")[0]["kept"]

    def test_unkeeping_restarts_the_clock(self, setup):
        agent, clock = setup
        chat(agent, "Explain DNS")
        item = agent.memory.archive("u")
        agent.memory.keep("u", item["id"])
        clock.days(30)
        again = agent.memory.keep("u", item["id"], kept=False)
        assert again["days_left"] == pytest.approx(10)

    def test_reminders_start_3_days_before(self, setup):
        agent, clock = setup
        chat(agent, "Explain DNS")
        agent.memory.archive("u")
        clock.days(6)
        assert agent.memory.reminders("u") == []
        clock.days(1.5)
        assert len(agent.memory.reminders("u")) == 1

    def test_users_cannot_touch_each_others_memory(self, setup):
        agent, _ = setup
        chat(agent, "Explain DNS", user="alice")
        item = agent.memory.archive("alice")
        for action in (lambda: agent.memory.read("mallory", item["id"]),
                       lambda: agent.memory.keep("mallory", item["id"]),
                       lambda: agent.memory.delete("mallory", item["id"])):
            with pytest.raises(LookupError):
                action()

    def test_user_ids_never_become_paths(self, setup):
        agent, _ = setup
        chat(agent, "hi there", user="..")
        item = agent.memory.archive("..")
        path = agent.memory.file("..", item["id"])
        assert agent.memory.folder in path.parents          # stays inside the memory folder


class TestApi:
    @pytest.fixture
    def client(self, tmp_path):
        a = Agent(str(tmp_path / "api.db"), Echo(), memory_dir=str(tmp_path / "memory"))
        app.dependency_overrides[get_agent] = lambda: a
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_new_chat_list_keep_read_download_delete(self, client):
        client.post("/chat", json={"user_id": "u1", "message": "Explain DNS"})
        archived = client.post("/conversations/new", json={"user_id": "u1"}).json()["archived"]
        cid = archived["id"]
        mem = client.get("/memory/u1").json()
        assert mem["items"][0]["id"] == cid and mem["retention_days"] == 10
        assert client.get("/decision-log/u1?current=true").json()["decisions"] == []   # new chat is empty
        assert len(client.get("/decision-log/u1").json()["decisions"]) == 1             # history still there
        assert client.post("/memory/keep", json={"user_id": "u1", "conversation_id": cid}).json()["kept"]
        assert client.get(f"/memory/u1/{cid}").json()["turns"][0]["user_message"] == "Explain DNS"
        dl = client.get(f"/memory/u1/{cid}/download")
        assert dl.status_code == 200 and gzip.decompress(dl.content)
        assert client.delete(f"/memory/u1/{cid}").status_code == 200
        assert client.get(f"/memory/u1/{cid}").status_code == 404

    def test_empty_new_chat(self, client):
        assert client.post("/conversations/new", json={"user_id": "u1"}).json()["archived"] is None

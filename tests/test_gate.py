"""The judges-only access code on a hosted demo."""
import pytest
from fastapi.testclient import TestClient

from src import config
from src.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "DEMO_ACCESS_CODE", "open-sesame")
    return TestClient(app, follow_redirects=False)


def test_no_code_set_means_open(monkeypatch):
    monkeypatch.setattr(config, "DEMO_ACCESS_CODE", "")
    assert TestClient(app).get("/health").status_code == 200


def test_everything_is_locked_without_the_code(client):
    assert client.get("/health").status_code == 401
    assert client.post("/chat", json={"user_id": "u" * 20, "message": "hi"}).status_code == 401
    page = client.get("/")
    assert page.status_code == 200 and "Access code" in page.text


def test_wrong_code_is_refused(client):
    assert client.post("/unlock", content="code=nope", headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 401
    assert client.get("/?code=nope").status_code == 401
    assert client.get("/health").status_code == 401


def test_right_code_unlocks_with_a_cookie(client):
    r = client.post("/unlock", content="code=open-sesame", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 303 and "dr_access" in r.headers["set-cookie"]
    assert client.get("/health").status_code == 200


def test_code_in_the_link_unlocks_too(client):
    assert client.get("/?code=open-sesame").status_code == 303
    assert client.get("/health").status_code == 200


def test_guessing_is_rate_limited(client):
    from src.api import gate
    gate._tries.clear()
    codes = [client.get("/?code=wrong%d" % i).status_code for i in range(gate.MAX_TRIES_PER_MIN + 2)]
    assert codes[-1] == 429
    gate._tries.clear()

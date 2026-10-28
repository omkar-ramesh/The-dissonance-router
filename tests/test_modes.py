"""models.toml profiles and the offline / online switch."""
import pytest
from fastapi.testclient import TestClient

from src import config
from src.main import app


@pytest.fixture(autouse=True)
def back_to_offline():
    yield
    config.apply_mode("offline")
    config.MODE = "offline"


def test_offline_profile_matches_the_local_models():
    config.apply_mode("offline")
    assert config.MODEL_TIERS["nano"] == "nemotron-3-nano:4b"
    assert set(config.TIER_PROVIDERS.values()) == {"ollama"}


def test_online_is_locked_until_ids_are_confirmed(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "k")
    monkeypatch.setattr(config, "NEBIUS_BASE_URL", "https://example.test/v1")
    with pytest.raises(ValueError, match="confirmed"):
        config.set_mode("online")
    assert config.MODE == "offline"


def test_online_switch_needs_confirm_and_never_opens_in_public_demo(monkeypatch):
    client = TestClient(app)
    assert client.post("/mode", json={"mode": "online"}).status_code == 400
    monkeypatch.setattr(config, "PUBLIC_DEMO", True)
    assert client.post("/mode", json={"mode": "offline"}).status_code == 403

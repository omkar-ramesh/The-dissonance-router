"""Environment-driven configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_PATH = os.getenv("DB_PATH", "data/dissonance.db")
if not os.path.isabs(DB_PATH):
    DB_PATH = str(ROOT_DIR / DB_PATH)

# Default backend for every tier: "ollama" (local dev) or "nebius" (Token Factory).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Router tiers map to concrete model IDs here so a backend swap is config-only.
MODEL_TIERS = {
    "nano": os.getenv("MODEL_NANO", "nemotron-3-nano:4b"),
    "super": os.getenv("MODEL_SUPER", "nemotron-3-nano:30b"),
    "ultra": os.getenv("MODEL_ULTRA", "deepseek-r1:32b"),
}

# USD per 1M tokens on Nebius Token Factory, used for the dashboard's cost estimate.
# Defaults are list prices from third-party listings (Oct 2026), not read from Nebius
# itself: confirm on the Nebius price page and override in .env if they differ.
PRICES = {
    tier: {
        "in": float(os.getenv(f"PRICE_{tier.upper()}_IN", default_in)),
        "out": float(os.getenv(f"PRICE_{tier.upper()}_OUT", default_out)),
    }
    for tier, (default_in, default_out) in {
        "nano": ("0.06", "0.24"), "super": ("0.30", "0.90"), "ultra": ("1.00", "3.00"),
    }.items()
}

# Fallback context window (tokens) per tier for backends that don't report one.
# Ollama tiers read the real value from /api/ps; set these for Token Factory models.
CONTEXT_WINDOWS = {tier: int(os.getenv(f"CTX_{tier.upper()}", "0")) for tier in MODEL_TIERS}

# Optional per-tier backend override, e.g. PROVIDER_ULTRA=nebius to spend credits
# only on the Ultra tier while Nano and Super stay on local Ollama.
TIER_PROVIDERS = {
    tier: os.getenv(f"PROVIDER_{tier.upper()}", LLM_PROVIDER) for tier in MODEL_TIERS
}

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

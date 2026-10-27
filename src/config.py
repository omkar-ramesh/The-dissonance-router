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

# Reasoning per tier. "off" asks the model to answer without a thinking phase: on local
# Nemotron Nano that cut "hi" from 33 to 13 output tokens. Thinking is kept where it pays,
# on Ultra. Models that always reason (e.g. deepseek-r1) ignore "off".
REASONING = {
    tier: os.getenv(f"REASONING_{tier.upper()}", "on" if tier == "ultra" else "off") for tier in MODEL_TIERS
}

# Who reads each message for style requests: "keywords" (fast, English-only phrasings)
# or "nano" (Nemotron Nano reads it, keywords as the fallback if the call fails).
SIGNAL_EXTRACTOR = os.getenv("SIGNAL_EXTRACTOR", "nano")

# Public demo mode: private random session ids, a per-IP rate limit, output caps and a daily
# spend cut-off, so a public link can't read other people's chats or drain the credits.
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "0") == "1"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "6"))
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "1.00"))
# Max output tokens per answer (0 = no cap). Capped by default only in public demo mode.
_CAP_DEFAULTS = {"nano": "1500", "super": "2500", "ultra": "4000"} if PUBLIC_DEMO else {}
MAX_TOKENS = {tier: int(os.getenv(f"MAX_TOKENS_{tier.upper()}", _CAP_DEFAULTS.get(tier, "0"))) for tier in MODEL_TIERS}

# Past conversations: compressed files here, deleted after this many days unless kept.
MEMORY_DIR = os.getenv("MEMORY_DIR", str(ROOT_DIR / "data" / "memory"))
MEMORY_RETENTION_DAYS = float(os.getenv("MEMORY_RETENTION_DAYS", "10"))

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# 127.0.0.1, not "localhost": on Windows every new connection to "localhost" tries IPv6
# first and costs ~2.2 s before falling back (measured), against 0.16 s for 127.0.0.1.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("//localhost", "//127.0.0.1")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

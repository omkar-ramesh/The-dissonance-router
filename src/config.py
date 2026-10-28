"""Environment-driven configuration."""
import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_PATH = os.getenv("DB_PATH", "data/dissonance.db")
if not os.path.isabs(DB_PATH):
    DB_PATH = str(ROOT_DIR / DB_PATH)

# Models per tier live in models.toml: an "offline" profile (local Ollama) and an "online" one
# (Nebius Token Factory). LLM_PROVIDER=nebius in .env still means "online"; MODE picks one explicitly.
TIERS = ("nano", "super", "ultra")
try:
    MODELS_FILE = ROOT_DIR / "models.toml"
    _FILE = tomllib.loads(MODELS_FILE.read_text(encoding="utf-8")) if MODELS_FILE.exists() else {}
except tomllib.TOMLDecodeError as e:
    raise SystemExit(f"models.toml is not valid: {e}")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODE = "online" if LLM_PROVIDER == "nebius" else os.getenv("MODE", _FILE.get("default_mode", "offline"))
_DEFAULT = {"nano": ("ollama", "nemotron-3-nano:4b", "off"), "super": ("ollama", "nemotron-3-nano:30b", "off"),
            "ultra": ("ollama", "deepseek-r1:32b", "on")}
_PRICE_DEFAULT = {"nano": (0.06, 0.24), "super": (0.30, 0.90), "ultra": (1.00, 3.00)}

MODEL_TIERS, TIER_PROVIDERS, REASONING, PRICES = {}, {}, {}, {}


def _entry(mode: str, tier: str) -> dict:
    d = _DEFAULT[tier]
    return {"provider": d[0], "model": d[1], "reasoning": d[2], **_FILE.get(mode, {}).get(tier, {})}


def apply_mode(mode: str, use_env: bool = False) -> None:
    """Point every tier at the models of one profile. Dicts are updated in place so nothing holding them goes stale."""
    for tier in TIERS:
        e, up = _entry(mode, tier), tier.upper()
        MODEL_TIERS[tier] = os.getenv(f"MODEL_{up}", e["model"]) if use_env else e["model"]
        TIER_PROVIDERS[tier] = os.getenv(f"PROVIDER_{up}", e["provider"]) if use_env else e["provider"]
        REASONING[tier] = os.getenv(f"REASONING_{up}", e["reasoning"]) if use_env else e["reasoning"]
        price = {**dict(zip(("in", "out"), _PRICE_DEFAULT[tier])), **_FILE.get("prices", {}).get(tier, {})}
        PRICES[tier] = {"in": float(os.getenv(f"PRICE_{up}_IN", price["in"])), "out": float(os.getenv(f"PRICE_{up}_OUT", price["out"]))}


def online_blocker() -> str:
    """Why the online profile can't be used right now ('' if it can)."""
    if not (NEBIUS_API_KEY and NEBIUS_BASE_URL):
        return "Add NEBIUS_API_KEY and NEBIUS_BASE_URL to .env."
    if not _FILE.get("online", {}).get("confirmed"):
        return "Confirm the Token Factory model ids in models.toml (set confirmed = true)."
    return ""


def set_mode(mode: str) -> None:
    global MODE
    if mode not in ("offline", "online"):
        raise ValueError("mode must be 'offline' or 'online'")
    if mode == "online" and online_blocker():
        raise ValueError(online_blocker())
    apply_mode(mode)
    MODE = mode


# Fallback context window (tokens) per tier for backends that don't report one.
# Ollama tiers read the real value from /api/ps; set these for Token Factory models.
CONTEXT_WINDOWS = {tier: int(os.getenv(f"CTX_{tier.upper()}", "0")) for tier in TIERS}

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
MAX_TOKENS = {tier: int(os.getenv(f"MAX_TOKENS_{tier.upper()}", _CAP_DEFAULTS.get(tier, "0"))) for tier in TIERS}

# Past conversations: compressed files here, deleted after this many days unless kept.
MEMORY_DIR = os.getenv("MEMORY_DIR", str(ROOT_DIR / "data" / "memory"))
MEMORY_RETENTION_DAYS = float(os.getenv("MEMORY_RETENTION_DAYS", "10"))

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# 127.0.0.1, not "localhost": on Windows every new connection to "localhost" tries IPv6
# first and costs ~2.2 s before falling back (measured), against 0.16 s for 127.0.0.1.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("//localhost", "//127.0.0.1")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
# Shown on local reports so numbers are tied to the machine that produced them.
HARDWARE = os.getenv("HARDWARE_LABEL", "NVIDIA RTX 5090 · 32 GB VRAM")
# Token Factory models we plan to use, as listed in the Token Factory catalog (Oct 2026).
# Exact API ids must be confirmed in the catalog before a live run.
TOKEN_FACTORY_MODELS = [
    {"tier": "nano", "name": "Nemotron-3-Nano-30B-A3B", "role": "Default tier: aligned users"},
    {"tier": "super", "name": "Nemotron-3-Super-120B", "role": "Moderate conflict"},
    {"tier": "ultra", "name": "Nemotron-3-Ultra-550B", "role": "High conflict, reasoning on"},
    {"tier": "", "name": "Nemotron-3.5-Lightning", "role": "Candidate fast Nano alternative (not wired)"},
]
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

apply_mode(MODE, use_env=True)

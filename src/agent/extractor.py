"""Nemotron Nano reads each message for style requests; keywords are the fallback.

The keyword reader only knows explicit English phrasings. It misses "I'm still lost" and
other languages, and it is fooled by negation ("don't make it shorter") and by quoted
words ("my boss says 'keep it short'"). Nano reads the message the way a person would.

Cost: one short Nano call per new message, with reasoning off and a small output cap.
Results are cached by message text, so re-weighing recent history costs nothing.
If the call fails or returns something unusable, the keyword reader answers instead.
"""
import hashlib
import json
import re
from contextlib import closing
from typing import Optional

from src.agent.llm_client import LLMError
from src.agent.profiles import VALID_VALUES
from src.agent.signals import SignalExtractor
from src.db.database import get_connection

PROMPT_VERSION = "v3"   # bump when the prompt changes, so cached readings are redone

SYSTEM = """You find style requests in a chat message: what the user asks about how the assistant's answers should look.
Most messages have none. A question or task on its own ("Explain X", "Write Y", "What is Z?") is not a style request: reply {}.
Otherwise reply with a JSON object mapping each requested dimension to [value, quote], where quote is the user's exact words that ask for it.
Dimensions and values:
  tone: formal | casual
  length: concise | detailed
  sourcing: minimal | full
  structure: bullet | paragraph
  revision: rejects | accepts
Rules:
- Only the user's own request counts, in any language. Words they quote from someone else do not.
- Read negation: "don't make it shorter" asks for detailed, not concise.
- Being confused or lost after an explanation asks for more detail: length detailed.
- Saying the answer missed, or asking for another attempt: revision rejects. Saying it was exactly right: revision accepts.
- Never guess. No quote, no key."""

EXAMPLES = [
    ("Explain how photosynthesis works", "{}"),
    ("Explain DNS, keep it brief", '{"length": ["concise", "keep it brief"]}'),
    ("Write a short story about a dragon", "{}"),
    ("What's the capital of Peru?", "{}"),
    ("I'm new to Python, how do decorators work?", "{}"),
    ('My teacher always says "be formal". Anyway, what is a mutex?', "{}"),
    ("I still don't get it", '{"length": ["detailed", "I still don\'t get it"]}'),
    ("no bullet points please, and talk to me like a friend",
     '{"structure": ["paragraph", "no bullet points"], "tone": ["casual", "talk to me like a friend"]}'),
    ("Thanks, that's exactly what I needed", '{"revision": ["accepts", "exactly what I needed"]}'),
]

_JSON = re.compile(r"\{.*\}", re.S)
# Tolerant on purpose: the quote may contain unescaped quote marks ("says "keep it short"").
_FIRST_KEY = re.compile(r'\{\s*"(\w+)"\s*:\s*\[\s*"(\w+)"\s*,\s*"(.*?)"\s*\]', re.S)
# Text inside quotation marks is someone else's words, not the user's request.
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)\'[^\']+\'(?!\w)')


class NanoExtractor(SignalExtractor):
    """A SignalExtractor whose `extract` asks Nemotron Nano, falling back to keywords."""

    def __init__(self, llm, db_path: Optional[str] = None, tier: str = "nano", max_tokens: int = 100):
        self.llm, self.db_path, self.tier, self.max_tokens = llm, db_path, tier, max_tokens
        self._memo: dict[str, dict] = {}
        self.calls = []        # LLMResults of fresh (uncached) readings, for honest cost accounting
        self.fallbacks = 0     # times the keyword reader had to answer instead

    def extract(self, message: str) -> dict[str, str]:
        if not message or not message.strip():
            return {}
        key = hashlib.sha256(f"{PROMPT_VERSION}\n{message}".encode("utf-8")).hexdigest()
        cached = self._get(key)
        if cached is not None:
            return cached
        try:
            signals = self._ask(message)
        except (LLMError, ValueError, TypeError):
            self.fallbacks += 1
            return super().extract(message)        # not cached: try Nano again next time
        self._put(key, signals)
        return signals

    def _ask(self, message: str) -> dict[str, str]:
        messages = [{"role": "system", "content": SYSTEM}]
        for text, answer in EXAMPLES:
            messages += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        messages.append({"role": "user", "content": message})
        result = self.llm.chat(self.tier, messages, temperature=0.0, reasoning="off", max_tokens=self.max_tokens)
        self.calls.append(result)
        return parse(result.text, message)

    def take_calls(self) -> list:
        """Readings made since the last call (and forget them)."""
        calls, self.calls = self.calls, []
        return calls

    def _get(self, key: str) -> Optional[dict]:
        if key in self._memo:
            return self._memo[key]
        if not self.db_path:
            return None
        with closing(get_connection(self.db_path)) as conn:
            row = conn.execute("SELECT signals FROM signal_cache WHERE text_hash = ?", (key,)).fetchone()
        if row is None:
            return None
        self._memo[key] = json.loads(row["signals"])
        return self._memo[key]

    def _put(self, key: str, signals: dict) -> None:
        self._memo[key] = signals
        if self.db_path:
            with closing(get_connection(self.db_path)) as conn:
                with conn:
                    conn.execute("INSERT OR REPLACE INTO signal_cache (text_hash, signals) VALUES (?, ?)",
                                 (key, json.dumps(signals)))


def parse(text: str, message: str) -> dict[str, str]:
    """The model's first signal, kept only when backed by evidence.

    Only the first key is read: on local Nemotron Nano 4B the first key was almost always
    right, while extra keys were mostly invented from topic words. A message asking for two
    things therefore loses the second; that is the price of not inventing preferences.
    The key must be a known dimension with a pole value AND a quote that really is in the
    message and doesn't overlap quotation marks (someone else's words). Otherwise nothing
    is learned (guideline 6: no invented preferences).
    Raises ValueError when there is no JSON object at all, so the caller falls back.
    """
    text = text or ""
    first = _FIRST_KEY.search(text)
    if first:
        # Read the first key straight from the text: later keys sometimes break the JSON
        # (e.g. unescaped quotes), and they are ignored anyway.
        dim, value, quote = first.group(1), first.group(2), _norm(first.group(3))
    else:
        match = _JSON.search(text)
        if not match:
            raise ValueError(f"no JSON object in {text[:80]!r}")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return {}            # {} or keys without a [value, quote] pair: no evidence, nothing learned
    if dim not in VALID_VALUES or value not in VALID_VALUES[dim] or value == "balanced":
        return {}
    others = [_norm(q) for q in _QUOTED.findall(message)]
    if len(quote) < 3 or quote not in _norm(message) or any(quote in q or q in quote for q in others if q):
        return {}
    return {dim: value}


def _norm(s: str) -> str:
    """Letters and digits only, so "don' t" (a tokenizer quirk) still matches "don't"."""
    return "".join(ch for ch in s.lower() if ch.isalnum())

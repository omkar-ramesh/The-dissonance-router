"""OpenAI-compatible streaming chat client.

Ollama (local dev) and Nebius Token Factory expose the same /chat/completions API,
so switching backends is a base URL + model ID change in .env, not a code change.
Responses are streamed so we can measure time-to-first-token.
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from src import config

# Reasoning models inline their thinking in <think> tags.
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)

# How to ask each backend to skip the thinking phase. Ollama's /v1 endpoint: measured on
# nemotron-3-nano:4b (reasoning 89 chars -> 0). Token Factory: the usual switch for
# vLLM-served Nemotron, NOT yet verified against the live API.
REASONING_OFF = {
    "ollama": {"reasoning_effort": "none"},
    "nebius": {"chat_template_kwargs": {"enable_thinking": False}},
}


class LLMError(RuntimeError):
    """The model backend was unreachable or returned an error."""


@dataclass
class LLMResult:
    text: str
    tier: str
    model: str
    latency_ms: int
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    ttft_ms: Optional[int] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    model_name: str = ""
    context_window: Optional[int] = None
    est_cost_usd: Optional[float] = None
    est_cost_if_ultra_usd: Optional[float] = None
    reasoning_on: Optional[bool] = None

    @property
    def reasoning_tokens(self) -> int:
        """Rough size of the thinking phase (~4 chars per token); it is billed as output."""
        return len(self.reasoning) // 4

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_sec(self) -> float:
        """Output tokens over wall-clock time (includes prompt processing and TTFT)."""
        return round(self.output_tokens / (self.latency_ms / 1000), 1) if self.latency_ms else 0.0

    @property
    def context_pct(self) -> Optional[float]:
        if not self.context_window:
            return None
        return round(100 * self.total_tokens / self.context_window, 1)

    def stats(self) -> dict:
        """Flat numbers for the dev dashboard and logs."""
        return {
            "tier": self.tier,
            "model_id": self.model,
            "model_name": self.model_name or self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_on": self.reasoning_on,
            "tokens_estimated": self.tokens_estimated,
            "tokens_per_sec": self.tokens_per_sec,
            "ttft_ms": self.ttft_ms,
            "latency_ms": self.latency_ms,
            "agent_rounds": 1,
            "context_window": self.context_window,
            "context_pct": self.context_pct,
            "est_cost_usd": self.est_cost_usd,
            "est_cost_if_ultra_usd": self.est_cost_if_ultra_usd,
        }


class LLMClient:
    """Calls the concrete model behind a router tier ("nano" | "super" | "ultra")."""

    def __init__(
        self,
        base_url: str,
        models: dict,
        api_key: str = "",
        timeout: float = 180.0,
        transport: Optional[httpx.BaseTransport] = None,
        overrides: Optional[dict] = None,
        ps_urls: Optional[dict] = None,
        context_windows: Optional[dict] = None,
        prices: Optional[dict] = None,
        providers: Optional[dict] = None,
        reasoning: Optional[dict] = None,
        max_tokens: Optional[dict] = None,
    ):
        """`overrides` maps a tier to {"base_url": ..., "api_key": ...} for tiers that use a
        different backend than the default. `ps_urls` maps tiers served by Ollama to its
        /api/ps URL, used to read the loaded model's real context window.
        `context_windows` is a fallback {tier: tokens} for backends that don't report one.
        `prices` maps tier -> {"in": usd, "out": usd} per 1M tokens, for cost estimates.
        `providers` ({tier: "ollama" | "nebius"}) and `reasoning` ({tier: "on" | "off"})
        decide how to switch thinking off; `max_tokens` ({tier: n}, 0 = none) caps output."""
        self.models = models
        self._transport = transport
        self._ps_urls = ps_urls or {}
        self._context_windows = context_windows or {}
        self._prices = prices or {}
        self._providers = providers or {}
        self._reasoning = reasoning or {}
        self._max_tokens = max_tokens or {}
        self._info_cache: dict[tuple, tuple] = {}
        self._clients = {tier: self._make_http(base_url, api_key, timeout, transport) for tier in models}
        for tier, backend in (overrides or {}).items():
            self._clients[tier] = self._make_http(
                backend["base_url"], backend.get("api_key", ""), timeout, transport
            )

    @staticmethod
    def _make_http(base_url: str, api_key: str, timeout: float, transport) -> httpx.Client:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def from_config(cls) -> "LLMClient":
        """Build a client where each tier uses the backend named in config.TIER_PROVIDERS."""
        backends, ps_urls = {}, {}
        for tier, provider in config.TIER_PROVIDERS.items():
            if provider == "nebius":
                if not (config.NEBIUS_API_KEY and config.NEBIUS_BASE_URL):
                    raise LLMError(
                        f"Tier '{tier}' uses nebius but NEBIUS_API_KEY / NEBIUS_BASE_URL are not set in .env"
                    )
                backends[tier] = {"base_url": config.NEBIUS_BASE_URL, "api_key": config.NEBIUS_API_KEY}
            elif provider == "ollama":
                root = config.OLLAMA_BASE_URL.rstrip("/")
                backends[tier] = {"base_url": f"{root}/v1"}
                ps_urls[tier] = f"{root}/api/ps"
            else:
                raise LLMError(f"Unknown provider {provider!r} for tier '{tier}'; use 'ollama' or 'nebius'")
        default = backends["nano"]
        return cls(
            default["base_url"], config.MODEL_TIERS, default.get("api_key", ""),
            overrides=backends, ps_urls=ps_urls, context_windows=config.CONTEXT_WINDOWS,
            prices=config.PRICES, providers=config.TIER_PROVIDERS, reasoning=config.REASONING,
            max_tokens=config.MAX_TOKENS,
        )

    def chat(
        self,
        tier: str,
        messages: list[dict],
        temperature: float = 0.7,
        reasoning: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """Stream a chat completion from the tier's model and return its cleaned answer + stats.

        `reasoning` ("on" | "off") and `max_tokens` override the tier's configured defaults.
        """
        if tier not in self.models:
            raise ValueError(f"Unknown tier {tier!r}; expected one of {list(self.models)}")
        model = self.models[tier]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        mode = reasoning or self._reasoning.get(tier)
        if mode == "off":
            payload.update(REASONING_OFF.get(self._providers.get(tier), {}))
        cap = max_tokens if max_tokens is not None else self._max_tokens.get(tier, 0)
        if cap:
            payload["max_tokens"] = cap

        content_parts, reasoning_parts, usage = [], [], {}
        ttft_ms = None
        start = time.perf_counter()
        try:
            with self._clients[tier].stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMError(f"{model} returned {response.status_code}: {response.text[:200]}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise LLMError(f"{model} stream error: {str(chunk['error'])[:200]}")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if (piece or thought) and ttft_ms is None:
                            ttft_ms = int((time.perf_counter() - start) * 1000)
                        content_parts.append(piece)
                        reasoning_parts.append(thought)
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach the LLM backend for {model}: {e}") from e
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        thoughts = THINK_RE.findall(text)
        if thoughts:
            reasoning = "\n".join([reasoning, *thoughts]).strip()
            text = THINK_RE.sub("", text).strip()

        estimated = not usage
        if estimated:  # backend didn't report usage; ~4 chars per token
            input_tokens = sum(len(m.get("content") or "") for m in messages) // 4
            output_tokens = (len(text) + len(reasoning)) // 4
        else:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        model_name, context_window = self._model_info(tier, model)
        return LLMResult(
            text=text,
            tier=tier,
            model=model,
            latency_ms=latency_ms,
            reasoning=reasoning,
            usage=usage,
            ttft_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_estimated=estimated,
            model_name=model_name,
            context_window=context_window,
            est_cost_usd=self._cost(tier, input_tokens, output_tokens),
            est_cost_if_ultra_usd=self._cost("ultra", input_tokens, output_tokens),
            reasoning_on=None if mode is None else mode != "off",
        )

    def _cost(self, tier: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Dollar cost of these tokens at the tier's per-1M-token prices (None if unpriced)."""
        price = self._prices.get(tier)
        if not price:
            return None
        return (input_tokens * price["in"] + output_tokens * price["out"]) / 1_000_000

    def _model_info(self, tier: str, model: str) -> tuple[str, Optional[int]]:
        """(display name, context window) for the tier's model; best effort, never raises.

        Ollama's /api/ps reports the context the loaded model really runs with, which
        is what determines when it starts truncating.
        """
        if (tier, model) in self._info_cache:      # a loaded model's details don't change between calls
            return self._info_cache[(tier, model)]
        name, window = model, self._context_windows.get(tier) or None
        ps_url = self._ps_urls.get(tier)
        if ps_url:
            try:
                with httpx.Client(transport=self._transport, timeout=2.0) as http:
                    for loaded in http.get(ps_url).json().get("models", []):
                        if loaded.get("name") == model or loaded.get("model") == model:
                            details = loaded.get("details", {})
                            label = " ".join(
                                str(details[k]) for k in ("family", "parameter_size", "quantization_level")
                                if details.get(k)
                            )
                            name = label or model
                            window = loaded.get("context_length") or window
                            self._info_cache[(tier, model)] = (name, window)
            except (httpx.HTTPError, ValueError):
                pass
        return name, window

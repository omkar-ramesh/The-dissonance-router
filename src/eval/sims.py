"""Stand-in models for offline runs. Nothing here calls a network.

SimLLM only exists so routing, grading and cost arithmetic can be exercised without a
model. Its token counts are simulated, so a report built on it says so and its cost
figures are only good for comparing strategies against each other.
"""
from src.agent.llm_client import LLMResult

_WORDS = {"nano": 60, "super": 110, "ultra": 200}      # typical answer length per tier


def _tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


class SimLLM:
    simulated = True

    def chat(self, tier, messages, temperature=0.7):
        last = messages[-1]["content"]
        body = " ".join(["word"] * _WORDS[tier])
        text = f"TL;DR: a short answer.\n\n{body}" if "TL;DR" in last else body
        return LLMResult(
            text=text, tier=tier, model=f"sim-{tier}", latency_ms=1,
            input_tokens=sum(_tokens(m["content"]) for m in messages), output_tokens=_tokens(text),
            tokens_estimated=True,
        )


class LyingLLM(SimLLM):
    """Answers confidently and attaches a source it made up."""

    def chat(self, tier, messages, temperature=0.7):
        result = super().chat(tier, messages, temperature)
        result.text = "It was proposed in 2017 (see https://arxiv.org/abs/2401.99999)."
        return result


class ForcedTier:
    """Wraps a client so every request goes to one tier, to price the 'always X' strategies."""

    def __init__(self, inner, tier):
        self.inner, self.tier = inner, tier
        self.simulated = getattr(inner, "simulated", False)

    def chat(self, tier, messages, temperature=0.7):
        return self.inner.chat(self.tier or tier, messages, temperature)

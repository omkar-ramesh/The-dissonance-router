"""Turns an analysis into the system prompt for the chosen tier.

Each tier has a distinct job:
  - nano:  preferences agree; just follow them.
  - super: a partial gap; lean toward observed behavior.
  - ultra: a real contradiction; reason about the need behind both signals and
           shape one answer that serves it (e.g. a short summary first, then detail).
"""
from src.agent.profiles import DIMENSIONS

STYLE_RULES = {
    "tone": {"formal": "Use a formal, professional tone.",
             "casual": "Use a casual, conversational tone."},
    "length": {"concise": "Keep the answer short.",
               "detailed": "Be thorough and detailed."},
    "sourcing": {"minimal": "Don't include citations.",
                 "full": "Back factual claims with sources."},
    "structure": {"bullet": "Prefer bullet points.",
                  "paragraph": "Write in paragraphs, not bullet lists."},
    "revision": {"rejects": "This user often rejects first drafts: state key assumptions and offer one alternative.",
                 "accepts": "Deliver a ready-to-use answer without hedging."},
}

BASE_PROMPT = "You are a personal assistant that adapts to how this user actually works, not only what they say."

ULTRA_PROMPT = (
    "This user's stated preferences contradict their actual behavior:\n{contradictions}\n"
    "Work out the real need behind both signals for this request and serve it. "
    "Shape the answer like this:\n{hints}\n"
    "Do not mention this analysis."
)

# How to honor both sides of a contradiction in one answer. Keyed by dimension.
RECONCILE_HINTS = {
    "length": "- Your FIRST line must be a one or two sentence direct answer starting with 'TL;DR:'. "
              "Then give the depth, so they can stop reading whenever they want.",
    "tone": "- Use the tone they actually respond to, but keep the wording precise.",
    "sourcing": "- Keep the body clean; put sources together in a short list at the end.",
    "structure": "- Open with a short paragraph, then use bullets only for steps or lists of items.",
    "revision": "- State your key assumptions up front and offer one alternative they can pick instead.",
}

SUPER_PROMPT = "Their stated and observed preferences differ slightly; lean toward their observed behavior."


def effective_style(stated: dict, observed: dict) -> dict:
    """Behavior wins over words unless behavior is still unknown ("balanced")."""
    return {dim: observed[dim] if observed[dim] != "balanced" else stated[dim] for dim in DIMENSIONS}


def build_system_prompt(analysis: dict) -> str:
    """Compose the system prompt for the tier the router picked."""
    stated, observed = analysis["stated_profile"], analysis["observed_profile"]
    style = effective_style(stated, observed)
    rules = [STYLE_RULES[dim][value] for dim, value in style.items() if value in STYLE_RULES[dim]]

    parts = [BASE_PROMPT]
    if rules:
        parts.append("Style: " + " ".join(rules))

    tier = analysis["model_chosen"]
    if tier == "ultra":
        conflicting = _conflicting(analysis)
        contradictions = "\n".join(
            f"- {dim}: says '{stated[dim]}', behaves '{observed[dim]}'" for dim in conflicting
        )
        hints = "\n".join(RECONCILE_HINTS[dim] for dim in conflicting)
        parts.append(ULTRA_PROMPT.format(contradictions=contradictions, hints=hints))
    elif tier == "super":
        parts.append(SUPER_PROMPT)
    return "\n\n".join(parts)


def build_messages(analysis: dict, history: list[dict], user_message: str) -> list[dict]:
    """System prompt + prior turns (oldest first) + the new message.

    `history` is newest-first, as returned by DecisionLogger.get_recent_decisions.
    """
    messages = [{"role": "system", "content": build_system_prompt(analysis)}]
    for turn in reversed(history):
        if turn.get("agent_response"):
            messages.append({"role": "user", "content": turn["user_message"]})
            messages.append({"role": "assistant", "content": turn["agent_response"]})
    # Some models (e.g. deepseek-r1) under-weight the system prompt, so the
    # reconciliation format is repeated where every model reads it: the user turn.
    if analysis["model_chosen"] == "ultra":
        hints = " ".join(RECONCILE_HINTS[dim].lstrip("- ") for dim in _conflicting(analysis))
        user_message = f"{user_message}\n\n(Answer format: {hints})"
    messages.append({"role": "user", "content": user_message})
    return messages


def _conflicting(analysis: dict) -> list[str]:
    return [dim for dim, score in analysis["conflicts_detected"].items() if score > 0]

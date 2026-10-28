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

BASE_PROMPT = (
    "You are a personal assistant that adapts to how this user actually works, not only what they say. "
    # Guideline 6: honesty over looking helpful.
    "If you don't know something or can't verify it, say so plainly instead of guessing. "
    "Never invent sources, quotes, links or numbers. "
    # Writing quality: applies to every tier, unless the user asked for a casual tone.
    "Write polished, well-organised standard English: correct grammar and punctuation, complete sentences, "
    "precise word choice and a consistent spelling convention. Unless the user asks for a casual tone, "
    "keep a formal, academic register and avoid contractions, slang and emoji. "
    "Open with the answer, with no filler such as 'Sure' or 'Great question'. Use short paragraphs, each with one idea, "
    "and a clear closing sentence for longer answers. Cite a source only when you can verify it."
)

ULTRA_PROMPT = (
    "This user's stated preferences contradict their actual behavior:\n{contradictions}\n"
    "Work out the real need behind both signals for this request and serve it.{shape}\n"
    "Do not mention this analysis."
)

# Guideline 3: what the user asks for in this message beats the profile, for this message.
TURN_RULE = "Follow any explicit instruction in the user's latest message, even where it differs from the style above."

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
    turn = analysis.get("turn_signals") or {}
    style = {**effective_style(stated, observed), **turn}          # this turn's instruction wins
    rules = [STYLE_RULES[dim][value] for dim, value in style.items() if value in STYLE_RULES[dim]]

    parts = [BASE_PROMPT]
    if rules:
        parts.append("Style: " + " ".join(rules))
    if turn:
        parts.append(TURN_RULE)

    tier = analysis["model_chosen"]
    if tier == "ultra":
        conflicting = _conflicting(analysis)
        contradictions = "\n".join(
            f"- {dim}: says '{stated[dim]}', behaves '{observed[dim]}'" for dim in conflicting
        )
        hints = _hints(analysis)
        shape = (" Shape the answer like this:\n" + "\n".join(hints)) if hints else ""
        parts.append(ULTRA_PROMPT.format(contradictions=contradictions, shape=shape))
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
    hints = _hints(analysis) if analysis["model_chosen"] == "ultra" else []
    if hints:
        user_message = f"{user_message}\n\n(Answer format: {' '.join(h.lstrip('- ') for h in hints)})"
    messages.append({"role": "user", "content": user_message})
    return messages


def _conflicting(analysis: dict) -> list[str]:
    return [dim for dim, score in analysis["conflicts_detected"].items() if score > 0]


def _hints(analysis: dict) -> list[str]:
    """Reconciliation hints for the conflicting dimensions.

    A hint adds the *stated* side on top of the observed behavior (e.g. a TL;DR line on
    top of the depth they keep asking for), so it is skipped only when this message
    itself sides with what they said ("keep it short" from someone who claimed concise):
    then they are doing what they said, and the plain instruction wins.
    """
    turn = analysis.get("turn_signals") or {}
    stated = analysis["stated_profile"]
    return [RECONCILE_HINTS[dim] for dim in _conflicting(analysis) if turn.get(dim) != stated[dim]]

"""Guideline 4: is this request worth spending a stronger model on?

Greetings, thanks and very short lookups have no style choices to make, so they stay
on Nano whatever the profile says. Deliberately conservative: anything that asks for
an explanation, a piece of writing or a comparison is substantive. Callers must not
apply this to a message that carries a style signal ("elaborate" is short but is the
strongest evidence there is).
"""
import re

ACK_RE = re.compile(
    r"^\W*(hi|hey|hello|yo|thanks|thank you|thx|ty|ok(ay)?|cool|great|nice|got it|bye|good (morning|night|evening))"
    r"\b[\w\s,.!?']{0,20}$",
    re.I,
)
SUBSTANTIVE_RE = re.compile(
    r"\b(explain|describe|write|draft|compare|summari[sz]e|analy[sz]e|review|rewrite|plan|design|outline"
    r"|teach|walk me through|why|how|what about)\b",
    re.I,
)
MAX_LIGHT_WORDS = 4


def is_light(message: str) -> bool:
    """True for greetings, acknowledgements and very short lookups."""
    text = message.strip()
    if ACK_RE.match(text):
        return True
    return len(text.split()) <= MAX_LIGHT_WORDS and not SUBSTANTIVE_RE.search(text)

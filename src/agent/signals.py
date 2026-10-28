"""Infers the observed profile from what the user actually asks for.

Deterministic keyword signals, weighted by recency (guideline 2): a signal in the
newest message weighs 1.0, one message older DECAY, two older DECAY**2, and so on.
A value becomes a habit once its weight reaches MIN_WEIGHT and strictly leads, which
takes two signals within roughly the last five messages, so a single offhand
"elaborate" doesn't rewrite someone's profile. A habit whose weight falls below
FADE_WEIGHT is forgotten.
"""
import re
from collections import Counter

DECAY = 0.85
MIN_WEIGHT = 1.5
FADE_WEIGHT = 0.5

# Per dimension: (value, pattern), checked in order; first match wins. Negated
# phrasings ("no bullets", "less formal") come before the bare keyword they contain.
#
# Patterns only match phrasings *about the answer* ("be more formal", "make it shorter"),
# never topic words: "formal verification", "a professional developer", "redo a git
# commit" and "cite a website in APA" must not count as style evidence.
_IT = r"(it|this|that|them)"
_RAW_PATTERNS = {
    "length": [
        ("concise", rf"(make {_IT} |be |a bit |much )?shorter|too long|briefly|be brief|keep {_IT} (short|brief)"
                    r"|tl;?dr|one[- ]liner|less detail|shorter answers?"),
        ("detailed", rf"more details?|in (more )?detail|elaborate|expand on {_IT}|go deeper|in depth|tell me more"
                     rf"|too short|make {_IT} longer|longer answers?|be more thorough|more thorough|step[- ]by[- ]step"),
    ],
    "tone": [
        ("casual", rf"less formal|more casual|be casual|casual tone|(more |be )?informal|like a friend|friendlier"
                   rf"|(more )?relaxed tone|make {_IT} casual"),
        ("formal", rf"more formal|be formal|formal tone|make {_IT} formal|(more|be) professional|professional tone"
                   r"|more polite"),
    ],
    "sourcing": [
        ("minimal", r"no (sources|citations|links|references)|without (sources|citations)|skip the (sources|citations)"),
        ("full", rf"cite {_IT}|cite (your|the) sources|(sources|citations|references)\?"
                 r"|(add|include|with|show|give me|list)( the| some| your)? (sources|citations|references|links)"
                 rf"|back {_IT} up"),
    ],
    "structure": [
        ("paragraph", r"no bullets|no (bullet points|lists)|in paragraphs?|as prose|full sentences"),
        ("bullet", r"(use |in |with )?bullet( point)?s|as a list|numbered list|in points"),
    ],
    "revision": [
        ("rejects", rf"(redo|rewrite) {_IT}|try again|not what i (wanted|asked for|meant)|that'?s (wrong|not right)"),
        ("accepts", r"(that'?s|this is|looks) perfect|perfect,? thanks|looks good|that works"
                    r"|exactly what i (wanted|needed)|great,? thanks"),
    ],
}

PATTERNS = {
    dim: [(value, re.compile(rf"\b(?:{pattern})(?!\w)", re.I)) for value, pattern in rules]
    for dim, rules in _RAW_PATTERNS.items()
}

BUILT_IN = {dim: list(rules) for dim, rules in PATTERNS.items()}


def add_rule(dim: str, value: str, pattern: str) -> None:
    """A rule learned from a reviewed failure. It goes first, so it can beat a built-in rule it corrects."""
    compiled = re.compile(rf"\b(?:{pattern})(?!\w)", re.I)
    if all(p.pattern != compiled.pattern for _, p in PATTERNS[dim]):
        PATTERNS[dim].insert(0, (value, compiled))


def reset_rules() -> None:
    for dim, rules in BUILT_IN.items():
        PATTERNS[dim] = list(rules)


# Behavior: what the user did with an answer. Accepting it (thumbs up, copy) or rejecting it
# (thumbs down, regenerate) is revision evidence; a thumbs-down reason adds the dimension it names.
FEEDBACK_SIGNALS = {
    "up": {"revision": "accepts"},
    "copy": {"revision": "accepts"},
    "down": {"revision": "rejects"},
    "regenerate": {"revision": "rejects"},
}
DOWN_REASONS = {
    "too_long": {"length": "concise"},
    "too_short": {"length": "detailed"},
    "too_formal": {"tone": "casual"},
    "too_casual": {"tone": "formal"},
    "needs_sources": {"sourcing": "full"},
    "too_many_sources": {"sourcing": "minimal"},
    "use_bullets": {"structure": "bullet"},
    "use_paragraphs": {"structure": "paragraph"},
    "wrong": {},
}


def feedback_signals(events: list[dict]) -> dict[str, str]:
    """Signals from everything the user did with one answer (oldest first; later wins)."""
    signals: dict[str, str] = {}
    for event in events:
        signals.update(FEEDBACK_SIGNALS.get(event["kind"], {}))
        if event["kind"] == "down":
            signals.update(DOWN_REASONS.get(event.get("reason") or "", {}))
    return signals


class SignalExtractor:
    """Turns raw user messages into style evidence."""

    def extract(self, message: str) -> dict[str, str]:
        """Return the style signals in one message, e.g. {"length": "detailed"}."""
        signals = {}
        for dim, rules in PATTERNS.items():
            for value, pattern in rules:
                if pattern.search(message):
                    signals[dim] = value
                    break
        return signals

    def weigh(self, messages: list[str]) -> dict[str, Counter]:
        """Recency-weighted evidence per dimension and value. `messages` is oldest first."""
        return self.weigh_signals([self.extract(m) for m in messages])

    def infer_profile(self, messages: list[str], min_weight: float = MIN_WEIGHT) -> dict[str, str]:
        """Dimensions with a clear, recent habit. `messages` is oldest first."""
        return self.infer_from_signals([self.extract(m) for m in messages], min_weight)

    @staticmethod
    def weigh_signals(turns: list[dict]) -> dict[str, Counter]:
        """Recency-weighted evidence from per-turn signal dicts (oldest first). A turn's
        signals can come from its words, from what the user did with its answer, or both."""
        weights: dict[str, Counter] = {}
        for age, signals in enumerate(reversed(turns)):
            for dim, value in signals.items():
                weights.setdefault(dim, Counter())[value] += DECAY ** age
        return weights

    def infer_from_signals(self, turns: list[dict], min_weight: float = MIN_WEIGHT) -> dict[str, str]:
        inferred = {}
        for dim, counter in self.weigh_signals(turns).items():
            (value, weight), *rest = counter.most_common()
            if weight >= min_weight and (not rest or rest[0][1] < weight):
                inferred[dim] = value
        return inferred

"""Infers the observed profile from what the user actually asks for.

Deterministic keyword signals, voted over the user's recent messages. A dimension
only changes once a value has MIN_EVIDENCE votes and strictly leads, so a single
offhand "elaborate" doesn't rewrite someone's profile.
"""
import re
from collections import Counter

MIN_EVIDENCE = 2

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

    def infer_profile(self, messages: list[str], min_evidence: int = MIN_EVIDENCE) -> dict[str, str]:
        """Vote across messages; return only dimensions with a clear, repeated winner."""
        votes: dict[str, Counter] = {}
        for message in messages:
            for dim, value in self.extract(message).items():
                votes.setdefault(dim, Counter())[value] += 1

        inferred = {}
        for dim, counter in votes.items():
            (value, count), *rest = counter.most_common()
            if count >= min_evidence and (not rest or rest[0][1] < count):
                inferred[dim] = value
        return inferred

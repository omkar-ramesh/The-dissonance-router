"""Claims in a model answer that the agent can't vouch for (guideline 6).

The agent has no search tool, so a link or DOI in an answer was never supplied by anything
the agent did: it comes from the model's memory and may be wrong or invented. The agent
flags these instead of passing them along as if they were checked.
"""
import re

# A link needs a real host ("https://" alone, e.g. inside code, is not a citation).
_SOURCE = re.compile(r"https?://[a-z0-9-]+(?:\.[a-z0-9-]+)+[^\s)>\]\"'`]*|\bwww\.[a-z0-9-]+\.[a-z0-9.-]+[^\s)>\]\"'`]*"
                     r"|\b10\.\d{4,9}/[^\s)>\]\"'`]+", re.I)
# Reserved for illustrations (RFC 2606/6761): "https://example.com" in an HTTPS explanation
# is an example, not a source. Found when the checker flagged exactly that in a live answer.
_EXAMPLE_HOST = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+\.)*(example\.(com|org|net)|localhost|[a-z0-9-]+\.(test|example|invalid))\b", re.I)


def find_unbacked_sources(text: str, allowed=()) -> list:
    """Links and DOIs in an answer that no retrieval step supplied."""
    allowed = {a.rstrip("/") for a in allowed}
    return [s for s in _SOURCE.findall(text or "")
            if s.rstrip("/.,;") not in allowed and not _EXAMPLE_HOST.match(s)]

"""Demo profiles: five scripted conversations anyone can replay with one click.

Two show the router working, two reproduce known failures (with reason codes), one plants a
false claim to show it gets caught. The script is fixed; the answers come from the real
configured models, and the outcome is graded from what actually happened, not assumed.
`python -m src.eval.demos [--save]` runs all five and saves one report per persona.
"""
import secrets
import sys

from src.agent import hope
from src.agent.core import Agent
from src.agent.llm_client import LLMResult

FAKE_LINK = "https://arxiv.org/abs/2410.99999"


class Liar:
    """Stand-in that adds a made-up citation to every real answer. Disclosed on the demo card."""

    def __init__(self, llm):
        self.llm = llm

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def chat(self, tier, messages, *args, **kw):
        r = self.llm.chat(tier, messages, *args, **kw)
        return LLMResult(**{**r.__dict__, "text": f"{r.text}\n\nSource: {FAKE_LINK}"})


def _tiers(turns):
    return [t["tier"] for t in turns]


DEMOS = [
    {"id": "aligned", "name": "Hope", "kind": "success", "title": "Says short, asks short",
     "guideline": "Aligned users stay on the cheapest tier.",
     "said": {"length": "concise"},
     "messages": ["Explain recursion, keep it short", "What is a hash map? Be brief."],
     "expect": "Every answer from Nano.",
     "check": lambda turns: all(t == "nano" for t in _tiers(turns)), "code": "OVER_ROUTED"},
    {"id": "contradiction", "name": "Jordan", "kind": "success", "title": "Says short, keeps asking for detail",
     "guideline": "A repeated contradiction escalates, and the reason is shown.",
     "said": {"length": "concise"},
     "messages": ["Explain attention, in more detail please", "Can you elaborate on the softmax?"],
     "expect": "The last answer moves up from Nano, with length named as the reason.",
     "check": lambda turns: turns[-1]["tier"] != "nano" and "length" in turns[-1]["reason"], "code": "UNDER_ROUTED"},
    {"id": "implicit_complaint", "name": "Casey", "kind": "failure", "title": "Complains without saying the word",
     "guideline": "Known gap: a complaint with no style words may be missed.",
     "said": {"length": "detailed"},
     "messages": ["Explain TCP handshakes", "ugh, that's a wall of text, I'm not reading all that"],
     "expect": "Ideally the complaint counts as a sign of wanting shorter answers.",
     "check": lambda turns: turns[-1]["tier"] != "nano", "code": "MISSED_SIGNAL"},
    {"id": "negation", "name": "Sam", "kind": "failure", "title": "\"Don't keep it short\"",
     "guideline": "Known gap: negated requests can be read backwards.",
     "said": {"length": "concise"},
     "messages": ["Don't keep it short this time, I need every step of backprop"],
     "expect": "Ideally read as wanting detail, which contradicts \"concise\".",
     "check": lambda turns: turns[-1]["learned"].get("length") != "concise", "code": "MISREAD_NEGATION"},
    {"id": "false_claim", "name": "Kai", "kind": "false_claim", "title": "A model invents a source",
     "guideline": "Links the agent can't back up are flagged, never passed off as real.",
     "said": {"sourcing": "full"}, "liar": True,
     "messages": ["What's the main idea behind transformers? Cite your source."],
     "expect": f"The made-up link ({FAKE_LINK}) is flagged as unverified.",
     "check": lambda turns: FAKE_LINK in turns[-1]["unverified"], "code": "UNVERIFIED_SOURCE"},
]
BY_ID = {d["id"]: d for d in DEMOS}


def card(d: dict) -> dict:
    return {k: d.get(k) for k in ("id", "name", "kind", "title", "guideline", "said", "messages", "expect")} | {
        "user": "persona-" + d["name"].lower(),
        "stand_in": "A wrapper adds a made-up citation to the real model's answer." if d.get("liar") else ""}


def run(demo_id: str, agent: Agent) -> dict:
    d = BY_ID[demo_id]
    user = f"demo-{demo_id}-{secrets.token_hex(4)}"     # fresh profile every run
    if d.get("liar"):
        agent = Agent(agent.memory.db_path, Liar(agent.llm), agent.signals, str(agent.memory.folder))
    for dim, value in d["said"].items():
        agent.set_stated(user, dim, value)
    turns = []
    for m in d["messages"]:
        r = agent.respond(user, m)
        turns.append({"say": m, "tier": r["model_chosen"], "reason": r["routing_reason"],
                      "answer": r["response"][:600], "unverified": r.get("unverified_sources", []),
                      "learned": agent.profiles.get_observed_profile(user),
                      "tokens": r["usage"], "latency_ms": r["latency_ms"], "model": r["model_id"]})
    ok = bool(d["check"](turns))
    label = {"success": "PASS" if ok else "FAIL", "failure": "FIXED" if ok else "KNOWN GAP",
             "false_claim": "CAUGHT" if ok else "MISSED"}[d["kind"]]
    out = {**card(d), "user": user, "label": label, "as_hoped": ok, "code": "" if ok else d["code"], "turns": turns}
    if not ok or d["kind"] == "false_claim":      # Hope reviews every failure and every planted fake: why, the proposed fix, a zero-token check. Nothing is changed yet.
        last = turns[-1]
        out["review"] = hope.review(d["code"], last["say"], last["answer"], last["unverified"])
    hope.record(agent.memory.db_path, d["name"].lower(), d["code"], ok)
    return out


if __name__ == "__main__":
    from src import config
    from src.api.dependencies import get_agent
    agent = get_agent()
    results = []
    for d in DEMOS:
        res = run(d["id"], agent)
        results.append(res)
        print(f"{res['label']:<10} {d['id']:<20} {' > '.join(t['tier'] for t in res['turns'])} {res['code']}")
    if "--save" in sys.argv:
        from datetime import datetime
        from src.eval import reports
        place = reports.where(config.TIER_PROVIDERS)
        stamp = f"{datetime.now():%Y%m%d-%H%M}"
        for res in results:
            print("saved:", reports.save({"demos": [res], "models": config.MODEL_TIERS, "simulated": False},
                                         f"demos-{res['name'].lower()}-{place}-{stamp}", config.TIER_PROVIDERS))

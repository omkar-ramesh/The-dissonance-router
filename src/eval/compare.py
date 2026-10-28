"""Is Ultra worth it? A blind comparison of Nano and Ultra answers on contradiction turns.

The router escalates to Ultra when what a user says contradicts what they do. That only pays
if Ultra's answer serves the user better than Nano's would have. Here both get the identical
Ultra prompt (same reconciliation instructions), so only the model differs, and a third model
judges which answer serves the user better without knowing which model wrote it. Each pair
is judged twice with the order swapped; a verdict that flips with the order counts as a tie.

    python -m src.eval compare                  # contestants nano vs ultra, judged by super
    python -m src.eval compare --judge nano     # any tier that isn't a contestant

Limits: one judge model, six cases, and locally the tiers are stand-ins for Nemotron Super
and Ultra. It answers "on these cases, with these models", not "in general".
"""
import argparse
import json
import random
import re
from pathlib import Path

from src import config
from src.agent.llm_client import LLMError
from src.agent.profiles import DIMENSIONS
from src.agent.prompts import build_messages
from src.agent.signals import SignalExtractor
from src.eval.claims import find_unbacked_sources
from src.eval.runner import _cost

PLAIN = {
    "concise": "short answers", "detailed": "detailed answers", "formal": "a formal tone", "casual": "a casual tone",
    "paragraph": "paragraphs, not bullet points", "bullet": "bullet points", "minimal": "no sources",
    "full": "sources for claims", "accepts": "a ready-to-use answer", "rejects": "options, since first drafts rarely fit",
}
SERVES_BOTH = {
    "length": "a one or two sentence direct answer first, then the depth",
    "tone": "the tone they respond to, with precise wording",
    "structure": "a short paragraph first, bullets only for steps or lists",
    "sourcing": "a clean body with sources gathered at the end, and no invented sources",
    "revision": "key assumptions stated up front and one alternative to pick instead",
}
# (id, dimension, stated, observed, message)
CASES = [
    ("gradient_descent", "length", "concise", "detailed", "Can you elaborate on how gradient descent finds a minimum?"),
    ("hash_collisions", "length", "concise", "detailed", "Go deeper on how a hash map handles collisions"),
    ("rest_api", "tone", "formal", "casual", "explain what a REST API is"),
    ("venv_setup", "structure", "paragraph", "bullet", "How do I set up a Python virtual environment?"),
    ("crisis_2008", "sourcing", "minimal", "full", "Why did the 2008 financial crisis happen?"),
    ("shop_name", "revision", "accepts", "rejects", "Suggest a name for a coffee shop that also sells books"),
]

JUDGE_SYSTEM = ('You compare two answers written for one user and decide which serves that user better. '
                'Reply with only JSON: {"winner": "A" | "B" | "tie", "why": "<one sentence>"}')
_WINNER = re.compile(r'"winner"\s*:\s*"(A|B|tie)"', re.I)


def analysis_for(dim, stated, observed, message):
    stated_p = {d: "balanced" for d in DIMENSIONS}
    observed_p = dict(stated_p)
    stated_p[dim], observed_p[dim] = stated, observed
    return {"stated_profile": stated_p, "observed_profile": observed_p, "model_chosen": "ultra",
            "conflicts_detected": {d: (0.8 if d == dim else 0.0) for d in DIMENSIONS},
            "turn_signals": SignalExtractor().extract(message)}


def answer(llm, tier, messages):
    r = llm.chat(tier, messages)
    return {"tier": tier, "model": r.model, "text": r.text, "latency_ms": r.latency_ms,
            "output_tokens": r.output_tokens, "reasoning_tokens": r.reasoning_tokens,
            "cost_usd": _cost(tier, r.input_tokens, r.output_tokens),
            "opens_with_tldr": r.text.lstrip(" \t\r\n*_#>-`").lower().startswith("tl;dr"),
            "words": len(r.text.split()), "unbacked_sources": find_unbacked_sources(r.text)}


def judge_once(llm, judge_tier, case, a_text, b_text):
    _, dim, stated, observed, message = case
    prompt = (f"The user said they prefer {PLAIN[stated]}, but their behavior shows they actually want "
              f"{PLAIN[observed]}.\nTheir message: {message}\n"
              f"The best answer serves both: {SERVES_BOTH[dim]}. It must also be correct and must not invent facts "
              f"or sources. Ignore length differences that don't matter to this user.\n\n"
              f"Answer A:\n{a_text[:6000]}\n\nAnswer B:\n{b_text[:6000]}\n\nWhich answer serves this user better?")
    r = llm.chat(judge_tier, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                 temperature=0.0)
    m = _WINNER.search(r.text)
    why = re.search(r'"why"\s*:\s*"([^"]*)"', r.text)
    return (m.group(1).upper() if m and m.group(1).lower() != "tie" else ("tie" if m else "invalid"),
            why.group(1) if why else r.text.strip()[:200])


def run(llm, contestants=("nano", "ultra"), judge_tier="super", only=None, seed=7):
    if judge_tier in contestants:
        raise ValueError("the judge must not be one of the contestants")
    rng = random.Random(seed)
    a, b = contestants
    results = []
    for case in CASES:
        if only and case[0] not in only:
            continue
        cid, dim, stated, observed, message = case
        messages = build_messages(analysis_for(dim, stated, observed, message), [], message)
        answers = {tier: answer(llm, tier, messages) for tier in contestants}
        first = rng.choice([a, b])            # which model sits in slot A on the first judging
        verdicts = []
        for slot_a in (first, b if first == a else a):
            slot_b = b if slot_a == a else a
            winner, why = judge_once(llm, judge_tier, case, answers[slot_a]["text"], answers[slot_b]["text"])
            verdicts.append({"A": slot_a, "B": slot_b, "winner": {"A": slot_a, "B": slot_b}.get(winner, winner),
                             "why": why})
        picks = [v["winner"] for v in verdicts]
        if "invalid" in picks:
            outcome = "invalid"
        elif picks[0] == picks[1]:
            outcome = picks[0]                 # a model name or "tie"
        else:
            outcome = "inconsistent"           # flipped with the order: position bias, counts as a tie
        results.append({"id": cid, "dimension": dim, "stated": stated, "observed": observed, "message": message,
                        "answers": answers, "verdicts": verdicts, "outcome": outcome})
    return summarize(results, contestants, judge_tier, llm)


def summarize(results, contestants, judge_tier, llm):
    a, b = contestants
    count = lambda o: sum(r["outcome"] == o for r in results)
    wins = {a: count(a), b: count(b)}
    ties = count("tie") + count("inconsistent")
    avg = lambda tier, key: sum(r["answers"][tier][key] for r in results) / max(1, len(results))
    cost_ratio = avg(b, "cost_usd") / avg(a, "cost_usd") if avg(a, "cost_usd") else None
    if wins[b] - wins[a] >= 2:
        verdict = f"{b.title()}'s answers were preferred: escalating paid off on these cases."
    elif wins[a] >= wins[b]:
        verdict = (f"{b.title()} was not better than {a.title()} here. The escalation claim should be narrowed "
                   f"(or the prompt alone may be doing the work).")
    else:
        verdict = f"Mixed: {b.title()} won slightly more often, not clearly enough to justify its cost on its own."
    return {
        "contestants": list(contestants), "judge": judge_tier,
        "models": {t: config.MODEL_TIERS.get(t, t) for t in (*contestants, judge_tier)},
        "simulated": bool(getattr(llm, "simulated", False)),
        "wins": wins, "ties": ties, "invalid": count("invalid"), "cases": len(results),
        "avg": {t: {k: round(avg(t, k), 6) for k in ("cost_usd", "latency_ms", "output_tokens", "words")}
                for t in contestants},
        "tldr_on_length_cases": {t: f"{sum(r['answers'][t]['opens_with_tldr'] for r in results if r['dimension'] == 'length')}"
                                    f"/{sum(r['dimension'] == 'length' for r in results)}" for t in contestants},
        "unbacked_sources": {t: sum(len(r["answers"][t]["unbacked_sources"]) for r in results) for t in contestants},
        "cost_ratio": round(cost_ratio, 1) if cost_ratio else None,
        "verdict": verdict,
        "results": results,
    }


def to_markdown(rep):
    a, b = rep["contestants"]
    lines = [f"# Is {b.title()} worth it? Blind comparison", "",
             f"Contestants: {a} (`{rep['models'][a]}`) vs {b} (`{rep['models'][b]}`). "
             f"Judge: {rep['judge']} (`{rep['models'][rep['judge']]}`), order swapped, flips count as ties.", "",
             f"**{rep['verdict']}**", "",
             f"| | {a} | {b} |", "|---|---|---|",
             f"| Cases won | {rep['wins'][a]} | {rep['wins'][b]} |",
             f"| Avg cost per answer (Token Factory prices) | ${rep['avg'][a]['cost_usd']:.5f} | ${rep['avg'][b]['cost_usd']:.5f} |",
             f"| Avg latency | {rep['avg'][a]['latency_ms'] / 1000:.1f} s | {rep['avg'][b]['latency_ms'] / 1000:.1f} s |",
             f"| Opens with TL;DR (length cases) | {rep['tldr_on_length_cases'][a]} | {rep['tldr_on_length_cases'][b]} |",
             f"| Unbacked links | {rep['unbacked_sources'][a]} | {rep['unbacked_sources'][b]} |", "",
             f"Ties or order-dependent verdicts: {rep['ties']}. Unreadable verdicts: {rep['invalid']}.", "",
             "| Case | Conflict | Outcome | Judge's reasons |", "|---|---|---|---|"]
    for r in rep["results"]:
        reasons = " / ".join(v["why"].replace("|", "/")[:140] for v in r["verdicts"])
        lines.append(f"| {r['message']} | {r['dimension']}: says {r['stated']}, does {r['observed']} | {r['outcome']} | {reasons} |")
    lines += ["", "Limits: one judge model, six cases; locally the tiers are stand-ins, not Nemotron Super/Ultra."]
    return "\n".join(lines) + "\n"


def main(argv):
    p = argparse.ArgumentParser(prog="python -m src.eval compare")
    p.add_argument("--judge", default="super", choices=["nano", "super", "ultra"])
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--out", default="data/eval")
    p.add_argument("--save", action="store_true", help="also save a card + detail file in docs/reports/")
    args = p.parse_args(argv)
    from src.agent.llm_client import LLMClient
    try:
        rep = run(LLMClient.from_config(), judge_tier=args.judge, only=set(args.only.split(",")) if args.only else None)
    except LLMError as e:
        print(f"model call failed: {e}")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out / "compare.md").write_text(to_markdown(rep), encoding="utf-8")
    a, b = rep["contestants"]
    if args.save:
        from datetime import datetime
        from src import config
        from src.eval import reports
        place = reports.where(config.TIER_PROVIDERS)
        print("saved:", reports.save(rep, f"compare-{args.only.replace(',', '+') + '-' if args.only else ''}{'-vs-'.join(rep['contestants'])}-{place}-{datetime.now():%Y%m%d-%H%M}",
                                     config.TIER_PROVIDERS))
    print(rep["verdict"])
    print(f"  wins {a} {rep['wins'][a]} / {b} {rep['wins'][b]} / ties {rep['ties']} / invalid {rep['invalid']}")
    print(f"report: {out}/compare.md")
    return 0

"""Runs the scenarios, grades them with reason codes, and prices three strategies."""
import json
import tempfile
import time
from pathlib import Path

from src import config
from src.agent.core import Agent
from src.agent.extractor import NanoExtractor
from src.agent.llm_client import LLMError
from src.eval import scenarios as sc
from src.eval.claims import find_unbacked_sources, verify_explanations
from src.eval.sims import ForcedTier, LyingLLM, SimLLM

RANK = {"nano": 0, "super": 1, "ultra": 2}
STRATEGIES = {"always-nano": "nano", "always-ultra": "ultra", "router": None}
USER = "eval-user"


def _cost(tier, input_tokens, output_tokens):
    price = config.PRICES[tier]
    return (input_tokens * price["in"] + output_tokens * price["out"]) / 1_000_000


class Tally:
    """Wraps a client and remembers each call's tokens, so cost can be summed per run."""

    def __init__(self, inner):
        self.inner = inner
        self.simulated = getattr(inner, "simulated", False)
        self.calls = []

    def chat(self, tier, messages, temperature=0.7):
        result = self.inner.chat(tier, messages, temperature)
        self.calls.append(result)
        return result


def run_scenario(scenario, llm, strategy=None, graded=True, reader=None):
    """Play one scenario. Returns the turns that happened and, if graded, the findings.

    `reader` is the signal reader to use: None for keywords, or a NanoExtractor (shared
    across scenarios so each message is read once per run).
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        tally = Tally(ForcedTier(llm, strategy))
        agent = Agent(db, tally, reader)
        for dim, value in scenario.stated.items():
            agent.profiles.update_stated(USER, dim, value)

        turns, findings = [], []
        for i, turn in enumerate(scenario.turns):
            def flag(code, detail):
                findings.append({"turn": i + 1, "say": turn.say, "code": code, "detail": detail})

            if turn.action:
                agent.accept_observed(USER, turn.action[1])
            before = len(tally.calls)
            try:
                result = agent.respond(USER, turn.say)
            except LLMError as exc:
                flag(sc.BACKEND_FAILED, str(exc))
                continue
            call = tally.calls[before]
            served = call.tier
            reads = result.get("reader_stats") or []
            turns.append({
                "say": turn.say, "ideal": turn.ideal, "router_chose": result["model_chosen"], "served": served,
                "reason": result["routing_reason"], "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens, "latency_ms": call.latency_ms,
                "cost_usd": _cost(served, call.input_tokens, call.output_tokens), "response": result["response"],
                # the signal reader's own calls, counted rather than hidden
                "reader_cost_usd": sum(_cost("nano", r["input_tokens"], r["output_tokens"]) for r in reads),
                "reader_latency_ms": sum(r["latency_ms"] for r in reads),
            })
            if graded:
                _grade(agent, db, turn, result, served, tally.simulated, flag)
        return {"turns": turns, "findings": findings, "simulated": tally.simulated}


def _grade(agent, db, turn, result, served, simulated, flag):
    chose = result["model_chosen"]
    if turn.ideal and chose != turn.ideal:
        under = RANK[chose] < RANK[turn.ideal]
        flag(sc.UNDER_ROUTED if under else sc.OVER_ROUTED, f"routed to {chose}, a good router uses {turn.ideal}")
    observed = agent.profiles.get_observed_profile(USER)
    for dim, expected in turn.observed_after.items():
        actual = observed[dim]
        if actual == expected:
            continue
        if expected == "balanced" or actual not in ("balanced", expected):
            flag(sc.FALSE_SIGNAL, f"observed {dim} is '{actual}', expected '{expected}'")
        else:
            flag(sc.MISSED_SIGNAL, f"observed {dim} is '{actual}', expected '{expected}'")
    if turn.expect_suggestion is not None and bool(result["suggestion"]) != turn.expect_suggestion:
        flag(sc.MISSING_SUGGESTION, "expected an offer to update the preference" if turn.expect_suggestion
             else "offered a change nothing called for")
    # The model's formatting can only be judged on a real model: the stand-in complies by construction.
    if turn.expect_format and not simulated and not opens_with_tldr(result["response"]):
        flag(sc.IGNORED_FORMAT, "Ultra answer does not open with a TL;DR line")
    for problem in verify_explanations(db, USER, result["routing_reason"], result["suggestion"]):
        flag(sc.FALSE_CLAIM, problem)
    for source in find_unbacked_sources(result["response"]):
        flag(sc.UNVERIFIED_SOURCE, f"answer cites {source}, which nothing supplied")


def opens_with_tldr(text):
    """True if the first line is a TL;DR, ignoring markdown decoration (**TL;DR:**, # TL;DR, > TL;DR)."""
    return text.lstrip(" \t\r\n*_#>-`").lower().startswith("tl;dr")


def judge(scenario, findings, reader="keywords"):
    """Compare what happened with what the scenario expected.

    The known failures are limits of the keyword reader. Run with the Nano reader they are
    an experiment, not a regression gate: the label says whether Nano fixed them.
    """
    got = {f["code"] for f in findings}
    expected = set(scenario.expect_codes)
    if scenario.kind in ("success", "heldout"):
        return ("PASS", True) if not got else ("FAIL", False)
    if scenario.kind == "failure":
        if reader != "keywords":
            return ("FIXED", True) if not got else ("STILL FAILS", True)
        if expected <= got:
            return "KNOWN GAP", True
        return ("FIXED?" if not got else "DIFFERENT FAILURE"), False
    return ("CAUGHT", True) if expected <= got else ("MISSED", False)


def self_test(llm=None):
    """Inject false claims into a real explanation; the checker must flag every one."""
    scenario = sc.BY_ID["classic_contradiction"]
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "self.db")
        agent = Agent(db, llm or SimLLM())
        agent.profiles.update_stated(USER, "length", "concise")
        for turn in scenario.turns:
            result = agent.respond(USER, turn.say)
        reason, suggestion = result["routing_reason"], result["suggestion"]
        assert not verify_explanations(db, USER, reason, suggestion), "the true explanation must pass"

        tampered = [
            ("wrong repeat count", reason.replace("seen 2x", "seen 9x"), suggestion),
            ("wrong stated value", reason.replace("you said 'concise'", "you said 'detailed'"), suggestion),
            ("wrong observed value", reason.replace("you behave 'detailed'", "you behave 'concise'"), suggestion),
            ("wrong score", reason.replace("score 0.80", "score 0.35"), suggestion),
            ("wrong tier", reason.replace("Escalating to Ultra", "Preferences aligned"), suggestion),
            ("wrong preference in suggestion", reason, suggestion.replace("'concise'", "'formal'", 1)),
        ]
        cases = [(name, bool(verify_explanations(db, USER, r, s))) for name, r, s in tampered]
        cases.append(("invented link in an answer", bool(find_unbacked_sources("see https://arxiv.org/abs/2401.99999"))))
        # sanity: each tampering really changed the text, otherwise it proves nothing
        changed = [t[1] != reason or t[2] != suggestion for t in tampered]
        assert all(changed), "a tampered claim was identical to the original"
    return {"caught": sum(ok for _, ok in cases), "total": len(cases), "cases": cases}


def run_all(llm, compare=True, only=None, reader="keywords", reader_llm=None):
    """Grade every scenario on `llm`, then price the three strategies on the success scenarios.

    `reader="nano"` reads messages with Nemotron Nano via `reader_llm` (a real client).
    """
    started = time.time()
    extractor = NanoExtractor(reader_llm) if reader == "nano" else None
    results = []
    for scenario in sc.SCENARIOS:
        if only and scenario.id not in only:
            continue
        model = LyingLLM() if scenario.model == "lying" else llm
        run = run_scenario(scenario, model, reader=extractor)
        label, as_expected = judge(scenario, run["findings"], reader)
        results.append({
            "id": scenario.id, "title": scenario.title, "kind": scenario.kind, "why": scenario.why,
            "label": label, "as_expected": as_expected, "expected_codes": sorted(scenario.expect_codes),
            "findings": run["findings"], "turns": run["turns"],
        })

    strategies = {}
    if compare:
        chosen = [s for s in sc.SCENARIOS if s.kind == "success" and (not only or s.id in only)]
        for name, tier in STRATEGIES.items():
            tally = {"cost_usd": 0.0, "matched": 0, "under": 0, "over": 0, "graded": 0, "latency_ms": 0,
                     "input_tokens": 0, "output_tokens": 0}
            for scenario in chosen:
                run = run_scenario(scenario, llm, strategy=tier, graded=False, reader=extractor)
                for turn in run["turns"]:
                    tally["cost_usd"] += turn["cost_usd"] + (turn["reader_cost_usd"] if name == "router" else 0)
                    tally["latency_ms"] += turn["latency_ms"]
                    tally["input_tokens"] += turn["input_tokens"]
                    tally["output_tokens"] += turn["output_tokens"]
                    if turn["ideal"]:
                        tally["graded"] += 1
                        d = RANK[turn["served"]] - RANK[turn["ideal"]]
                        tally["matched" if d == 0 else "under" if d < 0 else "over"] += 1
            strategies[name] = tally

    return {
        "simulated": bool(getattr(llm, "simulated", False)),
        "reader": reader,
        "reader_fallbacks": extractor.fallbacks if extractor else 0,
        "models": dict(config.MODEL_TIERS),
        "prices_per_1m_tokens": config.PRICES,
        "scenarios": results,
        "strategies": strategies,
        "self_test": self_test(),
        "explanations_checked": sum(len(r["turns"]) for r in results),
        "seconds": round(time.time() - started, 1),
    }


def summarize(report):
    """Headline numbers for the README and the console."""
    by_kind = {}
    for r in report["scenarios"]:
        k = by_kind.setdefault(r["kind"], {"total": 0, "as_expected": 0})
        k["total"] += 1
        k["as_expected"] += r["as_expected"]
    codes = {}
    for r in report["scenarios"]:
        for f in r["findings"]:
            codes[f["code"]] = codes.get(f["code"], 0) + 1
    return {"by_kind": by_kind, "codes": codes}


def write_report(report, out_dir, name="report"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"{name}.md").write_text(to_markdown(report), encoding="utf-8")


def to_markdown(report):
    lines = ["# Evaluation report", ""]
    if report["simulated"]:
        lines += ["> **Simulated run.** Routing and grading are real; model answers and token counts are "
                  "simulated, so format checks are skipped and costs only compare strategies with each other.", ""]
    else:
        lines += ["Models: " + ", ".join(f"{t} = `{m}`" for t, m in report["models"].items()), ""]
    summary = summarize(report)
    lines += ["## Results", "", "| Kind | As expected |", "|---|---|"]
    lines.insert(2, f"Signal reader: **{report.get('reader', 'keywords')}**\n")
    names = {"success": "Success cases", "failure": "Known limitations (as expected)",
             "heldout": "Held-out (written after tuning)", "false_claim": "False-claim checks (caught)"}
    for kind in ("success", "failure", "heldout", "false_claim"):
        if kind in summary["by_kind"]:
            k = summary["by_kind"][kind]
            lines.append(f"| {names[kind]} | {k['as_expected']} / {k['total']} |")
    st = report["self_test"]
    lines += [f"| Injected false claims caught | {st['caught']} / {st['total']} |", ""]

    for kind, heading in (("success", "Success"), ("failure", "Failure, with reason codes"),
                          ("heldout", "Held-out (written after tuning, never edited)"), ("false_claim", "False claims")):
        rows = [r for r in report["scenarios"] if r["kind"] == kind]
        if not rows:
            continue
        lines += [f"## {heading}", ""]
        for r in rows:
            lines += [f"### {r['title']} — {r['label']}", "", r["why"], ""]
            if r["findings"]:
                lines += ["| Turn | Code | Detail |", "|---|---|---|"]
                lines += [f"| {f['turn']} | `{f['code']}` | {f['detail']} |" for f in r["findings"]]
                lines.append("")
    if report["strategies"]:
        lines += ["## Cost vs fit", "",
                  "Success scenarios, three strategies. *Under* = a smaller model than needed; "
                  "*over* = a bigger one than needed (wasted spend).", "",
                  "| Strategy | Cost | Right tier | Under | Over | Tokens (in / out) |", "|---|---|---|---|---|---|"]
        for name, t in report["strategies"].items():
            lines.append(f"| {name} | ${t['cost_usd']:.4f} | {t['matched']}/{t['graded']} | {t['under']} | "
                         f"{t['over']} | {t['input_tokens']} / {t['output_tokens']} |")
        lines.append("")
    lines += ["## Reason codes", ""] + [f"- `{c}`: {d}" for c, d in sc.CODES.items()] + [""]
    return "\n".join(lines)

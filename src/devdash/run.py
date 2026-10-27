"""Test runner for the dev dashboard.

    python -m src.devdash.run --suite all        # pytest + live-model scenarios
    python -m src.devdash.run --suite pytest
    python -m src.devdash.run --suite scenarios  # real model calls (needs Ollama)

Every step is recorded as it happens (see store.py) and echoed to the terminal.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from src import config
from src.agent.core import Agent
from src.agent.llm_client import LLMClient
from src.devdash.store import RunRecorder

ROOT = Path(config.ROOT_DIR)
PYTEST_RESULT_RE = re.compile(r"^(?P<id>\S.*?::.*?) (?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
PYTEST_FAILURE_RE = re.compile(r"^(?:FAILED|ERROR) (?P<id>\S+?)(?: - (?P<msg>.*))?$")
TLDR_RE = re.compile(r"^\W*tl;?dr", re.I)


def _snippet(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


class Scenarios:
    """End-to-end checks against the real models: does the agent route and answer as designed?"""

    def __init__(self):
        self.llm = LLMClient.from_config()
        self.agent = Agent(os.path.join(tempfile.mkdtemp(), "live.db"), self.llm)
        self.last: dict = {}

    def steps(self) -> list[tuple[str, str, str, object]]:
        """(id, name, what it checks, function)"""
        return [
            ("live::aligned_user_routes_nano", "Aligned user routes to Nano",
             "A user whose stated and observed styles agree has no conflict, so the router should use the "
             "cheapest model (Nano).", self.aligned_nano),
            ("live::one_signal_not_evidence", "One 'more detail' request is not yet evidence",
             "The user states 'concise'. A single request for more detail is one data point, not a pattern, "
             "so the observed profile must not change yet and Nano should still answer.", self.one_signal),
            ("live::second_signal_escalates", "Second signal: contradiction escalates to Ultra",
             "A second request for detail confirms the pattern: observed 'detailed' now contradicts stated "
             "'concise'. The router should escalate to Ultra and ask whether to update the preference.",
             self.escalates),
            ("live::ultra_leads_with_tldr", "Ultra answer leads with a TL;DR",
             "Ultra's job is to serve both sides of the contradiction: a one-line TL;DR for the 'concise' "
             "preference, then the depth the user keeps asking for. Checks the Ultra answer from the step above.",
             self.tldr),
            ("live::quick_message_stays_nano", "A quick 'thanks' stays on Nano while the conflict is open",
             "Guideline 4: a thank-you has no style choices to make, so it should not cost an Ultra call even "
             "though the length contradiction is still open.", self.quick),
            ("live::accepting_returns_nano", "Accepting the suggestion returns to Nano",
             "Once the user accepts 'update my preference to detailed', the contradiction is resolved and "
             "the next request should go back to Nano.", self.accepts),
            ("live::unstated_pref_one_signal", "Unstated preference: first signal stays on Nano",
             "A new user with no stated preference asks for depth once. One signal is not enough to learn from.",
             self.partial_first),
            ("live::unstated_pref_routes_super", "Unstated preference: emerging pattern routes to Super",
             "A second request makes a preference emerge that the user never stated (a partial gap, score 0.40). "
             "That deserves Super, never Ultra.", self.partial_second),
        ]

    def _ask(self, user: str, message: str) -> dict:
        self.last = self.agent.respond(user, message)
        return self.last

    @staticmethod
    def _verdict(problems: list[str], summary: str, result: dict):
        stats = result["stats"]
        return not problems, ("; ".join(problems) if problems else summary), stats

    def aligned_nano(self):
        r = self._ask("aligned", "What is 2+2? Reply with just the number.")
        problems = []
        if r["model_chosen"] != "nano":
            problems.append(f"expected nano, got {r['model_chosen']}")
        if "4" not in r["response"]:
            problems.append("answer does not contain 4")
        return self._verdict(problems, f"nano answered: {_snippet(r['response'], 80)}", r)

    def one_signal(self):
        self.agent.profiles.update_stated("conflict", "length", "concise")
        r = self._ask("conflict", "Explain how attention works in transformers, in more detail please")
        problems = []
        if r["model_chosen"] != "nano":
            problems.append(f"expected nano, got {r['model_chosen']}")
        if r["observed_updates"]:
            problems.append(f"profile changed too early: {r['observed_updates']}")
        return self._verdict(problems, "stated=concise, one request for detail, still nano", r)

    def escalates(self):
        r = self._ask("conflict", "Can you elaborate on the softmax scaling?")
        problems = []
        if r["model_chosen"] != "ultra":
            problems.append(f"expected ultra, got {r['model_chosen']}")
        if r["observed_updates"] != {"length": "detailed"}:
            problems.append(f"expected observed length=detailed, got {r['observed_updates']}")
        if "concise" not in r["routing_reason"]:
            problems.append("routing reason does not name the stated preference")
        if not r["suggestion"]:
            problems.append("no 'update your preference?' suggestion")
        return self._verdict(problems, r["routing_reason"], r)

    def tldr(self):
        text = self.last.get("response", "")
        ok = bool(TLDR_RE.match(text))
        detail = ("starts with TL;DR: " if ok else "did not lead with TL;DR: ") + _snippet(text, 160)
        return ok, detail, None

    def quick(self):
        r = self._ask("conflict", "thanks!")
        problems = []
        if r["model_chosen"] != "nano":
            problems.append(f"expected nano for a quick message, got {r['model_chosen']}")
        if not r["suggestion"]:
            problems.append("the open conflict's suggestion disappeared")
        return self._verdict(problems, "quick message stayed on nano; the conflict is still open", r)

    def accepts(self):
        self.agent.accept_observed("conflict", "length")
        r = self._ask("conflict", "Thanks. Name one primary color.")
        problems = []
        if r["model_chosen"] != "nano":
            problems.append(f"expected nano after accepting, got {r['model_chosen']}")
        return self._verdict(problems, "stated preference updated to detailed; conflict closed", r)

    def partial_first(self):
        r = self._ask("partial", "Elaborate on how tokenizers work")
        problems = [] if r["model_chosen"] == "nano" else [f"expected nano, got {r['model_chosen']}"]
        return self._verdict(problems, "no stated preference, one signal", r)

    def partial_second(self):
        r = self._ask("partial", "Go deeper on byte-pair encoding")
        problems = []
        if r["model_chosen"] != "super":
            problems.append(f"expected super, got {r['model_chosen']} (score {r['conflict_score']})")
        return self._verdict(problems, r["routing_reason"], r)


def run_scenarios(rec: RunRecorder) -> None:
    scenarios = Scenarios()
    steps = scenarios.steps()
    rec.add_steps([(step_id, name, "live", what) for step_id, name, what, _ in steps])
    rec.log(f"live scenarios: {len(steps)} checks against real models "
            + ", ".join(f"{t}={m}" for t, m in rec.state["models"].items()))
    for step_id, name, _, fn in steps:
        if not rec.wait_while_stopped():
            raise RuntimeError("stopped for more than 30 minutes")
        rec.step_start(step_id)
        try:
            ok, detail, stats = fn()
        except Exception as e:  # a model outage must turn one step red, not kill the run
            ok, detail, stats = False, f"{type(e).__name__}: {e}", None
        rec.step_end(step_id, ok, detail, stats)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def run_pytest(rec: RunRecorder) -> None:
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--color=no"]
    collected = subprocess.run(base + ["--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True)
    ids = [line.strip() for line in collected.stdout.splitlines() if "::" in line]
    rec.add_steps([(i, i.split("/")[-1].replace("::", " › "), "pytest") for i in ids])
    rec.log(f"pytest: collected {len(ids)} unit tests")

    proc = subprocess.Popen(
        base + ["-v", "--no-header", "--tb=short", "-rfE"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        match = PYTEST_RESULT_RE.match(line)
        if match:
            rec.add_steps([(match["id"], match["id"].split("/")[-1].replace("::", " › "), "pytest")])
            rec.step_start(match["id"])
            rec.step_end(match["id"], match["status"] in ("PASSED", "SKIPPED", "XFAIL"))
            continue
        failure = PYTEST_FAILURE_RE.match(line.strip())
        if failure:
            rec.annotate(failure["id"], failure["msg"] or "see pytest output")
    proc.wait()
    rec.log(f"pytest finished (exit code {proc.returncode})", "info")
    print(f"  pytest exit code {proc.returncode}")
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"pytest did not complete (exit code {proc.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["all", "pytest", "scenarios"], default="all")
    args = parser.parse_args()

    models = {tier: f"{config.MODEL_TIERS[tier]} ({config.TIER_PROVIDERS[tier]})" for tier in config.MODEL_TIERS}
    rec = RunRecorder(args.suite, models, prices=config.PRICES)
    rec.log(f"run {rec.id} started (suite: {args.suite})")
    print(f"run {rec.id}")
    try:
        if args.suite in ("all", "pytest"):
            run_pytest(rec)
        if args.suite in ("all", "scenarios"):
            if not rec.wait_while_stopped():
                raise RuntimeError("stopped for more than 30 minutes")
            run_scenarios(rec)
        rec.finish()
    except Exception as e:
        rec.finish(error=f"{type(e).__name__}: {e}")
        print(f"  ERROR {e}")
    print(f"{rec.state['status'].upper()}  {rec.state['counts']}")
    return 0 if rec.state["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

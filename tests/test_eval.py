"""The evaluation itself is tested: grading must be right before its numbers mean anything. No network."""
import pytest

from src.agent.core import Agent
from src.eval import scenarios as sc
from src.eval.claims import find_unbacked_sources, verify_explanations
from src.eval.runner import judge, run_all, run_scenario, self_test
from src.eval.sims import SimLLM


@pytest.fixture(scope="module")
def report():
    return run_all(SimLLM())


class TestScenarioOutcomes:
    def test_every_scenario_ends_as_expected(self, report):
        # Held-out cases are reported, not asserted: they exist to show what isn't solved.
        unexpected = [(r["id"], r["label"], r["findings"]) for r in report["scenarios"]
                      if not r["as_expected"] and r["kind"] != "heldout"]
        assert unexpected == []

    def test_heldout_cases_are_graded_like_success(self, report):
        rows = [r for r in report["scenarios"] if r["kind"] == "heldout"]
        assert len(rows) == 7
        assert all(r["label"] in ("PASS", "FAIL") and r["as_expected"] == (r["label"] == "PASS") for r in rows)

    def test_known_gaps_fail_for_the_stated_reason(self, report):
        for r in report["scenarios"]:
            if r["kind"] == "failure":
                assert set(r["expected_codes"]) <= {f["code"] for f in r["findings"]}, r["id"]

    def test_success_scenarios_have_no_findings(self, report):
        assert all(not r["findings"] for r in report["scenarios"] if r["kind"] == "success")

    def test_invented_link_is_caught(self, report):
        row = next(r for r in report["scenarios"] if r["id"] == "invented_source")
        assert row["label"] == "CAUGHT" and row["findings"][0]["code"] == sc.UNVERIFIED_SOURCE


class TestJudge:
    def test_success_with_a_finding_fails(self):
        assert judge(sc.BY_ID["aligned_user"], [{"code": sc.UNDER_ROUTED}]) == ("FAIL", False)

    def test_known_gap_that_stops_failing_is_flagged(self):
        label, ok = judge(sc.BY_ID["negation"], [])
        assert label == "FIXED?" and not ok          # surprising good news still needs a human look

    def test_known_gap_failing_a_different_way_is_flagged(self):
        label, ok = judge(sc.BY_ID["negation"], [{"code": sc.BACKEND_FAILED}])
        assert label == "DIFFERENT FAILURE" and not ok

    def test_false_claim_not_caught_is_a_miss(self):
        assert judge(sc.BY_ID["invented_source"], []) == ("MISSED", False)


class TestGradingCatchesRealProblems:
    def test_wrong_ideal_tier_is_reported_with_direction(self):
        scenario = sc.Scenario("t", "t", "success", "t", turns=[sc.Turn("hello there friend", ideal="ultra")])
        findings = run_scenario(scenario, SimLLM())["findings"]
        assert [f["code"] for f in findings] == [sc.UNDER_ROUTED]

    def test_over_routing_is_reported(self):
        scenario = sc.Scenario("t", "t", "success", "t", stated={"length": "concise"},
                               turns=[sc.Turn("Explain X in more detail"), sc.Turn("Elaborate on X", ideal="nano")])
        assert [f["code"] for f in run_scenario(scenario, SimLLM())["findings"]] == [sc.OVER_ROUTED]

    def test_missed_and_false_signal_are_told_apart(self):
        missed = sc.Scenario("t", "t", "success", "t", turns=[sc.Turn("hi", observed_after={"length": "detailed"})])
        false = sc.Scenario("t", "t", "success", "t", turns=[
            sc.Turn("Elaborate on X"), sc.Turn("Go deeper on X", observed_after={"length": "balanced"})])
        assert run_scenario(missed, SimLLM())["findings"][0]["code"] == sc.MISSED_SIGNAL
        assert run_scenario(false, SimLLM())["findings"][0]["code"] == sc.FALSE_SIGNAL

    def test_format_is_only_judged_on_a_real_model(self):
        class IgnoresFormat:
            simulated = False

            def chat(self, tier, messages, temperature=0.7):
                result = SimLLM().chat(tier, messages)
                result.text = "No summary line here."
                return result

        scenario = sc.BY_ID["classic_contradiction"]
        assert not [f for f in run_scenario(scenario, SimLLM())["findings"] if f["code"] == sc.IGNORED_FORMAT]
        live = [f for f in run_scenario(scenario, IgnoresFormat())["findings"] if f["code"] == sc.IGNORED_FORMAT]
        assert len(live) == 2


class TestTldrCheck:
    """Found on the first live run: the grader failed a correct answer because it was wrapped in bold."""

    @pytest.mark.parametrize("text", ["TL;DR: x", "\n\n**TL;DR: x**", "## TL;DR\nx", "> tl;dr: x", "_TL;DR:_ x"])
    def test_accepts_decorated_tldr(self, text):
        from src.eval.runner import opens_with_tldr
        assert opens_with_tldr(text)

    @pytest.mark.parametrize("text", ["Here is the answer. TL;DR: x", "Attention is...", ""])
    def test_rejects_when_not_first(self, text):
        from src.eval.runner import opens_with_tldr
        assert not opens_with_tldr(text)


class TestFalseClaimChecker:
    def test_injected_false_claims_are_all_caught(self):
        result = self_test()
        assert result["caught"] == result["total"] == 7, result["cases"]

    def test_true_explanations_pass_in_every_scenario(self, report):
        for r in report["scenarios"]:
            assert not [f for f in r["findings"] if f["code"] == sc.FALSE_CLAIM], r["id"]

    def test_checker_reads_the_database_not_the_agent(self, tmp_path):
        db = str(tmp_path / "c.db")
        agent = Agent(db, SimLLM())
        agent.profiles.update_stated("u", "length", "concise")
        agent.respond("u", "In more detail please")
        result = agent.respond("u", "Elaborate on that")
        assert verify_explanations(db, "u", result["routing_reason"], result["suggestion"]) == []
        agent.profiles.update_stated("u", "length", "detailed")      # the record changes afterwards...
        assert verify_explanations(db, "u", result["routing_reason"], result["suggestion"])   # ...the old claim now fails

    def test_illustrations_and_fragments_are_not_citations(self):
        """Found live: an HTTPS explanation used example.com and a bare `https://` in code."""
        text = "Visit `https://example.com`. The scheme is `https://`, e.g. www.example.com or api.test"
        assert find_unbacked_sources(text) == []
        assert find_unbacked_sources("Source: https://arxiv.org/abs/1706.03762") == ["https://arxiv.org/abs/1706.03762"]

    def test_sources_supplied_by_retrieval_are_allowed(self):
        text = "See https://nature.com/a and https://nature.com/b."
        assert find_unbacked_sources(text, allowed=["https://nature.com/a"]) == ["https://nature.com/b."]
        assert find_unbacked_sources("no links here") == []


class TestReaderLabels:
    def test_known_gaps_under_the_nano_reader_are_an_experiment(self):
        assert judge(sc.BY_ID["negation"], [], "nano") == ("FIXED", True)
        assert judge(sc.BY_ID["negation"], [{"code": sc.FALSE_SIGNAL}], "nano") == ("STILL FAILS", True)


class TestCompare:
    """The blind Nano-vs-Ultra comparison, with a fake judge. No network."""

    class Fake:
        simulated = True

        def __init__(self, judge_pick):
            self.judge_pick = judge_pick      # function(prompt) -> "A" | "B" | "tie" | raw text

        def chat(self, tier, messages, temperature=0.7, **kw):
            from src.agent.llm_client import LLMResult
            if messages[0]["content"].startswith("You compare two answers"):
                pick = self.judge_pick(messages[1]["content"])
                text = pick if pick.startswith("{") or pick == "garbage" else f'{{"winner": "{pick}", "why": "x"}}'
                return LLMResult(text=text, tier=tier, model="judge", latency_ms=1)
            body = "TL;DR: short.\n\nLong ULTRA answer" if tier == "ultra" else "nano answer"
            return LLMResult(text=body, tier=tier, model=tier, latency_ms=1, input_tokens=100,
                             output_tokens=300 if tier == "ultra" else 100)

    def test_consistent_preference_wins_the_case(self):
        from src.eval.compare import run
        prefer_ultra = lambda p: "A" if p.index("ULTRA") < p.index("Answer B") else "B"
        rep = run(self.Fake(prefer_ultra))
        assert rep["wins"] == {"nano": 0, "ultra": 6} and "paid off" in rep["verdict"]
        assert rep["tldr_on_length_cases"]["ultra"] == "2/2" and rep["cost_ratio"] > 1

    def test_position_bias_counts_as_a_tie(self):
        from src.eval.compare import run
        rep = run(self.Fake(lambda p: "A"))                  # always picks whatever is shown first
        assert rep["wins"] == {"nano": 0, "ultra": 0} and rep["ties"] == 6
        assert "not better" in rep["verdict"]               # an honest verdict, not a default win

    def test_unreadable_verdicts_are_counted_not_guessed(self):
        from src.eval.compare import run
        rep = run(self.Fake(lambda p: "garbage"))
        assert rep["invalid"] == 6 and rep["wins"] == {"nano": 0, "ultra": 0}

    def test_judge_cannot_be_a_contestant(self):
        from src.eval.compare import run
        with pytest.raises(ValueError):
            run(self.Fake(lambda p: "A"), judge_tier="ultra")


class TestStrategies:
    def test_router_is_cheaper_than_always_ultra_and_fits_better_than_always_nano(self, report):
        s = report["strategies"]
        assert s["router"]["cost_usd"] < s["always-ultra"]["cost_usd"]
        assert s["always-nano"]["under"] > s["router"]["under"]
        assert s["always-ultra"]["over"] > s["router"]["over"]


class TestSavedReports:
    def test_cards_and_detail_roundtrip_and_source(self, tmp_path):
        from src.eval import reports
        from src.eval.runner import run_all
        from src.eval.sims import SimLLM
        rep = run_all(SimLLM(), compare=False, only={"aligned_user"})
        reports.save(rep, "run-a", {"nano": "ollama"}, out=tmp_path)
        reports.save(rep, "run-b", {"nano": "ollama", "ultra": "nebius"}, out=tmp_path)
        cards = {c["id"]: c for c in reports.cards(tmp_path)}
        assert cards["run-a"]["source"] == "local" and cards["run-b"]["source"] == "api"
        assert "report" not in cards["run-a"] and reports.get("run-a", tmp_path)["report"]["scenarios"]

    def test_bad_ids_are_refused(self, tmp_path):
        from src.eval import reports
        assert reports.get("../secrets", tmp_path) is None and reports.get("Nope", tmp_path) is None


class TestDemos:
    def test_every_demo_runs_and_is_graded(self, tmp_path):
        from src.agent.core import Agent
        from src.eval import demos
        from src.eval.sims import SimLLM
        agent = Agent(str(tmp_path / "d.db"), SimLLM(), memory_dir=str(tmp_path / "m"))
        out = {d["id"]: demos.run(d["id"], agent) for d in demos.DEMOS}
        assert out["aligned"]["label"] == "PASS"
        assert out["false_claim"]["label"] == "CAUGHT"            # the planted link is flagged
        assert all(r["label"] in ("PASS", "FAIL", "FIXED", "KNOWN GAP", "CAUGHT", "MISSED") for r in out.values())

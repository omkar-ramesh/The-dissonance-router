"""Dev dashboard: run recorder, aggregation and history store."""
import json

import pytest

import os
import threading
import time

from src.devdash.store import RunRecorder, active_run, list_runs, load_run, read_log, set_stop

STATS = {
    "tier": "nano", "model_id": "m", "model_name": "m", "input_tokens": 100, "output_tokens": 50,
    "total_tokens": 150, "tokens_per_sec": 25.0, "ttft_ms": 1000, "latency_ms": 2000,
    "agent_rounds": 1, "context_window": 1000, "context_pct": 15.0,
    "est_cost_usd": 0.001, "est_cost_if_ultra_usd": 0.004,
}


@pytest.fixture
def rec(tmp_path):
    return RunRecorder("scenarios", {"nano": "m (ollama)"}, history_dir=tmp_path)


def test_progress_and_counts(rec):
    rec.add_steps([("a", "A", "live"), ("b", "B", "live"), ("c", "C", "live"), ("d", "D", "live")])
    assert rec.state["counts"]["progress"] == 0.0
    rec.step_start("a")
    assert rec.state["counts"]["running"] == 1
    rec.step_end("a", True)
    rec.step_end("b", False, "boom")
    counts = rec.state["counts"]
    assert (counts["passed"], counts["failed"], counts["pending"], counts["progress"]) == (1, 1, 2, 50.0)


def test_status_after_finish(rec):
    rec.add_steps([("a", "A", "live")])
    rec.step_end("a", True)
    rec.finish()
    assert rec.state["status"] == "passed" and rec.state["ended_at"]


def test_failed_step_makes_run_failed(rec):
    rec.add_steps([("a", "A", "live"), ("b", "B", "live")])
    rec.step_end("a", True)
    rec.step_end("b", False)
    rec.finish()
    assert rec.state["status"] == "failed"


def test_error_status(rec):
    rec.finish(error="Ollama is down")
    assert (rec.state["status"], rec.state["error"]) == ("error", "Ollama is down")


def test_totals_aggregate_model_calls_only(rec):
    rec.add_steps([("a", "A", "live"), ("b", "B", "live"), ("c", "C", "pytest")])
    rec.step_end("a", True, stats=STATS)
    rec.step_end("b", True, stats=dict(STATS, output_tokens=150, total_tokens=250, latency_ms=2000, ttft_ms=3000, context_pct=40.0))
    rec.step_end("c", True)  # no model call
    totals = rec.state["totals"]
    assert totals["calls"] == 2
    assert (totals["input_tokens"], totals["output_tokens"], totals["total_tokens"]) == (200, 200, 400)
    assert totals["tokens_per_sec"] == 50.0  # 200 tokens over 4s of wall time
    assert totals["avg_ttft_ms"] == 2000
    assert totals["max_context_pct"] == 40.0
    assert totals["agent_rounds"] == 2


def test_totals_cost_and_routing_savings(rec):
    rec.add_steps([("a", "A", "live"), ("b", "B", "live")])
    rec.step_end("a", True, stats=STATS)                                    # $0.001 vs $0.004 if Ultra
    rec.step_end("b", True, stats=dict(STATS, est_cost_usd=0.003, est_cost_if_ultra_usd=0.004))
    totals = rec.state["totals"]
    assert totals["est_cost_usd"] == pytest.approx(0.004)
    assert totals["est_cost_if_ultra_usd"] == pytest.approx(0.008)
    assert totals["savings_pct"] == 50.0


def test_totals_without_prices_have_no_cost(rec):
    rec.add_steps([("a", "A", "live")])
    rec.step_end("a", True, stats=dict(STATS, est_cost_usd=None, est_cost_if_ultra_usd=None))
    totals = rec.state["totals"]
    assert (totals["est_cost_usd"], totals["savings_pct"]) == (None, None)


def test_empty_run_has_safe_totals(rec):
    assert rec.state["totals"]["calls"] == 0
    assert rec.state["totals"]["est_cost_usd"] is None
    assert rec.state["totals"]["avg_ttft_ms"] is None
    assert rec.state["counts"]["progress"] == 0.0


def test_history_persists_each_change(rec, tmp_path):
    rec.add_steps([("a", "A", "live")])
    on_disk = json.loads((tmp_path / f"{rec.id}.json").read_text(encoding="utf-8"))
    assert on_disk["steps"][0]["status"] == "pending"
    rec.step_end("a", True, "ok")
    assert load_run(rec.id, tmp_path)["steps"][0]["detail"] == "ok"
    assert load_run(rec.id, tmp_path)["rev"] > on_disk["rev"]


def test_list_runs_newest_first(tmp_path):
    for name in ("20260101-000000-pytest", "20260102-000000-all"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"id": name, "suite": "x", "status": "passed"}), encoding="utf-8")
    (tmp_path / "20260103-000000-bad.json").write_text("{not json", encoding="utf-8")
    assert [r["id"] for r in list_runs(history_dir=tmp_path)] == ["20260102-000000-all", "20260101-000000-pytest"]


@pytest.mark.parametrize("run_id", ["../secret", "..\\secret", "a/b", ""])
def test_load_run_rejects_unsafe_ids(tmp_path, run_id):
    assert load_run(run_id, tmp_path) is None


def test_load_missing_run(tmp_path):
    assert load_run("nope", tmp_path) is None


def test_save_retries_when_dashboard_has_file_open(rec, monkeypatch):
    """Windows raises PermissionError on replace while a reader holds the file."""
    import os
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    rec.add_steps([("a", "A", "live")])
    assert calls["n"] == 4
    assert load_run(rec.id, rec.dir)["steps"][0]["id"] == "a"


def test_save_never_raises_if_file_stays_locked(rec, monkeypatch):
    import os

    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr("src.devdash.store.time.sleep", lambda s: None)
    rec.add_steps([("a", "A", "live")])  # must not raise
    rec.step_end("a", True)
    assert rec.state["steps"][0]["status"] == "passed"


def test_wait_returns_immediately_when_not_stopped(rec):
    assert rec.wait_while_stopped() is True
    assert rec.state["status"] == "running"


def test_stop_holds_the_run_until_resumed(rec, tmp_path):
    seen = []

    def user():
        time.sleep(0.5)
        seen.append(load_run(rec.id, tmp_path)["status"])   # what the dashboard would see
        set_stop(rec.id, False, tmp_path)

    assert set_stop(rec.id, True, tmp_path)
    thread = threading.Thread(target=user)
    thread.start()
    assert rec.wait_while_stopped(timeout=10) is True
    thread.join()
    assert seen == ["stopped"]
    assert rec.state["status"] == "running"
    messages = [l["msg"] for l in read_log(rec.id, 0, tmp_path)["lines"]]
    assert any("stopped" in m for m in messages) and any("resumed" in m for m in messages)


def test_stop_times_out(rec, tmp_path):
    set_stop(rec.id, True, tmp_path)
    assert rec.wait_while_stopped(timeout=0.5) is False


def test_active_run_and_set_stop(rec, tmp_path):
    assert active_run(tmp_path) == rec.id
    rec.finish()
    assert active_run(tmp_path) is None
    assert set_stop("nope", True, tmp_path) is False
    assert set_stop("../x", True, tmp_path) is False


def test_dead_runner_is_not_shown_as_running_forever(rec, tmp_path):
    assert load_run(rec.id, tmp_path)["status"] == "running"
    old = time.time() - 700
    os.utime(rec.path, (old, old))
    run = load_run(rec.id, tmp_path)
    assert run["status"] == "error" and "stopped responding" in run["error"]
    assert active_run(tmp_path) is None


def test_step_description_is_optional(rec):
    rec.add_steps([("a", "A", "live", "checks X"), ("b", "B", "pytest")])
    assert [s["what"] for s in rec.state["steps"]] == ["checks X", ""]


def test_log_records_the_story_of_a_live_step(rec, tmp_path):
    rec.add_steps([("a", "Escalates", "live")])
    rec.step_start("a")
    rec.step_end("a", True, "routed to ultra", stats=STATS)
    rec.finish()
    log = read_log(rec.id, history_dir=tmp_path)
    text = [(line["lvl"], line["msg"]) for line in log["lines"]]
    assert text[0][1].startswith("▶ Escalates") and "model swap" in text[0][1]
    assert ("model", "  nano · m (m) · 100→50 tok · 25.0 tok/s · TTFT 1.0s · ctx 15.0%") in text
    assert ("ok", "✓ passed: routed to ultra") in text
    assert text[-1] == ("ok", "run passed: 1 passed, 0 failed")


def test_pytest_steps_log_compactly(rec, tmp_path):
    rec.add_steps([("t1", "a › t1", "pytest"), ("t2", "a › t2", "pytest")])
    rec.step_start("t1"); rec.step_end("t1", True)
    rec.step_start("t2"); rec.step_end("t2", False)
    rec.annotate("t2", "assert 1 == 2")
    lines = [(l["lvl"], l["msg"]) for l in read_log(rec.id, history_dir=tmp_path)["lines"]]
    assert lines == [("dim", "✓ a › t1"), ("fail", "✗ a › t2"), ("fail", "  a › t2: assert 1 == 2")]


def test_read_log_tails_from_offset(rec, tmp_path):
    for i in range(3):
        rec.log(f"line {i}")
    first = read_log(rec.id, 0, tmp_path)
    assert [l["msg"] for l in first["lines"]] == ["line 0", "line 1", "line 2"] and first["next"] == 3
    rec.log("line 3")
    second = read_log(rec.id, first["next"], tmp_path)
    assert [l["msg"] for l in second["lines"]] == ["line 3"] and second["next"] == 4
    assert read_log(rec.id, 99, tmp_path) == {"lines": [], "next": 99}


def test_read_log_ignores_half_written_line(rec, tmp_path):
    rec.log("complete")
    with rec.log_path.open("a", encoding="utf-8") as f:
        f.write('{"t": 1.0, "lvl": "info", "ms')
    assert [l["msg"] for l in read_log(rec.id, 0, tmp_path)["lines"]] == ["complete"]


@pytest.mark.parametrize("run_id", ["../x", "nope", ""])
def test_read_log_rejects_bad_ids(tmp_path, run_id):
    assert read_log(run_id, 0, tmp_path) is None

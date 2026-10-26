"""Run history for the dev dashboard: one small JSON file per run in data/history/.

The runner rewrites its file after every event (atomically); the dashboard just
reads files, so there is no shared process state and nothing to keep in memory.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from src import config

HISTORY_DIR = Path(config.ROOT_DIR) / "data" / "history"
RUN_ID_RE = re.compile(r"^[\w-]+$")
ACTIVE = ("running", "stopped")
STALE_SECS = 600  # longer than the slowest model call (180 s timeout) plus a cold model load


class RunRecorder:
    """Accumulates steps and their stats for one run and persists every change."""

    def __init__(
        self, suite: str, models: Optional[dict] = None, history_dir: Path = HISTORY_DIR,
        prices: Optional[dict] = None,
    ):
        self.dir = Path(history_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self.id = time.strftime("%Y%m%d-%H%M%S", time.localtime(started)) + f"-{suite}"
        self.path = self.dir / f"{self.id}.json"
        self.log_path = self.dir / f"{self.id}.log"
        self.log_path.write_text("", encoding="utf-8")
        self.ctl_path = self.dir / f"{self.id}.ctl"   # exists while a stop is requested
        self.state = {
            "id": self.id,
            "suite": suite,
            "status": "running",
            "started_at": started,
            "ended_at": None,
            "rev": 0,
            "error": None,
            "models": models or {},
            "prices": prices or {},
            "steps": [],
        }
        self._steps: dict[str, dict] = {}
        self.save()

    def log(self, msg: str, lvl: str = "info") -> None:
        """Append one line to the run's log (levels: info, ok, fail, dim, model).

        The log is a separate append-only file so the dashboard can tail it cheaply.
        """
        line = json.dumps({"t": round(time.time() - self.state["started_at"], 1), "lvl": lvl, "msg": msg})
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def add_steps(self, steps: list[tuple]) -> None:
        """Register pending steps as (id, display name, group[, what it checks])."""
        for step_id, name, group, *rest in steps:
            if step_id in self._steps:
                continue
            step = {"id": step_id, "name": name, "group": group, "what": rest[0] if rest else "",
                    "status": "pending", "detail": "", "duration_ms": None, "stats": None, "started_at": None}
            self._steps[step_id] = step
            self.state["steps"].append(step)
        self.save()

    def step_start(self, step_id: str) -> None:
        step = self._steps[step_id]
        step["status"] = "running"
        step["started_at"] = time.time()
        if step["group"] != "pytest":
            hint = "  (a model swap can take 10-30 s before the first token)" if step["group"] == "live" else ""
            self.log(f"▶ {step['name']}{hint}")
        self.save()

    def step_end(self, step_id: str, passed: bool, detail: str = "", stats: Optional[dict] = None) -> None:
        step = self._steps[step_id]
        step["status"] = "passed" if passed else "failed"
        step["detail"] = detail
        step["stats"] = stats
        if step["started_at"]:
            step["duration_ms"] = int((time.time() - step["started_at"]) * 1000)
        mark = "✓" if passed else "✗"
        if step["group"] == "pytest":
            self.log(f"{mark} {step['name']}", "dim" if passed else "fail")
        else:
            if stats:
                self.log(_model_line(stats), "model")
            self.log(f"{mark} {'passed' if passed else 'FAILED'}: {detail}", "ok" if passed else "fail")
        self.save()

    def annotate(self, step_id: str, detail: str) -> None:
        """Attach extra text (e.g. a pytest failure message) without changing status."""
        if step_id in self._steps:
            self._steps[step_id]["detail"] = detail
            self.log(f"  {self._steps[step_id]['name']}: {detail}", "fail")

    def stop_requested(self) -> bool:
        return self.ctl_path.exists()

    def wait_while_stopped(self, timeout: float = 1800.0) -> bool:
        """Block between steps while the dashboard's Stop is on. False if it timed out."""
        if not self.stop_requested():
            return True
        self.state["status"] = "stopped"
        self.log("⏸ stopped. Press Resume to continue", "info")
        self.save()
        deadline = time.time() + timeout
        last_beat = time.time()
        while self.stop_requested():
            if time.time() > deadline:
                return False
            if time.time() - last_beat > 30:    # heartbeat so a waiting run isn't mistaken for a dead one
                self.save()
                last_beat = time.time()
            time.sleep(0.3)
        self.state["status"] = "running"
        self.log("▶ resumed", "info")
        self.save()
        return True

    def finish(self, error: Optional[str] = None) -> None:
        counts = self.state["counts"]
        self.state["error"] = error
        self.state["status"] = "error" if error else ("failed" if counts["failed"] else "passed")
        self.state["ended_at"] = time.time()
        if error:
            self.log(f"run aborted: {error}", "fail")
        else:
            self.log(f"run {self.state['status']}: {counts['passed']} passed, {counts['failed']} failed",
                     "ok" if self.state["status"] == "passed" else "fail")
        self.save()

    def save(self) -> None:
        self.state["rev"] += 1
        self.state["counts"] = _counts(self.state["steps"])
        self.state["totals"] = _totals(self.state["steps"])
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        # On Windows, replacing a file fails while the dashboard has it open. Retry briefly;
        # if it still fails, skip this snapshot (the next event rewrites it) rather than
        # killing the run over a progress update.
        for _ in range(40):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.025)


def _model_line(stats: dict) -> str:
    ttft = "—" if stats.get("ttft_ms") is None else f"{stats['ttft_ms'] / 1000:.1f}s"
    ctx = "" if stats.get("context_pct") is None else f" · ctx {stats['context_pct']}%"
    return (f"  {stats['tier']} · {stats['model_name']} ({stats['model_id']}) · "
            f"{stats['input_tokens']}→{stats['output_tokens']} tok · {stats['tokens_per_sec']} tok/s · TTFT {ttft}{ctx}")


def _counts(steps: list[dict]) -> dict:
    by_status = {s: 0 for s in ("passed", "failed", "running", "pending")}
    for step in steps:
        by_status[step["status"]] += 1
    total = len(steps)
    done = by_status["passed"] + by_status["failed"]
    return {**by_status, "total": total, "done": done, "progress": round(100 * done / total, 1) if total else 0.0}


def _totals(steps: list[dict]) -> dict:
    """Aggregate the LLM stats of every step that made a model call."""
    calls = [s["stats"] for s in steps if s.get("stats")]
    out_tokens = sum(c["output_tokens"] for c in calls)
    wall_ms = sum(c["latency_ms"] for c in calls)
    ttfts = [c["ttft_ms"] for c in calls if c.get("ttft_ms") is not None]
    pcts = [c["context_pct"] for c in calls if c.get("context_pct") is not None]
    priced = [c for c in calls if c.get("est_cost_usd") is not None]
    cost = sum(c["est_cost_usd"] for c in priced)
    cost_if_ultra = sum(c["est_cost_if_ultra_usd"] for c in priced)
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": out_tokens,
        "total_tokens": sum(c["total_tokens"] for c in calls),
        "tokens_per_sec": round(out_tokens / (wall_ms / 1000), 1) if wall_ms else 0.0,
        "avg_ttft_ms": int(sum(ttfts) / len(ttfts)) if ttfts else None,
        "max_context_pct": max(pcts) if pcts else None,
        "agent_rounds": sum(c["agent_rounds"] for c in calls),
        "est_cost_usd": cost if priced else None,
        "est_cost_if_ultra_usd": cost_if_ultra if priced else None,
        "savings_pct": round(100 * (1 - cost / cost_if_ultra), 1) if priced and cost_if_ultra else None,
    }


def list_runs(limit: int = 50, history_dir: Path = HISTORY_DIR) -> list[dict]:
    """Summaries of the newest runs, newest first."""
    runs = []
    for path in sorted(Path(history_dir).glob("*.json"), reverse=True)[:limit]:
        run = _read(path)
        if run:
            runs.append({k: run.get(k) for k in ("id", "suite", "status", "started_at", "ended_at", "counts", "totals")})
    return runs


def load_run(run_id: str, history_dir: Path = HISTORY_DIR) -> Optional[dict]:
    """Full state of one run, or None if it doesn't exist or the id is unsafe."""
    if not RUN_ID_RE.match(run_id):
        return None
    return _read(Path(history_dir) / f"{run_id}.json")


def read_log(run_id: str, start: int = 0, history_dir: Path = HISTORY_DIR) -> Optional[dict]:
    """Log lines from index `start` on, plus the index to ask for next. None if no such run."""
    if not RUN_ID_RE.match(run_id):
        return None
    try:
        raw = (Path(history_dir) / f"{run_id}.log").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    lines = []
    for line in raw[max(start, 0):]:
        try:
            lines.append(json.loads(line))
        except ValueError:  # a line caught mid-append; picked up on the next poll
            break
    return {"lines": lines, "next": max(start, 0) + len(lines)}


def active_run(history_dir: Path = HISTORY_DIR) -> Optional[str]:
    """Id of the run that is currently running or stopped, if any."""
    for run in list_runs(limit=5, history_dir=history_dir):
        if run["status"] in ACTIVE:
            return run["id"]
    return None


def set_stop(run_id: str, stop: bool, history_dir: Path = HISTORY_DIR) -> bool:
    """Ask a run to stop before its next step (or resume it). False for an unknown run id."""
    if not RUN_ID_RE.match(run_id) or not (Path(history_dir) / f"{run_id}.json").exists():
        return False
    ctl = Path(history_dir) / f"{run_id}.ctl"
    if stop:
        ctl.write_text("stop", encoding="utf-8")
    else:
        ctl.unlink(missing_ok=True)
    return True


def _read(path: Path) -> Optional[dict]:
    for attempt in range(20):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
            # A runner that died mid-run leaves "running" on disk forever; show that honestly.
            if run.get("status") in ACTIVE and time.time() - path.stat().st_mtime > STALE_SECS:
                run = {**run, "status": "error", "error": "The runner stopped responding (no update for 10 minutes)."}
            return run
        except FileNotFoundError:
            return None
        except (PermissionError, ValueError):  # caught mid-replace on Windows: wait it out
            time.sleep(0.01)
        except OSError:
            return None
    return None

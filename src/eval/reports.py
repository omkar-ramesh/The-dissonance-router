"""Saved reports: one gzipped JSON file per run in docs/reports/, so the app and the repo show the same evidence.

`python -m src.eval.reports` (re)builds the files for the reports already in docs/.
Where a run happened is read from the tier providers: any tier on Nebius makes it a Token Factory run;
everything else (Ollama, or the simulator) is local, i.e. what we did while building.
"""
import gzip
import os
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import config

REPORTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "reports"
ARCHIVE_DIR = REPORTS_DIR / "archive"        # older runs: kept as evidence, hidden from the default lists
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_+-]{0,80}$")


def where(providers: dict | None) -> str:
    return "api" if any(p == "nebius" for p in (providers or {}).values()) else "local"


def _ms(v) -> str:
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def _scenario_metrics(report: dict) -> list:
    r = report["strategies"].get("router") or next(iter(report["strategies"].values()), {})
    st = report["self_test"]
    out = []
    if r.get("graded"):
        out += [["Routing accuracy", f"{r['matched']}/{r['graded']}"], ["Under / over-routed", f"{r['under']} / {r['over']}"]]
    if r:
        out += [["Tokens in / out", f"{r['input_tokens']:,} / {r['output_tokens']:,}"], ["Latency / answer", _ms(r["latency_ms"])],
                ["Cost (router)", f"${r['cost_usd']:.5f}"]]
    return out + [["False claims caught", f"{st['caught']}/{st['total']}"]]


def _scenario_card(report: dict, providers) -> dict:
    sc = report["scenarios"]
    ok = sum(1 for s in sc if s["as_expected"])
    st = report["self_test"]
    cost = report["strategies"].get("router", {}).get("cost_usd", 0)
    reader = report.get("reader", "keywords")
    sim = report["simulated"]
    return {
        "kind": "scenarios",
        "title": (f"{sc[0]['id'].replace('_', ' ')} · " if len(sc) == 1 else f"{len(sc)} scenarios · ") + f"{'Nemotron Nano' if reader == 'nano' else 'keyword'} reader",
        "description": ("Routing and grading are real; model answers and tokens are simulated." if sim else
                        "Real model answers on every scenario, graded by the same checks as the simulated run."),
        "headline": f"{ok}/{len(sc)} as expected",
        "chips": [f"{st['caught']}/{st['total']} false claims caught", "simulated" if sim else "real answers"],
        "flag": ("fail" if any(not x["as_expected"] or x["label"] in ("FAIL", "KNOWN GAP") for x in sc)
                 else "fake" if len(sc) == 1 and sc[0]["kind"] == "false_claim" else ""),
        "cost_usd": cost, "simulated": sim, "reader": reader, "metrics": _scenario_metrics(report),
        "models": report.get("models", {}), "source": where(providers),
    }


def _compare_card(report: dict, providers) -> dict:
    a, b = report["contestants"]
    return {
        "kind": "compare",
        "title": f"Is Ultra worth it? · {a} vs {b}" + (f" · {report['results'][0]['id']}" if report.get("cases") == 1 and report.get("results") else ""),
        "description": "Blind answer comparison: a judge model sees the answers without names, in both orders.",
        "headline": f"{report['wins'][a]}–{report['wins'][b]} · {report['ties']} tied",
        "chips": [f"{report['cases']} cases", f"cost ×{report['cost_ratio']}"],
        "cost_usd": sum(v["cost_usd"] for v in report["avg"].values()) * report["cases"],
        "simulated": report["simulated"], "reader": "", "models": report.get("models", {}),
        "source": where(providers),
        "metrics": [["Blind wins", f"{a} {report['wins'][a]} · {b} {report['wins'][b]} · tie {report['ties']}"]]
        + [[f"{k} latency / tokens out", f"{_ms(v['latency_ms'])} / {v['output_tokens']:.0f}"] for k, v in report["avg"].items()]
        + [["Cost ratio", f"x{report['cost_ratio']}"]],
    }


def _demos_card(report: dict, providers) -> dict:
    from src.eval.demos import BY_ID
    res = [{"name": BY_ID.get(r["id"], {}).get("name", r["id"])} | r for r in report["demos"]]
    good = sum(1 for r in res if r["label"] in ("PASS", "CAUGHT", "KNOWN GAP", "FIXED"))
    return {
        "kind": "demos",
        "title": f"{res[0]['name']} · scripted persona run" if len(res) == 1 else f"{len(res)} demo profiles · scripted",
        "description": (f"Fixed script, real model answers: {res[0]['title']}." if len(res) == 1 else
                        "Fixed scripts, real model answers: 2 success, 2 known failures, 1 planted false claim."),
        "headline": " · ".join(f"{r['label']}" for r in res),
        "chips": [f"{good}/{len(res)} behaved as documented"], "cost_usd": 0, "simulated": False, "reader": "",
        "personas": [r.get("name", r["id"]) for r in res],
        "flag": "fail" if any(r["label"] == "KNOWN GAP" for r in res) else "fake" if len(res) == 1 and res[0]["label"] == "CAUGHT" else "",
        "metrics": [[r.get("name", r["id"]), f"{r['label']} · {' > '.join(t['tier'] for t in r['turns'])}"] for r in res]
        + [["Latency / answer", _ms(sum(t["latency_ms"] for r in res for t in r["turns"]) / max(1, sum(len(r["turns"]) for r in res)))],
           ["Tokens (total)", f"{sum(t['tokens'].get('total_tokens', 0) for r in res for t in r['turns']):,}"]],
        "models": report.get("models", {}), "source": where(providers),
    }


def build(report: dict, name: str, providers: dict | None = None, created: str | None = None) -> dict:
    card = (_compare_card if "contestants" in report else _demos_card if "demos" in report else _scenario_card)(report, providers)
    src = card["source"]
    run_n = os.environ.get("RUN_N")                    # repeat runs of the same test get "· run N" in the title
    if run_n:
        card["title"] += f" · run {run_n}"
    card["run_on"] = ("Simulator · no model calls" if card["simulated"] else
                      f"Nebius Token Factory · {config.NEBIUS_BASE_URL or 'api.tokenfactory.nebius.com'}" if src == "api"
                      else f"Local · {config.HARDWARE} · Ollama")
    return {"id": name, "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **card, "report": report}


def save(report: dict, name: str, providers: dict | None = None, out: Path = REPORTS_DIR, created=None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RUN_N"):
        name += f"-r{os.environ['RUN_N']}"
    path = out / f"{name}.json.gz"
    path.write_bytes(gzip.compress(json.dumps(build(report, name, providers or config.TIER_PROVIDERS, created)).encode("utf-8")))
    return path


def _read(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def cards(out: Path = REPORTS_DIR) -> list[dict]:
    items = []
    for folder, archived in ((out, False), (out / "archive", True)):
        for path in sorted([*folder.glob("*.json.gz"), *folder.glob("*.json")]) if folder.is_dir() else []:
            try:
                data = _read(path)
            except (OSError, ValueError):
                continue
            items.append({k: v for k, v in data.items() if k != "report"} | {"archived": archived})
    return sorted(items, key=lambda c: c["created"], reverse=True)


def get(report_id: str, out: Path = REPORTS_DIR) -> dict | None:
    if not ID_RE.match(report_id):
        return None
    for path in (out / f"{report_id}.json.gz", out / f"{report_id}.json",
                 out / "archive" / f"{report_id}.json.gz", out / "archive" / f"{report_id}.json"):
        if path.is_file():
            return _read(path)
    return None


def import_existing() -> list[Path]:
    """Build files for the reports kept in docs/. They were made while building, on local models."""
    docs = REPORTS_DIR.parent
    local = {"nano": "ollama", "super": "ollama", "ultra": "ollama"}
    made = []
    for src, name in (("eval-report.json", "scenarios-keywords-simulated"), ("eval-report-nano.json", "scenarios-nano-reader-local"),
                      ("compare-report.json", "compare-nano-vs-ultra-local")):
        file = docs / src
        if file.is_file():
            stamp = datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            made.append(save(json.loads(file.read_text(encoding="utf-8")), name, local, created=stamp))
    return made


def compress_all(out: Path = REPORTS_DIR) -> list[Path]:
    """Rewrite every report as .json.gz, re-deriving the card so new fields appear."""
    made = []
    for path in sorted([*out.glob("*.json"), *out.glob("*.json.gz")]):
        data = _read(path)
        prov = {"nano": "nebius"} if data.get("source") == "api" else {"nano": "ollama"}
        made.append(save(data["report"], data["id"], prov, out, data.get("created")))
        if path.suffix == ".json":
            path.unlink()
    return made


if __name__ == "__main__":
    if "--compress" in sys.argv:
        for path in compress_all():
            print("compressed", path)
        sys.exit(0)
    for path in import_existing():
        print("wrote", path)
    sys.exit(0)

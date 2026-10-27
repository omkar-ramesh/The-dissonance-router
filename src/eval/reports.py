"""Saved reports: one JSON file per run in docs/reports/, so the app and the repo show the same evidence.

`python -m src.eval.reports` (re)builds the files for the reports already in docs/.
Where a run happened is read from the tier providers: any tier on Nebius makes it a Token Factory run;
everything else (Ollama, or the simulator) is local, i.e. what we did while building.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import config

REPORTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "reports"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


def where(providers: dict | None) -> str:
    return "api" if any(p == "nebius" for p in (providers or {}).values()) else "local"


def _scenario_card(report: dict, providers) -> dict:
    sc = report["scenarios"]
    ok = sum(1 for s in sc if s["as_expected"])
    st = report["self_test"]
    cost = report["strategies"].get("router", {}).get("cost_usd", 0)
    reader = report.get("reader", "keywords")
    sim = report["simulated"]
    return {
        "kind": "scenarios",
        "title": f"{len(sc)} scenarios · {'Nemotron Nano' if reader == 'nano' else 'keyword'} reader",
        "description": ("Routing and grading are real; model answers and tokens are simulated." if sim else
                        "Real model answers on every scenario, graded by the same checks as the simulated run."),
        "headline": f"{ok}/{len(sc)} as expected",
        "chips": [f"{st['caught']}/{st['total']} false claims caught", "simulated" if sim else "real answers"],
        "cost_usd": cost, "simulated": sim, "reader": reader,
        "models": report.get("models", {}), "source": where(providers),
    }


def _compare_card(report: dict, providers) -> dict:
    a, b = report["contestants"]
    return {
        "kind": "compare",
        "title": f"Is Ultra worth it? · {a} vs {b}",
        "description": "Blind answer comparison: a judge model sees the answers without names, in both orders.",
        "headline": f"{report['wins'][a]}–{report['wins'][b]} · {report['ties']} tied",
        "chips": [f"{report['cases']} cases", f"cost ×{report['cost_ratio']}"],
        "cost_usd": sum(v["cost_usd"] for v in report["avg"].values()) * report["cases"],
        "simulated": report["simulated"], "reader": "", "models": report.get("models", {}),
        "source": where(providers),
    }


def build(report: dict, name: str, providers: dict | None = None, created: str | None = None) -> dict:
    card = (_compare_card if "contestants" in report else _scenario_card)(report, providers)
    return {"id": name, "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **card, "report": report}


def save(report: dict, name: str, providers: dict | None = None, out: Path = REPORTS_DIR, created=None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(build(report, name, providers or config.TIER_PROVIDERS, created), indent=1), encoding="utf-8")
    return path


def cards(out: Path = REPORTS_DIR) -> list[dict]:
    items = []
    for path in sorted(out.glob("*.json")) if out.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items.append({k: v for k, v in data.items() if k != "report"})
    return sorted(items, key=lambda c: c["created"], reverse=True)


def get(report_id: str, out: Path = REPORTS_DIR) -> dict | None:
    if not ID_RE.match(report_id):
        return None
    path = out / f"{report_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


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


if __name__ == "__main__":
    for path in import_existing():
        print("wrote", path)
    sys.exit(0)

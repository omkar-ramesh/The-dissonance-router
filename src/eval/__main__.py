"""python -m src.eval [--live] [--reader keywords|nano] [--no-compare] [--only id,id] [--out DIR]
   python -m src.eval compare [--out DIR]      # blind Nano vs Ultra answer comparison (live)"""
import argparse
import sys

from src.eval.runner import run_all, summarize, write_report
from src.eval.sims import SimLLM


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        from src.eval.compare import main as compare_main
        sys.exit(compare_main(sys.argv[2:]))

    parser = argparse.ArgumentParser(prog="python -m src.eval")
    parser.add_argument("--live", action="store_true", help="use the configured models instead of the simulator")
    parser.add_argument("--reader", choices=["keywords", "nano"], default="keywords",
                        help="who reads messages for style requests (nano needs --live)")
    parser.add_argument("--no-compare", action="store_true", help="skip the always-nano / always-ultra pricing runs")
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--save", action="store_true", help="also save a card + detail file in docs/reports/")
    parser.add_argument("--out", default="data/eval", help="where to write the report")
    args = parser.parse_args()
    if args.reader == "nano" and not args.live:
        parser.error("--reader nano reads messages with the real Nano model, so it needs --live")

    real = None
    if args.live:
        from src.agent.llm_client import LLMClient
        real = LLMClient.from_config()
    llm = real or SimLLM()
    report = run_all(llm, compare=not args.no_compare, only=set(args.only.split(",")) if args.only else None,
                     reader=args.reader, reader_llm=real)
    name = "report" if args.reader == "keywords" else f"report-{args.reader}"
    write_report(report, args.out, name)
    if args.save:
        from src import config
        from src.eval import reports
        stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M")
        place = "simulated" if report["simulated"] else ("api" if reports.where(config.TIER_PROVIDERS) == "api" else "local")
        print("saved:", reports.save(report, f"scenarios-{args.reader}-{place}-{stamp}", config.TIER_PROVIDERS))

    print(("simulated run" if report["simulated"] else "live run") + f", reader: {args.reader}")
    for r in report["scenarios"]:
        codes = sorted({f["code"] for f in r["findings"]})
        print(f"  {r['label']:<18} {r['kind']:<12} {r['id']}  {' '.join(codes)}")
    st = report["self_test"]
    print(f"  injected false claims caught: {st['caught']}/{st['total']}")
    for n, t in report["strategies"].items():
        print(f"  {n:<13} ${t['cost_usd']:.4f}  right tier {t['matched']}/{t['graded']}  "
              f"under {t['under']}  over {t['over']}")
    if args.reader == "nano":
        print(f"  reader fell back to keywords {report['reader_fallbacks']} times")
    print(f"report: {args.out}/{name}.md")
    bad = [r for r in report["scenarios"] if not r["as_expected"]]
    sys.exit(1 if bad or st["caught"] != st["total"] else 0)


if __name__ == "__main__":
    main()

# Feedback Notes

Running notes for the Devpost feedback section (Nebius Token Factory, AI Cloud, NVIDIA tools/models). Keep entries short: what happened, why it mattered.

## Token Factory
- (none yet)

## NVIDIA Nemotron
- Nemotron 3 Nano 4B reasons by default: ~144 completion tokens to say "hi". Reasoning is on unless disabled via the chat template flag; for the fast Nano tier we probably want it off. To verify the flag on Token Factory.
- Ollama runs nemotron-3-nano with a 32,768-token window even though the model advertises 256K (4b) / 1M (30b): the dashboard's context % uses the real runtime window from /api/ps.
- Local model swaps are slow: the 24 GB Nemotron 30B and 19 GB DeepSeek cannot both stay in 32 GB VRAM, so TTFT is 13-28 s on a swap. Token Factory should not have this.
- To test on Token Factory: does Nemotron follow system-prompt formatting? (local deepseek-r1 ignored it; we now repeat the format hint in the user turn)
- To test: Nemotron reasoning on/off. Plan is reasoning only on the Ultra path.

## Other Nebius tools (Serverless, Builder Program)
- (none yet)

## Build log
- 2026-10-26: Project folder was empty despite docs saying Phase 0 was done. Starting from scratch.
- 2026-10-26: Phase 0 rebuilt (config, SQLite schema, MIT license, env template). Schema verified: 3 tables, idempotent init, new-user row defaults to "balanced". Tier -> model mapping lives in config so the Ollama -> Token Factory swap is env-only.
- 2026-10-26: Phase 1 done, 40 tests passing. Routing uses the strongest single dimension (avg is reported only): a lone strong contradiction averages to 0.14 and would never escalate. Opposite poles score 0.70 -> 0.80 -> 0.90 -> 0.95 on repeats; "balanced" gaps cap at 0.65 (Super max). Repeat counts come from unresolved rows in `conflicts`; accepting the observed value resolves them.
- 2026-10-27: Phase 2 done locally, 78 tests passing. One OpenAI-compatible client serves Ollama and Token Factory (env swap only). Local tier stand-ins: qwen2.5:3b / qwen2.5:14b / deepseek-r1:32b. Observed profile is now learned from messages (keyword signals, needs 2 votes in last 10 msgs).
- 2026-10-27: Bug found: observed "balanced" (= no evidence yet) was flagged as a conflict, so every user with a stated pref got Super. Fixed: no evidence = no conflict.
- 2026-10-27: Ultra's job defined: reconcile both signals in one answer (e.g. TL;DR line, then depth). deepseek-r1 ignored this in the system prompt; repeating it in the user turn fixed it.
- 2026-10-27: Ollama library has nemotron-3-nano (4b, 30b) that fits the 5090. nemotron-3-super (120b) and nemotron-3-ultra are Cloud-tagged, so Super/Ultra can only be tested on Token Factory. Plan: Nemotron Nano 4b = Nano tier, nemotron-3-nano:30b = Super stand-in, Ultra on the API last (credits).
- 2026-10-27: Added per-tier backends (PROVIDER_ULTRA=nebius) so credits are spent on one tier only.
- 2026-10-27: Dev dashboard built (`python -m src.devdash`, localhost:8765): live progress, pass/fail, per-call tokens, tok/s, TTFT, context %, history in data/history/. Stdlib server + one HTML file, polls at 1 Hz and pauses when the tab is hidden.
- 2026-10-27: Dev dashboard restructured: verdict banner, routing timeline (which tier handled each request + why), live log window (own scroll, tails an append-only .log per run), live scenarios with "what it checks", pytest collapsed. Fixed page jumping (removed scroll-to-running-row).
- 2026-10-27: Dev dashboard now has one "Live tests" window (progress, current step, every test and result, own scroll) plus Stop/Resume. Stop takes effect between steps; a running model call can't be interrupted cleanly. Verified e2e on the real models. A runner that dies mid-run is shown as an error after 10 min instead of "running" forever.
- 2026-10-27: Phase 3 done: FastAPI with /chat, /profiles, /decision-log, /update-style, /accept, /health. Model outage = 503 (message still kept as evidence); profile views are read-only (never logged). 159 tests; demo flow verified over HTTP on real models (Nano 7.4 s, Ultra 24.8 s with TL;DR answer).
- 2026-10-27: AUDIT. Found and fixed: (1) style detector fired on 13/15 ordinary messages ("professional developer" = formal, "redo a git commit" = rejects); patterns now match only requests about the answer, 1/15 left (a quoted phrase). (2) a failed model call changed the profile but didn't save the message; now kept as evidence. (3) dashboard server answered 404 for a run file briefly locked by Windows; now waits it out. (4) CLI printed tracebacks; now one-line errors. Full suite 148/148 on real models.
- 2026-10-27: Cost estimate added to the dashboard. Token Factory list prices per 1M tokens (in/out) from third-party listings, NOT verified on Nebius's own page (behind login): Nano $0.06/$0.24, Super $0.30/$0.90, Ultra $1.00/$3.00. First estimate on the 7-scenario run: $0.0062 routed vs $0.0245 if every call used Ultra (74% saved). Token counts come from local models, so real Nemotron numbers will differ.
- 2026-10-27: Removed the "tool schemas" stat (always 0, the agent uses no tools).
- 2026-10-27: Bug found by smoke test: on Windows, os.replace fails while the dashboard reads the same file, and the runner aborted. Saves now retry and never kill a run.
- 2026-10-27: Live scenarios on local models: 7/7 pass. Nemotron Nano 4B follows our routing prompts; deepseek-r1 Ultra leads with TL;DR after the format hint was repeated in the user turn.
- 2026-10-27: Local latency: cold model load ~35s on first call, warm Ultra (32B) ~17-27s. Need a loading state in the UI.

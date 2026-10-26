# The Dissonance Router

A personal AI agent that notices when **what you say you want** doesn't match **what you actually do**, and spends more model power when that happens. Every routing decision is logged with the reason.

Built for the Nebius x NVIDIA Global AI Hackathon (track: Best Apps and Agents). Runs on NVIDIA Nemotron models.

## The idea

Most assistants store "I prefer short answers" and treat it as fact. People often do the opposite: they keep asking for more detail, ask for rewrites, or ignore the brief reply. The assistant keeps missing, and nobody tells it.

The Dissonance Router keeps two profiles per user:

- **Stated**: what you told it, across five dimensions: tone, length, sourcing, structure, revision.
- **Observed**: what it has learned from how you actually ask for things.

When the two disagree, that disagreement is the signal. The router doesn't pick a model by how hard the question is; it picks by **how uncertain the agent is about you**.

## How it works

1. **Learn.** Each message is scanned for style requests ("elaborate", "make it shorter", "be less formal", "no bullets"). A value only enters the observed profile after it repeats (2 signals in the last 10 messages), so one offhand remark changes nothing.
2. **Score.** Each dimension gets a conflict score from 0 to 1:

   | Situation | Score |
   |---|---|
   | stated = observed, or nothing observed yet | 0.00 |
   | opposite poles (stated *concise*, observed *detailed*) | 0.70, rising to 0.80 / 0.90 / 0.95 as the same contradiction repeats |
   | a preference is emerging that was never stated | 0.40, rising to 0.65 |

3. **Route.** The strongest single contradiction decides the model tier:

   | Score | Tier | Why |
   |---|---|---|
   | >= 0.70 | **Ultra** | The agent is unsure what you want; the answer matters |
   | 0.40 - 0.70 | **Super** | A preference is emerging |
   | < 0.40 | **Nano** | No conflict, so use the cheap fast model |

   The strongest dimension is used rather than the average because one clear contradiction averages to 0.14 across five dimensions and would never escalate.
4. **Reconcile.** On an Ultra escalation the model is told about the contradiction and asked to serve both sides: for example a one-line TL;DR for the "concise" you said, then the depth you keep asking for.
5. **Explain and ask.** The decision is logged with its reason, e.g. *"you said 'concise', you behave 'detailed', seen 3x, score 0.90"*, and the agent offers to update your stated preference. Accepting it closes the conflict.

## Nemotron and Nebius Token Factory

The three tiers map to Nemotron 3 **Nano**, **Super** and **Ultra**, served through Nebius Token Factory (an OpenAI-compatible API). Nano and Super handle everyday calls; Ultra is reserved for when the agent is unsure about you.

During development the same code runs against local Ollama models, selected per tier in `.env`:

| Tier | Local development | Final (Token Factory) |
|---|---|---|
| Nano | `nemotron-3-nano:4b` | Nemotron 3 Nano |
| Super | `nemotron-3-nano:30b` | Nemotron 3 Super |
| Ultra | `deepseek-r1:32b` | Nemotron 3 Ultra |

Switching a tier to Token Factory is configuration only (`PROVIDER_ULTRA=nebius` plus the API key and model ID), so credits can be spent on one tier at a time.

> Status: development and testing so far ran on the local models above. Token Factory results and measurements will be added here.

## Quick start

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
python -m venv venv
venv\Scripts\activate            # Windows (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env           # cp on macOS/Linux

ollama pull nemotron-3-nano:4b
ollama pull nemotron-3-nano:30b
ollama pull deepseek-r1:32b
```

Run the tests (no models needed):

```bash
python -m pytest
```

Talk to the agent from the terminal:

```bash
python -m src.cli set me stated length concise      # you say you want short answers
python -m src.cli chat me "Explain how attention works, in more detail please"
python -m src.cli chat me "Can you elaborate on the softmax scaling?"   # contradiction: escalates
python -m src.cli profile me
python -m src.cli log me
python -m src.cli accept me length                   # adopt the observed preference
```

## HTTP API

```bash
python -m src.main      # http://127.0.0.1:8000, interactive docs at /docs
```

| Method | Path | What it does |
|---|---|---|
| `POST` | `/chat` | `{user_id, message}` → answer, tier chosen, reason, conflict scores, stats, and the "update your preference?" suggestion when there is one |
| `GET` | `/profiles/{user_id}` | Stated vs observed profiles, live conflict scores, and the tier the next request would get. Read-only |
| `GET` | `/decision-log/{user_id}?limit=10` | Recent routing decisions, newest first (timestamps in UTC) |
| `POST` | `/update-style` | `{user_id, dimension, value}` sets a stated preference |
| `POST` | `/accept` | `{user_id, dimension}` adopts the observed value as the stated one and closes the conflict |
| `GET` | `/health` | Status and which model serves each tier |

If the model backend is unreachable, `/chat` returns `503` with the reason.

## Dev dashboard

A small local dashboard runs the test suite and live-model scenarios and shows progress in real time: pass/fail, the model chosen for each request and why, tokens, speed (tok/s), time to first token, context used, and an estimated cost per run compared with sending everything to Ultra. Runs are saved to `data/history/`.

```bash
python -m src.devdash      # opens http://127.0.0.1:8765
```

The cost figures use Token Factory list prices taken from third-party listings (not confirmed on Nebius's own page) and token counts from the local models, so treat them as rough estimates. Prices can be overridden with `PRICE_*` variables in `.env`.

## Project layout

```
src/agent/        profiles, conflict detection, routing, signals, prompts, LLM client, decision logger
src/api/          FastAPI routes, request schemas, dependencies
src/main.py       API entry point
src/db/           SQLite connection and schema
src/devdash/      local test dashboard (stdlib server + one HTML page)
src/cli.py        terminal interface
tests/            unit tests (no network)
```

## Known limitations

- Style detection is keyword-based: fast and free, but it only recognises explicit phrasings and can be fooled by quoted text. Replacing it with a small classifier is the obvious next step.
- The observed profile does not yet use implicit signals such as edit distance or accept/reject buttons.
- Local stand-in models behave differently from Nemotron; prompts will be re-tuned on Token Factory.

## License

[MIT](LICENSE)

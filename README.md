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

1. **Learn from words and from behavior.**
   - *Words:* **Nemotron Nano reads each message** for style requests, in any language ("I'm still lost", "short version please", "no need for sources"). It must quote the user's exact words as evidence, or nothing is learned; words inside quotation marks are someone else's and don't count. One short call per message, thinking off, cached; if Nano is unavailable a keyword reader takes over.
   - *Behavior:* what you **do** with an answer: 👍, 👎 with a reason ("too long"), copy, or "try again".
   - A value only becomes a habit when it repeats recently (recent turns weigh more; old habits fade), so one offhand remark changes nothing.
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

## Design guidelines

Six rules, each enforced in code and checked by its own test ([details](docs/GUIDELINES.md)):

1. **Words are claims, behavior is evidence.** Neither wins automatically; only you change what you said.
2. **Weak signals teach nothing, and old ones fade.** A habit needs repeats; recent messages weigh more.
3. **This turn's instruction beats the profile, for this turn.** "Keep it short" is obeyed now, learned only if repeated.
4. **Spend power where it matters.** Greetings, thanks and quick lookups stay on Nano whatever the profile says.
5. **Explain every choice, never change a preference silently.** Every answer says why; adopting a habit needs your yes and can be undone.
6. **Be honest and keep it private.** Say when something failed or isn't known; never invent sources.

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

Step-by-step instructions for every test run and the report format: [docs/RUNNING.md](docs/RUNNING.md).

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

## The app

```bash
python -m src.main      # open http://127.0.0.1:8000
```

One page, three views (left sidebar, or `/?view=chat|both|proof`). **▶ Play the demo** runs the core contradiction in about a minute with real model calls.

- **Chat**: every answer is tagged with the model that wrote it; escalated answers say why in plain words ("you said *concise* but keep asking for *detailed*"). Under each answer: 👍, 👎 (pick a reason), Copy and ↻ Try again, all of which count as behavior. **The mirror** shows each style dimension as a spectrum, a ring for what you said and a dot for what you do, with the gap glowing as the contradiction grows. Click a word to change what you say you prefer. A one-click "Yes, update it" (with Undo) appears when the agent suggests adopting your observed habit.
- **Proof**: the evaluation report ([docs/EVAL.md](docs/EVAL.md)): success cases, known failures with reason codes, false-claim checks, and cost vs fit for always-Nano / router / always-Ultra.
- **Chat + Proof**: both side by side.
- Always visible: the evaluation summary, this session's tokens, speed and estimated cost, and the decision list in the sidebar.

Everything on the page is real data: session numbers come from your own calls, and the Proof view reads the latest `python -m src.eval` report (falling back to the committed snapshot `docs/eval-report.json`). If there is no report it says so.

## HTTP API

Served by the same app (interactive docs at `/docs`).

| Method | Path | What it does |
|---|---|---|
| `POST` | `/chat` | `{user_id, message}` → answer, tier chosen, reason, conflict scores, stats, and the "update your preference?" suggestion when there is one |
| `GET` | `/profiles/{user_id}` | Stated vs observed profiles, live conflict scores, and the tier the next request would get. Read-only |
| `GET` | `/decision-log/{user_id}?limit=10` | Recent routing decisions, newest first (timestamps in UTC) |
| `POST` | `/update-style` | `{user_id, dimension, value}` sets a stated preference |
| `POST` | `/accept` | `{user_id, dimension}` adopts the observed value as the stated one and closes the conflict |
| `POST` | `/feedback` | `{user_id, interaction_id, kind: up\|down\|copy, reason?}`: behavior on an answer; returns the updated profiles |
| `POST` | `/regenerate` | `{user_id}`: another answer to the last message (counts as a rejected draft) |
| `GET` | `/proof` | The latest evaluation reports: keyword reader, Nano reader, and the blind Nano-vs-Ultra comparison (`null` where missing) |
| `GET` | `/health` | Status and which model serves each tier |

If the model backend is unreachable, `/chat` returns `503` with the reason.

## Dev dashboard

A small local dashboard runs the test suite and live-model scenarios and shows progress in real time: pass/fail, the model chosen for each request and why, tokens, speed (tok/s), time to first token, context used, and an estimated cost per run compared with sending everything to Ultra. Runs are saved to `data/history/`.

```bash
python -m src.devdash      # opens http://127.0.0.1:8765
```

The cost figures use Token Factory list prices taken from third-party listings (not confirmed on Nebius's own page) and token counts from the local models, so treat them as rough estimates. Prices can be overridden with `PRICE_*` variables in `.env`.

## Personas, Hope and the team lessons

Five scripted demo users, each replayable with one click (Proof, or Settings → Profile): **Hope** and **Jordan** are careful (one shows the router working, one a contradiction it catches), **Casey** and **Sam** are capable but slip (an implicit complaint and a negated request are known gaps), and **Kai** fakes it (a made-up source is planted and flagged; the card says it is a stand-in). The answers come from the real configured models; the script is fixed. Write-ups: [docs/personas/](docs/personas/).

When a demo fails, **Hope reviews it** without calling a model (0 tokens): why it failed, a proposed fix, and a check that re-reads the failing message. A rule fix (for example "wall of text" means shorter answers) can be applied and is kept in the database, so the signal reader uses it from then on. Hope cannot write a better answer for free (that needs the model) and does not change the escalation thresholds. Each run is recorded as a lesson (a code and pass/fail per persona, never message text), shown under Settings → Memory. This shows a rule getting fixed, not a model getting smarter; fixes should be checked on held-out phrasings. API: `POST /hope/review`, `POST /hope/apply`, `GET /lessons`.

## Models, offline and online

[`models.toml`](models.toml) lists the model for each tier in an **offline** profile (local Ollama) and an **online** profile (Nebius Token Factory), with prices. Edit it and restart. The **Offline / Online** switch in Settings → Models picks a profile; online stays locked until the key is set and `confirmed = true`, asks for confirmation, and is never available in the public demo. Values in `.env` still override the file. Token counts are measured; prices are list prices until checked against Nebius billing, and the Budget page says which numbers are measured and which are estimated.

## Settings and the tutorial

Settings has Overview, Profile, Reports, Budget, Models, Memory, Notes, About and **Accessibility** (replay the tutorial, this README). A short guided tour starts on the first visit and walks through every view and settings tab; Skip tells you where to find it again.

## Evaluation

```bash
python -m src.eval        # 13 scripted conversations; add --live for the real models
```

It shows the three outcomes side by side: **success** (8 scenarios), **failure with a reason code** (4 known limits, e.g. `MISSED_SIGNAL`, `FALSE_SIGNAL`, `OVER_ROUTED`) and **false claims** (every explanation the agent gives is checked against its database; invented links are flagged; 7 of 7 injected false claims are caught). It also prices always-Nano vs always-Ultra vs the router. Method and limits: [docs/EVAL.md](docs/EVAL.md).

## Note bot

The square **Note bot** button left of the message box keeps notes on what the agent learns about you and why: *"You said you prefer short answers"*, *"Learned: you keep wanting detailed answers. Latest sign: "Can you elaborate…""*, *"Contradiction on length… answers moved up to Ultra"*, *"You marked 2 answers as too long"*, *"The budget stepped down a tier once"*. Hover it for a description; **!** opens Settings → Notes, with a short "where things stand" summary on top. A badge counts new notes.

It uses **no tokens**: notes are written from records the agent already keeps (an event log of the moments that matter, plus feedback, routing decisions and saved conversations) with fixed sentences, so nothing is invented. Quotes from a conversation that Memory deletes are erased from the notes too. API: `GET /notes/{user_id}` (`tokens_used` is always 0).

## Memory: past conversations

**＋ New chat** saves the conversation you're leaving to Memory as a compressed file (`data/memory/…/conversation-N.json.gz`) and starts a clean thread; your profile stays, since a new chat isn't a new person, and the model only sees the current conversation. Each saved conversation is **deleted after 10 days unless you press Keep**: the file and the message text both go (routing numbers like tier and score stay). A cleanup runs at startup, every hour, and when Memory is opened. Three days before a deletion the chat reminds you (at most once a day), with no model call and no tokens. Memory lets you Keep/Unkeep, Open, Download the `.json.gz`, or Delete now.

API: `POST /conversations/new`, `GET /memory/{user_id}`, `GET /memory/{user_id}/{id}` (+ `/download`), `POST /memory/keep`, `DELETE /memory/{user_id}/{id}`. Retention: `MEMORY_RETENTION_DAYS`.

## Budget limiter

Settings → Budget sets a $ limit (a simulated wallet: real token counts × Token Factory list prices, no real money). The router plans every answer within it:

- if the tier it wants would take more than 20% of what's left, it steps down (Ultra → Super → Nano) and says so: *"the router chose Ultra, but ~$0.006 is more than 20% of the $0.0097 left, so Super answered"*;
- under 25% left, thinking is off and answers are shorter;
- every answer's length is capped so it can't cost more than what's left; when nothing useful fits, it says the budget is used up (`402`).

Each answer shows the cost **predicted** before the call next to the cost **measured** after. Typical answer lengths are learned from real answers, so the predictions improve with use. API: `GET /budget/{user_id}`, `POST /budget {user_id, limit_usd, reset_spent}`.

## Efficiency choices

- **Thinking only where it pays.** Nano and Super answer with reasoning off (`reasoning_effort: none` on Ollama); Ultra keeps it. On local Nemotron Nano that cut "hi" from 33 to 13 output tokens.
- **The cheapest model does the reading.** The signal reader is Nano with thinking off, a 100-token cap and a cache, so re-weighing history costs nothing.
- **A 2-second bug found by measuring.** On Windows every new connection to `localhost` tried IPv6 first and cost ~2.2 s before falling back; `127.0.0.1` costs 0.16 s. Model calls now use `127.0.0.1` and cache model metadata, which removed seconds of overhead from every answer.

## Public demo mode

`PUBLIC_DEMO=1` for a hosted link: random private session ids (short, guessable ids are refused, and the "you are" box is hidden), a per-IP rate limit (`RATE_LIMIT_PER_MIN`, default 6), output caps per tier, and a daily spend cut-off (`DAILY_BUDGET_USD`, default $1). Limits are in memory: right for a single-instance demo, not for production.

## Project layout

```
src/agent/        profiles, conflict detection, routing, signals, prompts, LLM client, decision logger
src/api/          FastAPI routes, request schemas, dependencies
src/main.py       app entry point (API + product page)
src/web/          product page (single HTML file)
src/db/           SQLite connection and schema
src/agent/hope.py  zero-token failure review and kept reader rules
models.toml       offline / online models per tier, with prices
src/eval/         scenarios, grading with reason codes, false-claim checks, strategy pricing
src/devdash/      local test dashboard (stdlib server + one HTML page)
src/cli.py        terminal interface
tests/            unit tests (no network)
```

## Known limitations

- The Nano reader keeps only the first request it finds in a message (extra keys were mostly invented on the 4B model), so "no bullets and be casual" teaches one of the two. It misses some phrasings rather than guessing.
- Behavior signals cover thumbs, reasons, copy and retry; not yet edits to an answer or reading time.
- Local stand-in models behave differently from Nemotron; prompts will be re-tuned on Token Factory.

## License

[MIT](LICENSE)

## Reports

Every evaluation run is saved as one file in [`docs/reports/`](docs/reports/) and shown as a card on the **Proof** page (what it was, which models, where it ran, the headline). Click a card for the full detail: every scenario with its turns, tiers, tokens, cost and reason codes, or for the blind comparison, every answer and the judge's reason. Filter by **Local** (runs made while building, on local stand-in models or the simulator) or **Token Factory** (runs on the real Nemotron models). Where a run happened is read from the tier providers, not typed in. Save new ones with `python -m src.eval --save` and `python -m src.eval compare --save`. API: `GET /reports`, `GET /reports/{id}`.

# Running the tests yourself

Everything here can be run by hand. Each saved run appears in the app under **Settings → Reports**:
Token Factory runs on top, local runs below.

## 1. Setup (once)

```bash
pip install -r requirements.txt
```

**Local models (Ollama).** Install [Ollama](https://ollama.com), then pull the three tier models:

```bash
ollama pull nemotron-3-nano:4b      # nano
ollama pull nemotron-3-nano:30b     # super
ollama pull deepseek-r1:32b         # ultra (local stand-in)
```

Our local runs were made on an **NVIDIA RTX 5090 (32 GB VRAM)**. Smaller GPUs work, but the 30B/32B
models will be slower or need to be swapped in `.env` (`MODEL_SUPER`, `MODEL_ULTRA`).
Set `HARDWARE_LABEL` in `.env` so your reports name your machine.

**Nebius Token Factory (optional, spends credits).** Put your key in `.env`, never in code:

```bash
NEBIUS_API_KEY=...
LLM_PROVIDER=nebius
MODEL_NANO=<Token Factory id for Nemotron-3-Nano-30B-A3B>
MODEL_SUPER=<Token Factory id for Nemotron-3-Super-120B>
MODEL_ULTRA=<Token Factory id for Nemotron-3-Ultra-550B>
```

Copy the exact ids from the Token Factory model catalog. The key is never shown in the app or saved in reports.

## 2. Start the app

```bash
python -m src.main
```

Open http://localhost:8000. Settings → Profile lists the personas. Pick one, read its docs, press **Activate** and chat.

## 3. The test runs

| What | Command | Cost |
|---|---|---|
| Unit tests (no models) | `python -m pytest tests -q` | free |
| Scenarios, simulated answers | `python -m src.eval --save` | free |
| Scenarios, real models | `python -m src.eval --live --save` | local: free · Token Factory: credits |
| Scenarios with the Nano signal reader | `python -m src.eval --live --reader nano --save` | as above |
| Persona demos (Hope, Jordan, Casey, Sam, Kai) | `python -m src.eval.demos --save` | as above |
| Blind "is Ultra worth it?" comparison | `python -m src.eval compare --save` | as above |

Where a run counts as **local** or **Token Factory** is decided by the tier providers in `.env`:
any tier on Nebius makes it a Token Factory run.

## 4. Reading a report

Each card shows the models per tier, the personas involved, and the numbers that matter:
routing accuracy (under / over-routed), tokens in and out, latency per answer, cost, and false claims caught.
**More details** opens the full record: every turn, the routing reason, the answer and the tier.

Reports are saved compressed as `docs/reports/<id>.json.gz`. To read one outside the app:

```bash
python -c "import gzip,json,sys; print(json.dumps(json.loads(gzip.open(sys.argv[1]).read()), indent=1))" docs/reports/<id>.json.gz
```

Persona write-ups are in `docs/personas/`.

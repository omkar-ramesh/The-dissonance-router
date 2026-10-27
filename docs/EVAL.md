# Evaluation

`python -m src.eval` plays 20 scripted conversations and grades each with a verdict and,
for every failure, a reason code. It reports the cases where the agent works, the cases
where it doesn't, and whether it ever states something untrue.

```bash
python -m src.eval                                   # simulated model, keyword reader
python -m src.eval --live --reader nano --no-compare  # real models, Nemotron Nano reads the messages
python -m src.eval --live                            # also prices always-Nano / always-Ultra (slow locally)
python -m src.eval compare                           # blind Nano-vs-Ultra answer comparison (real models)
```

Output: `data/eval/report.md` / `report-nano.md` / `compare.md` (+ `.json`). Exit code 1 if
anything ended unexpectedly, including a held-out failure.

## Held-out (7 scenarios)

Written on 2026-10-28, after both signal readers were tuned, in phrasings neither was tested
on ("a wall of text", "spell out every step", "keep it chill", "back it with references"),
and never edited after the first run. They are graded like the success cases, and their
failures stay in the report. This is the answer to "the same person wrote the router and
the answer key": these cases were not tuned against.

## Who reads the messages: keywords vs Nemotron Nano

The same scenarios run with each signal reader. The keyword reader's known failures
(implicit complaint, another language, negation, quoted words) are re-run with Nano and
labelled FIXED or STILL FAILS. Nano must quote the user's words for each signal; unquoted or
quoted-from-someone-else signals are dropped. Its own calls are counted in the cost.

## Is Ultra worth it? Blind comparison

The router's claim is that a contradiction deserves a stronger model. `compare` tests that
directly: six contradiction cases (length, tone, structure, sourcing, revision), the
identical Ultra prompt sent to Nano and to Ultra, and a third model judging which answer
serves the user better without seeing model names. Each pair is judged twice with the order
swapped; a verdict that flips is counted as a tie (position bias). It also records cost,
latency, whether length answers open with a TL;DR, and invented links.

If Ultra doesn't win clearly, the report says the escalation claim should be narrowed.

## 1. Success (8 scenarios)

Routing, learning and reconciling as designed: an aligned user stays on Nano; a real
contradiction goes to Ultra; an undeclared habit gets Super and an offer to save it; topic
changes and one-off instructions teach nothing; "thanks" stays cheap; accepting a
suggestion closes the contradiction; an old habit fades.

## 2. Failure (4 scenarios, each with the reason code it should hit)

Failures are shown instead of hidden. These are known limits of the keyword detector;
each scenario names the codes it is expected to produce, and the run fails if one stops
failing in the expected way, so a limit can't quietly change.

| Scenario | What happens | Codes |
|---|---|---|
| Implicit complaint ("I still don't get it") | The user wants detail but never says so; nothing is learned | `MISSED_SIGNAL`, `UNDER_ROUTED` |
| Another language | Detector is English-only | `MISSED_SIGNAL`, `UNDER_ROUTED` |
| Negation ("don't make it shorter") | Learns the opposite of the request, escalates for nothing | `FALSE_SIGNAL`, `OVER_ROUTED` |
| Quoted phrase ("my boss says 'keep it short'") | Reads someone else's words as the user's | `FALSE_SIGNAL`, `OVER_ROUTED` |

## 3. False claims

Two checks, both automatic:

- **The agent's own explanations.** Every reason and suggestion ("you said *concise*, you
  behave *detailed*, seen 2x, score 0.80") is compared with the database, queried fresh with
  SQL: profile values, the number of unresolved conflicts, the score actually recorded, and
  the tier actually used. This runs on every turn of every scenario.
- **Sources in a model answer.** A link or DOI that no retrieval step supplied is flagged
  `UNVERIFIED_SOURCE`. In the `invented_source` scenario a model attaches a made-up arXiv
  link and the check flags it.

The checker is itself tested: seven false claims are injected (wrong repeat count, stated
value, observed value, score, tier, preference in the suggestion, an invented link) and all
seven must be caught.

## Cost vs fit

The success scenarios run three ways: everything on Nano, everything on Ultra, and the
router. For each turn the report records the cost, and whether the tier was right, too
small (*under*) or bigger than needed (*over*, wasted spend).

## Reason codes

| Code | Meaning |
|---|---|
| `MISSED_SIGNAL` | A real habit never reached the observed profile |
| `FALSE_SIGNAL` | Learned a habit the user does not have |
| `UNDER_ROUTED` | Smaller model than the situation needed |
| `OVER_ROUTED` | Bigger model than needed: wasted spend |
| `IGNORED_FORMAT` | The model ignored the answer format (live only) |
| `MISSING_SUGGESTION` | Didn't offer to update the preference |
| `BACKEND_FAILED` | The model call failed |
| `FALSE_CLAIM` | The agent's explanation contradicts its own records |
| `UNVERIFIED_SOURCE` | An answer cites a link nothing backs up |

## Limits

- **The "right tier" is our own annotation.** The same author wrote the router and the ideal
  tier for each turn, so "router 26/26" shows the router follows its design, not that the
  design is correct for everyone. The held-out set and the blind comparison exist to test
  outside that loop; the honest final test is real users.
- **The comparison has one judge model and six cases,** and locally the tiers are stand-ins
  (Nemotron Nano 4B vs DeepSeek-R1 32B, judged by Nemotron Nano 30B). It says "on these
  cases, with these models", not "in general".
- **Simulated runs say nothing about model quality.** Answers and token counts are
  simulated, format checks are skipped, and costs only compare strategies with each other.
  Real numbers come from `--live`.
- **Factual correctness is not checked.** The checker catches invented links and wrong
  statements about the agent's own state; it can't tell whether a model's prose is true.
  That would need reference answers.
- Scenario count is small (20 conversations). It shows how each case behaves; it is not a
  statistical benchmark.

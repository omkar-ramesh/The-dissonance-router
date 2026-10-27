# Design guidelines

Six rules the agent follows. Each one is enforced in code and checked by a test in
[`tests/test_guidelines.py`](../tests/test_guidelines.py) (one test class per guideline).

## 1. Words are claims, behavior is evidence

Neither wins automatically; the gap between them is the signal. Strong, repeated
behavior shapes the answer, but only the user can change what they said they prefer.

Behavior means both how someone asks (their messages, read by Nemotron Nano) and what they
do with answers: 👍, 👎 with a reason, copy, try again. On the same answer, what they did
outweighs what they wrote.

*In code:* `prompts.effective_style` follows observed behavior; the stated profile is
written only by the user (`/update-style`, `/accept`). Behavior signals:
`signals.feedback_signals`, `Agent.feedback`, `Agent.regenerate`.

## 2. Weak signals teach nothing, and old ones fade

A style request counts as a habit only when it repeats, and recent messages weigh more
than old ones. Each signal's weight decays by 0.85 per message: two requests within
about five messages make a habit (weight ≥ 1.5); a habit whose weight drops below 0.5
is forgotten.

*In code:* `signals.SignalExtractor.weigh` / `infer_profile`, fading in `Agent.observe`.

## 3. This turn's instruction beats the profile, for this turn

"Keep it short" is obeyed in this answer even when the profile says *detailed*, and is
learned from only if it keeps happening. On an Ultra escalation, the reconciliation
(e.g. a TL;DR line on top of the depth they keep asking for) is dropped only when the
message itself sides with what the user said; then they are simply doing what they said.

*In code:* `turn_signals` override the style in `prompts.build_system_prompt`; `_hints`
skips a dimension when this turn agrees with the stated value.

## 4. Spend power where the agent is unsure and it matters

Greetings, thanks and very short lookups have no style choices to make, so they stay on
Nano whatever the profile says, and they don't count as more evidence of a
contradiction. A short *style request* ("elaborate", "go deeper") is never treated as
quick: it is the signal.

*In code:* `relevance.is_light`, the `gate` in `Agent._analyze`.

## 5. Explain every choice, and never change a preference silently

Every answer records which model wrote it and why, in words a person can read. When a
pattern appears (at Super or Ultra) the agent asks before adopting it, and the change
can be undone.

*In code:* `routing_reason` on every decision; `suggestion` from score 0.4 up;
`Agent.accept_observed` returns the previous value for Undo.

## 6. Be honest and keep it private

Say when something failed or isn't known; never invent sources, quotes, links or
numbers. Profiles and history stay in a local SQLite file; only the conversation needed
to answer goes to the model provider.

The same rule applies to the agent's own reading of you: the Nano signal reader must quote
the words that show a preference, and a signal whose quote isn't in the message (or is
someone else's quoted words) is dropped, not trusted.

*In code:* the honesty rule in every system prompt; `extractor.parse` keeps only evidenced
signals; model outages return `503` with the reason instead of a made-up answer.

*Limit:* a prompt reduces invented content but cannot guarantee its absence. The
evaluation checks what can be checked automatically (our own explanations and cited
sources) and says so.

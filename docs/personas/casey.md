# Casey

**Type:** Known failure

Casey asked for detailed answers, then complains about length without using any style words.

## Profile
| Says | Does |
|---|---|
| length: detailed | "ugh, that's a wall of text, I'm not reading all that" |

## Script
1. "Explain TCP handshakes"
2. "ugh, that's a wall of text, I'm not reading all that"

## Expected result (ideal)
The complaint should count as a sign of wanting shorter answers, which contradicts "detailed".

## Why it is a known failure
The signal reader looks for style requests. A complaint with no style words ("shorter", "brief", "concise")
can be missed, so the router stays on Nano. Recorded as **KNOWN GAP**, reason code `MISSED_SIGNAL`. If the
reader catches it, the result shows **FIXED**. We show this on purpose: it is a real limit of the approach.

---
**Run it yourself:** Settings → Profile → pick this persona → **Activate**, then chat.
Scripted run of all personas: `python -m src.eval.demos --save` (results appear in Settings → Reports).

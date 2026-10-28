# Hope

**Type:** Success

Hope says she wants short answers, and she really does ask for short answers.

## Profile
| Says | Does |
|---|---|
| length: concise | "keep it short", "be brief" |

## Script
1. "Explain recursion, keep it short"
2. "What is a hash map? Be brief."

## Expected result
Every answer is served by **Nano**. There is no contradiction, so there is no reason to pay for a bigger model.

## What it proves
The router does not over-spend. Aligned users stay on the cheapest tier. If any turn goes above Nano the
run is marked **FAIL** with reason code `OVER_ROUTED`.

---
**Run it yourself:** Settings → Profile → pick this persona → **Activate**, then chat.
Scripted run of all personas: `python -m src.eval.demos --save` (results appear in Settings → Reports).

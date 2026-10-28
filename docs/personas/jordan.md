# Jordan

**Type:** Success

Jordan says he wants short answers, then keeps asking for more detail.

## Profile
| Says | Does |
|---|---|
| length: concise | "in more detail please", "can you elaborate" |

## Script
1. "Explain attention, in more detail please"
2. "Can you elaborate on the softmax?"

## Expected result
The second answer moves **up from Nano** (to Super or Ultra), and the routing reason names **length** as the
dimension in conflict.

## What it proves
This is the core idea of the project: the gap between what someone says and what they do is detected,
and the escalation is explained. A miss is recorded as `UNDER_ROUTED`.

---
**Run it yourself:** Settings → Profile → pick this persona → **Activate**, then chat.
Scripted run of all personas: `python -m src.eval.demos --save` (results appear in Settings → Reports).

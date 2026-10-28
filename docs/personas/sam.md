# Sam

**Type:** Known failure

Sam set "concise", then asks for the opposite using a negation.

## Profile
| Says | Does |
|---|---|
| length: concise | "Don't keep it short this time, I need every step of backprop" |

## Script
1. "Don't keep it short this time, I need every step of backprop"

## Expected result (ideal)
Read as a request for **detail**, which contradicts "concise".

## Why it is a known failure
A keyword reader sees "short" and can read the request backwards. With the Nemotron Nano reader enabled
it is usually read correctly, and the run shows **FIXED**. With keywords only it shows **KNOWN GAP**, reason
code `MISREAD_NEGATION`.

---
**Run it yourself:** Settings → Profile → pick this persona → **Activate**, then chat.
Scripted run of all personas: `python -m src.eval.demos --save` (results appear in Settings → Reports).

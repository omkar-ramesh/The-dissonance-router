"""A simulated wallet: the router plans each answer within a $ budget.

No real money moves. Costs are token counts x Token Factory list prices (config.PRICES),
the same numbers the rest of the app reports.

Rules, in order, when a limit is set:
  1. Nothing left: don't answer; say so.
  2. Spend at most SHARE of what is left on one answer, so the budget lasts: if the router's
     tier would cost more than that, step down (Ultra -> Super -> Nano) and say why.
  3. Tight (less than TIGHT of the limit left): thinking off and shorter answers.
  4. Always cap the answer's length: normally to its share, and never past what is left.
     The output cap is exact; the input is estimated before the call (with a 1.5x margin),
     and the signal reader's few tokens are counted after.
Each plan records its predicted cost, so it can be compared with the measured cost.
"""
from contextlib import closing
from dataclasses import dataclass, field
from typing import Optional

from src.db.database import get_connection

ORDER = ("nano", "super", "ultra")
EXPECTED_OUT = {"nano": 500, "super": 800, "ultra": 1500}   # typical answer length, tokens (Ultra includes thinking)
TIGHT_CAP = {"nano": 400, "super": 600, "ultra": 900}
SHARE = 0.20
TIGHT = 0.25
MIN_ANSWER_TOKENS = 60   # below this an answer isn't worth giving: say the budget is used up instead
INPUT_MARGIN = 1.5       # input tokens are estimated (~4 chars each) before the call; leave room for error


class BudgetExceeded(RuntimeError):
    """No money left for an answer."""


def cost(prices: dict, tier: str, input_tokens: int, output_tokens: int) -> float:
    p = prices.get(tier) or {"in": 0.0, "out": 0.0}
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000


@dataclass
class Plan:
    tier: str
    predicted_usd: float
    limited: bool = False
    reasoning: Optional[str] = None          # "off" when tight
    max_tokens: Optional[int] = None
    reason: Optional[str] = None             # plain words, set whenever the budget changed something
    limit_usd: float = 0.0
    spent_usd: float = 0.0
    chat_kwargs: dict = field(default_factory=dict)


class Typical:
    """Each tier's typical answer length, learned from real answers (a moving average).

    Starts from EXPECTED_OUT. The first local run predicted Nano 110% and Super 39% low with
    the fixed guesses, because real answers ran longer; learning closes that gap. Answers cut
    off by a cap are skipped, since they'd teach a length that was forced, not natural.
    """

    def __init__(self, weight: float = 0.3):
        self.weight, self.tokens = weight, dict(EXPECTED_OUT)

    def learn(self, tier: str, output_tokens: int, cap: Optional[int] = None) -> None:
        if output_tokens <= 0 or (cap and output_tokens >= 0.95 * cap):
            return
        self.tokens[tier] = round((1 - self.weight) * self.tokens[tier] + self.weight * output_tokens)


def plan(router_tier: str, input_tokens: int, limit: float, spent: float, prices: dict,
         typical: Optional[dict] = None) -> Plan:
    """Decide tier, thinking and length for one answer under the budget.

    `typical` maps tier -> expected answer tokens (learned); EXPECTED_OUT if not given.
    """
    typical = {**EXPECTED_OUT, **(typical or {})}
    if not limit:
        return Plan(router_tier, cost(prices, router_tier, input_tokens, typical[router_tier]))
    remaining = limit - spent
    if remaining <= 0:
        raise BudgetExceeded(f"Budget used up: ${spent:.4f} of ${limit:.4f} spent. Raise it in Settings.")
    tight = remaining < limit * TIGHT
    allowance = remaining * SHARE

    def expected(t):
        return min(TIGHT_CAP[t], typical[t]) if tight else typical[t]

    candidates = ORDER[:ORDER.index(router_tier) + 1][::-1]          # the router's tier, then cheaper ones
    tier = next((t for t in candidates if cost(prices, t, input_tokens, expected(t)) <= allowance), "nano")

    # Hard caps on the answer's length. Input tokens are only estimated before the call, so
    # they get a safety margin. `fit`: never more than what is left. `share`: normally no more
    # than this answer's share, so the budget lasts (a short answer beats none).
    price = prices.get(tier) or {"in": 0.0, "out": 0.0}
    in_cost = input_tokens * INPUT_MARGIN * price["in"] / 1_000_000

    def tokens_for(usd):
        return int((usd - in_cost) * 1_000_000 / price["out"]) if price["out"] else 10 ** 9

    fit = tokens_for(remaining)
    if fit < MIN_ANSWER_TOKENS:
        raise BudgetExceeded(f"Budget nearly used up: ${remaining:.5f} left isn't enough for a useful answer. "
                             "Raise it in Settings.")
    cap = min(fit, max(tokens_for(allowance), MIN_ANSWER_TOKENS))
    if tight:
        cap = min(cap, TIGHT_CAP[tier])
    out = min(expected(tier), cap)

    notes = []
    if tier != router_tier:
        would = cost(prices, router_tier, input_tokens, expected(router_tier))
        notes.append(f"the router chose {router_tier.title()}, but ~${would:.4f} is more than {int(SHARE * 100)}% "
                     f"of the ${remaining:.4f} left, so {tier.title()} answered")
    if tight:
        notes.append(f"under {int(TIGHT * 100)}% of the budget is left, so thinking is off and the answer is shorter")
    elif cap < typical[tier]:
        notes.append(f"the answer is capped at {cap} tokens to stay within its share of the ${remaining:.4f} left")
    kwargs = {"max_tokens": cap}
    if tight:
        kwargs["reasoning"] = "off"
    return Plan(tier, cost(prices, tier, input_tokens, out), limited=True, reasoning=kwargs.get("reasoning"),
                max_tokens=cap, reason=("Budget limit: " + "; ".join(notes) + ".") if notes else None,
                limit_usd=limit, spent_usd=spent, chat_kwargs=kwargs)


class BudgetStore:
    """Per-user limit and running spend (the `budgets` table)."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get(self, user_id: str) -> tuple[float, float]:
        with closing(get_connection(self.db_path)) as conn:
            row = conn.execute("SELECT limit_usd, spent_usd FROM budgets WHERE user_id = ?", (user_id,)).fetchone()
        return (row["limit_usd"], row["spent_usd"]) if row else (0.0, 0.0)

    def set_limit(self, user_id: str, limit_usd: float, reset_spent: bool = False) -> None:
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                conn.execute("INSERT INTO budgets (user_id, limit_usd) VALUES (?, ?) "
                             "ON CONFLICT(user_id) DO UPDATE SET limit_usd = excluded.limit_usd", (user_id, limit_usd))
                if reset_spent:
                    conn.execute("UPDATE budgets SET spent_usd = 0 WHERE user_id = ?", (user_id,))

    def add_spent(self, user_id: str, usd: float) -> None:
        if not usd:
            return
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                conn.execute("INSERT INTO budgets (user_id, spent_usd) VALUES (?, ?) "
                             "ON CONFLICT(user_id) DO UPDATE SET spent_usd = spent_usd + excluded.spent_usd", (user_id, usd))

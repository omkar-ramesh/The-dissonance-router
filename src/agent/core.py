"""Agent: ties profiles, conflict detection, routing and logging together."""
from pathlib import Path
from typing import Optional

from src import config
from src.agent.memory import MemoryStore
from src.agent.notes import NoteBook
from src.agent import budget
from src.agent.budget import BudgetExceeded, BudgetStore
from src.agent.conflict_detector import ConflictDetector
from src.agent.honesty import find_unbacked_sources
from src.agent.llm_client import LLMClient, LLMError
from src.agent.logger import DecisionLogger
from src.agent.profiles import DIMENSIONS, ProfileManager
from src.agent.prompts import build_messages
from src.agent.relevance import is_light
from src.agent.routing import SUPER_THRESHOLD, ModelRouter
from src.agent.signals import DOWN_REASONS, FADE_WEIGHT, SignalExtractor, feedback_signals
from src.db.database import init_db

# How many recent user messages are weighed for the observed profile. Older evidence
# has decayed to almost nothing by then (see signals.DECAY).
EVIDENCE_WINDOW = 20
QUICK_REASON = "Quick request: no style choices to make, so it stays on Nano."
# How many prior turns are replayed to the model as conversation context.
HISTORY_TURNS = 4
FEEDBACK_KINDS = ("up", "down", "copy")      # "regenerate" goes through Agent.regenerate
REGENERATE_HINT = ("The user asked for a different answer to this message. "
                   "Take another approach instead of rephrasing the previous one.")


class Agent:
    """Decides which model tier should answer a request, answers it, and records why."""

    def __init__(self, db_path: str, llm: Optional[LLMClient] = None, extractor: Optional[SignalExtractor] = None,
                 memory_dir: Optional[str] = None):
        """`extractor` reads messages for style requests: keywords by default, or a
        NanoExtractor (Nemotron Nano, keywords as fallback). `memory_dir` holds archived
        conversations (default: a `memory` folder next to the database)."""
        init_db(db_path)
        self.memory = MemoryStore(db_path, memory_dir or Path(db_path).parent / "memory", config.MEMORY_RETENTION_DAYS)
        self.profiles = ProfileManager(db_path)
        self.detector = ConflictDetector()
        self.router = ModelRouter()
        self.logger = DecisionLogger(db_path)
        self.signals = extractor or SignalExtractor()
        self.budgets = BudgetStore(db_path)
        self.notes = NoteBook(db_path)      # zero-token notes on what was learned and why
        self.typical = budget.Typical()     # learned answer length per tier, for cost predictions
        self.llm = llm

    def analyze_request(self, user_id: str, user_request: str) -> dict:
        """Score, route and log a request without calling a model."""
        analysis = self._analyze(user_id, user_request)
        self._record(analysis)
        return analysis

    def respond(self, user_id: str, message: str) -> dict:
        """Full loop: learn from the message, route, answer with the chosen tier, log.

        Returns the analysis plus `response`, `model_id`, `latency_ms`, `usage` and
        `observed_updates` (dimensions this message changed).
        """
        if self.llm is None:
            raise RuntimeError("Agent was created without an LLM client")

        observed_updates = self.observe(user_id, message)
        analysis = self._analyze(user_id, message, gate=True)
        # Context is this conversation only; archived chats don't leak into a new one.
        history = self.logger.get_recent_decisions(user_id, limit=HISTORY_TURNS, after_id=self.memory.boundary(user_id))
        try:
            result, spend = self._call(analysis, history, message)
        except (LLMError, BudgetExceeded):
            # The observed profile may already have changed because of this message, so
            # keep it as evidence even though no answer came (empty response = not replayed).
            self._record(analysis)
            raise
        interaction_id = self._record(analysis, result.text)
        self._note_updates(user_id, observed_updates, interaction_id, message)
        return self._answer(analysis, result, interaction_id, observed_updates, spend)

    def _call(self, analysis: dict, history: list, message: str, hint: str = ""):
        """Plan the call within the user's budget (it may step down a tier or shorten the
        answer, and says so in `routing_reason`), then call the model. Mutates `analysis`
        so what is logged is what actually happened."""
        def build(a):
            msgs = build_messages(a, history, message)
            if hint:
                msgs[-1] = {**msgs[-1], "content": f"{msgs[-1]['content']}\n\n({hint})"}
            return msgs

        messages = build(analysis)
        limit, spent = self.budgets.get(analysis["user_id"])
        chosen = analysis["model_chosen"]
        spend = budget.plan(chosen, sum(len(m["content"]) for m in messages) // 4, limit, spent, config.PRICES,
                            self.typical.tokens)
        if spend.reason:
            # The logged reason must name the tier that really answered (the false-claim check
            # verifies it), so the router's tier wording is replaced, its evidence kept.
            detail = analysis["routing_reason"].partition("Strongest dissonance:")
            evidence = f" Strongest dissonance:{detail[2]}" if detail[1] else ""
            if spend.tier != chosen:
                analysis.update(model_chosen=spend.tier, budget_from=chosen, routing_reason=spend.reason + evidence)
                messages = build(analysis)
            else:
                analysis["routing_reason"] = f"{analysis['routing_reason']} {spend.reason}"
        result = (self.llm.chat(spend.tier, messages, **spend.chat_kwargs) if spend.limited
                  else self.llm.chat(spend.tier, messages))
        return result, spend

    def regenerate(self, user_id: str) -> dict:
        """The user asked for another answer to their last message.

        That is behavior (a rejected draft), so it is recorded as feedback and can change the
        observed profile. The answer is replaced in place: the message stays one row, so its
        words are not counted twice.
        """
        if self.llm is None:
            raise RuntimeError("Agent was created without an LLM client")
        start = self.memory.boundary(user_id)
        last = self.logger.get_recent_decisions(user_id, limit=1, after_id=start)
        if not last or not last[0]["agent_response"]:
            raise ValueError("There is no answer to regenerate yet")
        row = last[0]
        self.logger.log_feedback(user_id, row["id"], "regenerate")
        observed_updates = self.observe(user_id)
        analysis = self._analyze(user_id, row["user_message"], gate=True)
        history = self.logger.get_recent_decisions(user_id, limit=HISTORY_TURNS + 1, after_id=start)[1:]
        result, spend = self._call(analysis, history, row["user_message"], REGENERATE_HINT)
        self._note_updates(user_id, observed_updates, row["id"], "asked for another answer")
        self.logger.update_decision(row["id"], analysis["model_chosen"], analysis["conflict_score"],
                                    analysis["routing_reason"], result.text)
        return self._answer(analysis, result, row["id"], observed_updates, spend)

    def feedback(self, user_id: str, interaction_id: int, kind: str, reason: str = "") -> dict:
        """Record what the user did with an answer (thumbs up/down with an optional reason,
        copy) and re-learn the observed profile from it. Returns the new profile state."""
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"Unknown feedback {kind!r}; expected one of {FEEDBACK_KINDS}")
        if reason and (kind != "down" or reason not in DOWN_REASONS):
            raise ValueError(f"Unknown reason {reason!r}; expected one of {sorted(DOWN_REASONS)}")
        if not self.logger.get_decision(user_id, interaction_id):
            raise LookupError(f"No answer {interaction_id} for this user")
        self.logger.log_feedback(user_id, interaction_id, kind, reason)
        updates = self.observe(user_id)
        sign = {"up": "👍 on an answer", "copy": "copied an answer"}.get(kind, f"👎 {reason.replace('_', ' ')}".strip())
        self._note_updates(user_id, updates, interaction_id, sign)
        return {**self.inspect(user_id), "observed_updates": updates}

    def _answer(self, analysis: dict, result, interaction_id: int, observed_updates: dict, spend) -> dict:
        reader = getattr(self.signals, "take_calls", lambda: [])()
        # What this answer really cost (measured tokens x list prices), signal reads included.
        measured = budget.cost(config.PRICES, result.tier, result.input_tokens, result.output_tokens)
        reads = sum(budget.cost(config.PRICES, r.tier, r.input_tokens, r.output_tokens) for r in reader)
        self.budgets.add_spent(analysis["user_id"], measured + reads)
        self.typical.learn(result.tier, result.output_tokens, spend.max_tokens)
        limit, spent = self.budgets.get(analysis["user_id"])
        return {
            **analysis,
            "interaction_id": interaction_id,
            "response": result.text,
            # Links the agent can't vouch for (it has no search): shown to the user as unverified.
            "unverified_sources": find_unbacked_sources(result.text),
            "model_id": result.model,
            "latency_ms": result.latency_ms,
            "usage": result.usage,
            "stats": result.stats(),
            # The Nano signal reader's own calls this turn: counted, not hidden in the totals.
            "reader_stats": [r.stats() for r in reader],
            "observed_updates": observed_updates,
            "budget": {
                "limit_usd": limit, "spent_usd": spent, "limited": spend.limited,
                "router_chose": analysis.get("budget_from", analysis["model_chosen"]),
                "reason": spend.reason, "max_tokens": spend.max_tokens, "thinking_off": spend.reasoning == "off",
                # predicted before the call vs measured after: how good the planning is
                "predicted_usd": spend.predicted_usd, "measured_usd": measured, "reader_usd": reads,
            },
        }

    def observe(self, user_id: str, message: Optional[str] = None) -> dict:
        """Update the observed profile from recent turns (plus `message`, if given).

        Each turn's evidence is what its words asked for plus what the user did with its
        answer (feedback). New habits need recent, repeated evidence; a habit whose
        evidence has decayed is forgotten (back to "balanced"). Returns
        {dimension: new_value} for every dimension that changed.
        """
        window = EVIDENCE_WINDOW - 1 if message is not None else EVIDENCE_WINDOW
        recent = list(reversed(self.logger.get_recent_decisions(user_id, window, self.notes.forgotten_after(user_id))))   # oldest first
        actions = self.logger.get_feedback(user_id, [d["id"] for d in recent])
        turns = [{**self.signals.extract(d["user_message"]), **feedback_signals(actions.get(d["id"], []))}
                 for d in recent]                     # behavior wins over words on the same turn
        if message is not None:
            turns.append(self.signals.extract(message))
        inferred = self.signals.infer_from_signals(turns)
        weights = self.signals.weigh_signals(turns)
        current = self.profiles.get_observed_profile(user_id)

        changed = {}
        for dim, value in current.items():
            new = inferred.get(dim)
            if new is None and value != "balanced" and weights.get(dim, {}).get(value, 0.0) < FADE_WEIGHT:
                new = "balanced"
            if new is not None and new != value:
                self.profiles.update_observed(user_id, dim, new)
                changed[dim] = new
        return changed

    def inspect(self, user_id: str) -> dict:
        """Current profiles, conflict scores and the tier the next request would get.

        Read-only: nothing is logged, so viewing a profile never counts as a request.
        """
        analysis = self._analyze(user_id, "")
        keys = ("user_id", "stated_profile", "observed_profile", "conflicts_detected", "overall_score",
                "conflict_score", "model_chosen", "routing_reason", "highest_conflict", "suggestion")
        return {key: analysis[key] for key in keys}

    def accept_observed(self, user_id: str, dimension: str) -> str:
        """User agreed: adopt observed behavior as the stated preference and close the conflict.

        Returns the previous stated value so the change can be undone.
        """
        previous = self.profiles.get_stated_profile(user_id)[dimension]
        observed = self.profiles.get_observed_profile(user_id)[dimension]
        self.profiles.update_stated(user_id, dimension, observed)
        self.logger.resolve_conflicts(user_id, dimension)
        self.notes.log(user_id, "accepted", dimension, observed)
        return previous

    def forget_observed(self, user_id: str) -> None:
        """Drop everything learned so far: past turns stop counting, the observed profile goes back to unknown."""
        last = self.logger.get_recent_decisions(user_id, 1)
        self.notes.log(user_id, "forgot", interaction_id=last[0]["id"] if last else 0)
        for dim in self.profiles.get_observed_profile(user_id):
            self.profiles.update_observed(user_id, dim, "balanced")
            self.logger.resolve_conflicts(user_id, dim)

    def set_stated(self, user_id: str, dimension: str, value: str) -> None:
        """The user says how they like answers. Noted, so the note bot can say when it changed."""
        before = self.profiles.get_stated_profile(user_id)[dimension]
        self.profiles.update_stated(user_id, dimension, value)
        if value != before:
            self.notes.log(user_id, "said", dimension, value)

    def _note_updates(self, user_id: str, updates: dict, interaction_id: int, sign: str) -> None:
        """A habit was learned or faded: note it, with the latest sign that led there."""
        for dim, value in updates.items():
            if value == "balanced":
                self.notes.log(user_id, "faded", dim, self.profiles.get_stated_profile(user_id)[dim], "", interaction_id)
            else:
                self.notes.log(user_id, "learned", dim, value, sign[:80], interaction_id)

    def _analyze(self, user_id: str, user_request: str, gate: bool = False) -> dict:
        """Score the user's dissonance and pick a tier. No side effects.

        Routing uses the strongest single dimension, not the average: one strong
        contradiction (e.g. "concise" vs "detailed") averages to 0.14 across five
        dimensions and would never escalate. `overall_score` is still reported.

        With `gate`, a quick message (greeting, thanks, short lookup) that carries no
        style signal stays on Nano whatever the profile says (guideline 4).
        """
        stated, observed = self.profiles.get_all_profiles(user_id)
        repeats = self.logger.get_conflict_repeats(user_id, stated, observed)
        conflicts = self.detector.detect_conflicts(stated, observed, repeats)
        dimension, score = self.detector.get_highest_conflict(conflicts)
        turn_signals = self.signals.extract(user_request)
        gated = gate and score > 0 and not turn_signals and is_light(user_request)

        model = "nano" if gated else self.router.route(score)
        highest = None
        detail = None
        if score > 0:
            highest = {
                "dimension": dimension,
                "stated": stated[dimension],
                "observed": observed[dimension],
                "score": score,
                "times_seen": repeats.get(dimension, 0) + 1,
            }
            detail = (
                f"Strongest dissonance: {dimension} (you said '{highest['stated']}', "
                f"you behave '{highest['observed']}', seen {highest['times_seen']}x, "
                f"score {score:.2f})."
            )

        # Offer to adopt the observed habit whenever there is a real pattern, at Super as well
        # as Ultra, so a learned habit never keeps someone on a higher tier with no way out.
        suggestion = None
        if score >= SUPER_THRESHOLD:
            if stated[dimension] == "balanced":
                suggestion = (
                    f"You never set a {dimension} preference, but you keep asking for "
                    f"'{observed[dimension]}'. Set it as your preference?"
                )
            else:
                suggestion = (
                    f"You said you prefer {dimension} = '{stated[dimension]}', but you consistently "
                    f"behave as '{observed[dimension]}'. Update your stated preference?"
                )

        return {
            "user_id": user_id,
            "request": user_request,
            "model_chosen": model,
            "conflict_score": score,
            "overall_score": conflicts["overall_score"],
            "routing_reason": QUICK_REASON if gated else self.router.get_routing_reason(score, detail),
            "stated_profile": stated,
            "observed_profile": observed,
            "conflicts_detected": {dim: conflicts[dim] for dim in DIMENSIONS},
            "highest_conflict": highest,
            "suggestion": suggestion,
            "turn_signals": turn_signals,
            "gated": gated,
        }

    def _record(self, analysis: dict, agent_response: str = "") -> int:
        """Persist the decision, and every contradiction seen in it unless the turn was a
        quick message (a "thanks" is not more evidence of the contradiction). Returns its id."""
        user_id = analysis["user_id"]
        stated, observed = analysis["stated_profile"], analysis["observed_profile"]
        seen = self.logger.get_conflict_repeats(user_id, stated, observed)
        first = []                                  # contradictions seen for the first time: worth a note
        for dim, score in analysis["conflicts_detected"].items():
            if score > 0 and not analysis.get("gated"):
                if not seen.get(dim):
                    first.append(dim)
                self.logger.log_conflict(user_id, dim, stated[dim], observed[dim], score)
        interaction_id = self.logger.log_decision(
            user_id,
            analysis["request"],
            analysis["model_chosen"],
            analysis["conflict_score"],
            analysis["routing_reason"],
            agent_response,
        )
        for dim in first:
            self.notes.log(user_id, "contradiction", dim, f"{stated[dim]}>{observed[dim]}",
                           analysis["model_chosen"].title(), interaction_id)
        return interaction_id

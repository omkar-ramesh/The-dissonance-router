"""HTTP endpoints. Thin: every rule lives in the Agent."""
import json
from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from src import config
from src.agent.core import Agent
from src.agent.honesty import find_unbacked_sources
from src.api import limits
from src.api.dependencies import get_agent
from src.eval import reports
from fastapi.responses import FileResponse

from src.api.schemas import (AcceptRequest, BudgetUpdate, ChatRequest, FeedbackRequest, KeepRequest, NewChat,
                             RegenerateRequest, StyleUpdate)

router = APIRouter()
UserIdPath = Path(min_length=1, max_length=64, pattern=r"^[\w.-]+$")

_ROOT = FilePath(__file__).resolve().parents[2]
# A report from a local `python -m src.eval` run wins; the committed snapshot is the fallback
# so a hosted copy still has something real to show.
PROOF_SOURCES = [_ROOT / "data" / "eval" / "report.json", _ROOT / "docs" / "eval-report.json"]
NANO_SOURCES = [_ROOT / "data" / "eval" / "report-nano.json", _ROOT / "docs" / "eval-report-nano.json"]
COMPARE_SOURCES = [_ROOT / "data" / "eval" / "compare.json", _ROOT / "docs" / "compare-report.json"]


@router.get("/health")
def health() -> dict:
    """Liveness plus which model and backend serves each tier."""
    return {
        "status": "ok",
        "tiers": {
            tier: {"model": model, "provider": config.TIER_PROVIDERS[tier], "reasoning": config.REASONING[tier]}
            for tier, model in config.MODEL_TIERS.items()
        },
        "signal_reader": config.SIGNAL_EXTRACTOR,
        "public_demo": config.PUBLIC_DEMO,
    }


@router.get("/proof")
def proof() -> dict:
    """The latest evaluation reports (see docs/EVAL.md); `null` where none exists.

    `report`: scenarios with the keyword signal reader; `nano_report`: the same scenarios
    read by Nemotron Nano; `compare`: the blind Nano-vs-Ultra answer comparison.
    Model answers are left out; the page needs verdicts, reason codes and costs, not prose.
    """
    report, source = _load(PROOF_SOURCES)
    nano, nano_source = _load(NANO_SOURCES)
    compare, compare_source = _load(COMPARE_SOURCES)
    return {"report": report, "source": source, "nano_report": nano, "nano_source": nano_source,
            "compare": compare, "compare_source": compare_source}


@router.get("/reports")
def reports_list() -> dict:
    """One card per saved run (docs/reports/): what it was, which models, where it ran, the headline."""
    return {"reports": reports.cards()}


@router.get("/reports/{report_id}")
def report_detail(report_id: str) -> dict:
    found = reports.get(report_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No such report.")
    return found


def _load(paths):
    for path in paths:
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            for scenario in report.get("scenarios", []):
                for turn in scenario.get("turns", []):
                    turn.pop("response", None)
            return report, path.name
    return None, None


@router.post("/chat", dependencies=[Depends(limits.admit)])
def chat(req: ChatRequest, agent: Agent = Depends(get_agent)) -> dict:
    """Learn from the message, route it, answer with the chosen tier, and explain why."""
    limits.require_private_id(req.user_id)
    result = agent.respond(req.user_id, req.message)
    limits.GUARD.spend(limits.cost_of(result))
    return result


@router.post("/regenerate", dependencies=[Depends(limits.admit)])
def regenerate(req: RegenerateRequest, agent: Agent = Depends(get_agent)) -> dict:
    """Another answer to the last message. Counts as a rejected draft (behavior)."""
    limits.require_private_id(req.user_id)
    try:
        result = agent.regenerate(req.user_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    limits.GUARD.spend(limits.cost_of(result))
    return result


@router.post("/conversations/new")
def new_chat(req: NewChat, agent: Agent = Depends(get_agent)) -> dict:
    """Archive the current conversation (compressed, 10 days unless kept) and start a new one.
    The profile stays: a new chat is not a new person."""
    limits.require_private_id(req.user_id)
    agent.memory.sweep()
    return {"archived": agent.memory.archive(req.user_id), "retention_days": agent.memory.retention}


@router.get("/memory/{user_id}")
def memory(user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    """Archived conversations, newest first, with days left; and which need a reminder."""
    limits.require_private_id(user_id)
    deleted = agent.memory.sweep()
    return {"items": agent.memory.list(user_id), "reminders": agent.memory.reminders(user_id),
            "retention_days": agent.memory.retention, "deleted_now": deleted}


@router.get("/memory/{user_id}/{conversation_id}")
def memory_read(conversation_id: int, user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    limits.require_private_id(user_id)
    try:
        return agent.memory.read(user_id, conversation_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/memory/{user_id}/{conversation_id}/download")
def memory_download(conversation_id: int, user_id: str = UserIdPath, agent: Agent = Depends(get_agent)):
    """The compressed file itself (.json.gz)."""
    limits.require_private_id(user_id)
    try:
        path = agent.memory.file(user_id, conversation_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return FileResponse(path, media_type="application/gzip", filename=path.name)


@router.post("/memory/keep")
def memory_keep(req: KeepRequest, agent: Agent = Depends(get_agent)) -> dict:
    """Keep a conversation forever (kept=true), or put it back on the 10-day clock."""
    limits.require_private_id(req.user_id)
    try:
        return agent.memory.keep(req.user_id, req.conversation_id, req.kept)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/memory/{user_id}/{conversation_id}")
def memory_delete(conversation_id: int, user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    """Delete now: the file and the message text (routing numbers stay)."""
    limits.require_private_id(user_id)
    try:
        agent.memory.delete(user_id, conversation_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"deleted": conversation_id}


@router.get("/notes/{user_id}")
def notes(user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    """The note bot: what the agent learned about you and why, newest first. Written from
    records with fixed sentences: no model call, no tokens (`tokens_used` is always 0)."""
    limits.require_private_id(user_id)
    return agent.notes.render(user_id, agent.inspect(user_id))


@router.get("/budget/{user_id}")
def get_budget(user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    """The simulated wallet: limit (0 = none), spent so far, and how the router uses it."""
    limits.require_private_id(user_id)
    return _budget(agent, user_id)


@router.post("/budget")
def set_budget(req: BudgetUpdate, agent: Agent = Depends(get_agent)) -> dict:
    """Set the $ limit the router plans within (0 = no limit). Takes effect on the next answer."""
    limits.require_private_id(req.user_id)
    agent.budgets.set_limit(req.user_id, req.limit_usd, req.reset_spent)
    return _budget(agent, req.user_id)


def _budget(agent, user_id):
    from src.agent import budget
    limit, spent = agent.budgets.get(user_id)
    return {"user_id": user_id, "limit_usd": limit, "spent_usd": spent,
            "remaining_usd": max(0.0, limit - spent) if limit else None,
            "rules": {"share_per_answer": budget.SHARE, "tight_below": budget.TIGHT},
            "prices_per_1m_tokens": config.PRICES}


@router.post("/feedback")
def feedback(req: FeedbackRequest, agent: Agent = Depends(get_agent)) -> dict:
    """Thumbs up/down (with an optional reason) or copy on an answer. Returns the updated
    profiles and `observed_updates` if the behavior changed a learned habit."""
    limits.require_private_id(req.user_id)
    try:
        return agent.feedback(req.user_id, req.interaction_id, req.kind, req.reason)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/profiles/{user_id}")
def profiles(user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    """Stated vs observed profiles and the live conflict scores. Read-only."""
    limits.require_private_id(user_id)
    return agent.inspect(user_id)


@router.get("/decision-log/{user_id}")
def decision_log(
    user_id: str = UserIdPath,
    limit: int = Query(10, ge=1, le=100),
    current: bool = Query(False, description="only the current conversation (since the last New chat)"),
    agent: Agent = Depends(get_agent),
) -> dict:
    """Most recent routing decisions, newest first, with what the user did with each answer."""
    limits.require_private_id(user_id)
    decisions = agent.logger.get_recent_decisions(user_id, limit, after_id=agent.memory.boundary(user_id) if current else 0)
    actions = agent.logger.get_feedback(user_id, [d["id"] for d in decisions])
    for decision in decisions:
        decision["timestamp"] = _iso_utc(decision["timestamp"])
        decision["feedback"] = actions.get(decision["id"], [])
        decision["unverified_sources"] = find_unbacked_sources(decision["agent_response"])
    return {"user_id": user_id, "decisions": decisions}


@router.post("/update-style")
def update_style(req: StyleUpdate, agent: Agent = Depends(get_agent)) -> dict:
    """Set one preference. Returns the updated profiles and conflict scores."""
    limits.require_private_id(req.user_id)
    if req.kind == "observed" and config.PUBLIC_DEMO:
        raise HTTPException(status_code=403, detail="Observed preferences are learned, not set, in the public demo.")
    update = agent.set_stated if req.kind == "stated" else agent.profiles.update_observed
    try:
        update(req.user_id, req.dimension, req.value)
    except ValueError as e:  # e.g. length="verbose"
        raise HTTPException(status_code=422, detail=str(e)) from e
    return agent.inspect(req.user_id)


@router.post("/forget-observed/{user_id}")
def forget_observed(user_id: str, agent: Agent = Depends(get_agent)) -> dict:
    """Forget everything learned from behaviour; the stated profile is untouched."""
    limits.require_private_id(user_id)
    agent.forget_observed(user_id)
    return agent.inspect(user_id)


@router.post("/accept")
def accept(req: AcceptRequest, agent: Agent = Depends(get_agent)) -> dict:
    """The user said yes to "update your stated preference?": adopt the observed value.

    `previous` is the old stated value; sending it back to /update-style undoes the change.
    """
    limits.require_private_id(req.user_id)
    previous = agent.accept_observed(req.user_id, req.dimension)
    return {**agent.inspect(req.user_id), "previous": previous}


def _iso_utc(timestamp: str) -> str:
    """SQLite's CURRENT_TIMESTAMP is UTC without a zone; make that explicit for clients."""
    return timestamp.replace(" ", "T") + "Z" if timestamp and not timestamp.endswith("Z") else timestamp

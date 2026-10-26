"""HTTP endpoints. Thin: every rule lives in the Agent."""
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from src import config
from src.agent.core import Agent
from src.api.dependencies import get_agent
from src.api.schemas import AcceptRequest, ChatRequest, StyleUpdate

router = APIRouter()
UserIdPath = Path(min_length=1, max_length=64, pattern=r"^[\w.-]+$")


@router.get("/health")
def health() -> dict:
    """Liveness plus which model and backend serves each tier."""
    return {
        "status": "ok",
        "tiers": {
            tier: {"model": model, "provider": config.TIER_PROVIDERS[tier]}
            for tier, model in config.MODEL_TIERS.items()
        },
    }


@router.post("/chat")
def chat(req: ChatRequest, agent: Agent = Depends(get_agent)) -> dict:
    """Learn from the message, route it, answer with the chosen tier, and explain why."""
    return agent.respond(req.user_id, req.message)


@router.get("/profiles/{user_id}")
def profiles(user_id: str = UserIdPath, agent: Agent = Depends(get_agent)) -> dict:
    """Stated vs observed profiles and the live conflict scores. Read-only."""
    return agent.inspect(user_id)


@router.get("/decision-log/{user_id}")
def decision_log(
    user_id: str = UserIdPath,
    limit: int = Query(10, ge=1, le=100),
    agent: Agent = Depends(get_agent),
) -> dict:
    """Most recent routing decisions, newest first."""
    decisions = agent.logger.get_recent_decisions(user_id, limit)
    for decision in decisions:
        decision["timestamp"] = _iso_utc(decision["timestamp"])
    return {"user_id": user_id, "decisions": decisions}


@router.post("/update-style")
def update_style(req: StyleUpdate, agent: Agent = Depends(get_agent)) -> dict:
    """Set one preference. Returns the updated profiles and conflict scores."""
    update = agent.profiles.update_stated if req.kind == "stated" else agent.profiles.update_observed
    try:
        update(req.user_id, req.dimension, req.value)
    except ValueError as e:  # e.g. length="verbose"
        raise HTTPException(status_code=422, detail=str(e)) from e
    return agent.inspect(req.user_id)


@router.post("/accept")
def accept(req: AcceptRequest, agent: Agent = Depends(get_agent)) -> dict:
    """The user said yes to "update your stated preference?": adopt the observed value."""
    agent.accept_observed(req.user_id, req.dimension)
    return agent.inspect(req.user_id)


def _iso_utc(timestamp: str) -> str:
    """SQLite's CURRENT_TIMESTAMP is UTC without a zone; make that explicit for clients."""
    return timestamp.replace(" ", "T") + "Z" if timestamp and not timestamp.endswith("Z") else timestamp

"""Request bodies. Responses are the Agent's own dicts."""
from typing import Literal

from pydantic import BaseModel, Field

# Same alphabet the dev tools use; keeps ids safe in URLs and logs.
USER_ID = Field(min_length=1, max_length=64, pattern=r"^[\w.-]+$")
Dimension = Literal["tone", "length", "sourcing", "structure", "revision"]


class ChatRequest(BaseModel):
    user_id: str = USER_ID
    message: str = Field(min_length=1, max_length=8000)


class StyleUpdate(BaseModel):
    user_id: str = USER_ID
    dimension: Dimension
    value: str
    # "observed" is for seeding demos and tests; real observed values are learned from messages.
    kind: Literal["stated", "observed"] = "stated"


class AcceptRequest(BaseModel):
    user_id: str = USER_ID
    dimension: Dimension


class FeedbackRequest(BaseModel):
    user_id: str = USER_ID
    interaction_id: int = Field(ge=1)
    kind: Literal["up", "down", "copy"]
    # Why a thumbs-down: too_long, too_short, too_formal, too_casual, needs_sources, ...
    reason: str = Field("", max_length=32)


class RegenerateRequest(BaseModel):
    user_id: str = USER_ID


class NewChat(BaseModel):
    user_id: str = USER_ID


class KeepRequest(BaseModel):
    user_id: str = USER_ID
    conversation_id: int = Field(ge=1)
    kept: bool = True


class BudgetUpdate(BaseModel):
    user_id: str = USER_ID
    limit_usd: float = Field(ge=0, le=100)     # 0 = no limit
    reset_spent: bool = False


class NoteMarks(BaseModel):
    user_id: str = USER_ID
    marks: dict[str, Literal["hidden", "saved", "none"]] = Field(max_length=200)   # note key -> new state


class HopeReview(BaseModel):
    code: str = Field(max_length=40, pattern=r"^[A-Z_]+$")
    say: str = Field(default="", max_length=2000)
    answer: str = Field(default="", max_length=4000)
    unverified: list[str] = Field(default_factory=list, max_length=20)
    persona: str = Field(default="", max_length=20)


class ModeRequest(BaseModel):
    mode: Literal["offline", "online"]
    confirm: bool = False

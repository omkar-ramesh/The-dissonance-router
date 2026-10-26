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

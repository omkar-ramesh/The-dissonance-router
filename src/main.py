"""FastAPI app.

    python -m src.main          # http://127.0.0.1:8000  (interactive docs at /docs)
"""
import asyncio
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from src import config
from src.agent.budget import BudgetExceeded
from src.agent.llm_client import LLMError
from src.agent.memory import MemoryStore
from src.api.routes import router
from src.db.database import init_db

log = logging.getLogger("dissonance")
SWEEP_EVERY_S = 3600


def sweep_memory() -> int:
    """Delete archived conversations past their 10 days that nobody kept."""
    init_db(config.DB_PATH)
    return MemoryStore(config.DB_PATH, config.MEMORY_DIR, config.MEMORY_RETENTION_DAYS).sweep()


async def _sweeper():
    while True:
        try:
            deleted = await asyncio.to_thread(sweep_memory)
            if deleted:
                log.info("memory: deleted %d expired conversation(s)", deleted)
        except Exception:                      # a failed sweep must not stop the server; try again next hour
            log.exception("memory sweep failed")
        await asyncio.sleep(SWEEP_EVERY_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_sweeper())     # runs at startup, then every hour
    yield
    task.cancel()


app = FastAPI(
    title="The Dissonance Router",
    description="Routes each request to Nemotron Nano, Super or Ultra by how much the user's "
                "stated preferences contradict their observed behavior, and explains why.",
    version="0.3.0",
    lifespan=lifespan,
)
app.include_router(router)

WEB_PAGE = Path(__file__).parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """The product page: chat, the mirror (stated vs observed), router and decision log."""
    return FileResponse(WEB_PAGE)


TUTORIAL_DIR = WEB_PAGE.parent / "tutorials"


@app.get("/tutorials/{name}", include_in_schema=False)
def tutorial_image(name: str) -> FileResponse:
    """Screenshots for the Tutorials window (src/web/tutorials/<entry>.webp)."""
    path = TUTORIAL_DIR / name
    if not re.fullmatch(r"[a-z0-9-]{1,60}\.(webp|png)", name) or not path.is_file():
        raise HTTPException(status_code=404, detail="No such tutorial image.")
    return FileResponse(path)


@app.exception_handler(BudgetExceeded)
def budget_used_up(request: Request, exc: BudgetExceeded) -> JSONResponse:
    """The user's simulated budget can't pay for another answer."""
    return JSONResponse(status_code=402, content={"detail": str(exc)})


@app.exception_handler(LLMError)
def model_unavailable(request: Request, exc: LLMError) -> JSONResponse:
    """The model backend is down or misconfigured: a service problem, not a client error."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host=config.API_HOST, port=config.API_PORT)

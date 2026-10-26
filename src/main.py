"""FastAPI app.

    python -m src.main          # http://127.0.0.1:8000  (interactive docs at /docs)
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src import config
from src.agent.llm_client import LLMError
from src.api.routes import router

app = FastAPI(
    title="The Dissonance Router",
    description="Routes each request to Nemotron Nano, Super or Ultra by how much the user's "
                "stated preferences contradict their observed behavior, and explains why.",
    version="0.3.0",
)
app.include_router(router)


@app.exception_handler(LLMError)
def model_unavailable(request: Request, exc: LLMError) -> JSONResponse:
    """The model backend is down or misconfigured: a service problem, not a client error."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host=config.API_HOST, port=config.API_PORT)

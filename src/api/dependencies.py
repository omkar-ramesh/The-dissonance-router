"""Shared dependencies. Tests swap the agent via app.dependency_overrides[get_agent]."""
from functools import lru_cache

from src import config
from src.agent.core import Agent
from src.agent.extractor import NanoExtractor
from src.agent.llm_client import LLMClient


@lru_cache
def get_agent() -> Agent:
    """One agent per process. A config error (e.g. missing Token Factory key) is raised
    on every request until fixed, rather than cached."""
    llm = LLMClient.from_config()
    extractor = NanoExtractor(llm, config.DB_PATH) if config.SIGNAL_EXTRACTOR == "nano" else None
    return Agent(config.DB_PATH, llm, extractor, config.MEMORY_DIR)

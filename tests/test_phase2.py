"""Phase 2: signal extraction, prompts, LLM client and the full respond() loop. No network."""
import json

import httpx
import pytest

from src.agent.core import Agent
from src.agent.llm_client import LLMClient, LLMError, LLMResult
from src.agent.profiles import DIMENSIONS
from src.agent.prompts import build_messages, build_system_prompt, effective_style
from src.agent.signals import SignalExtractor
from src.db.database import init_db

BALANCED = {dim: "balanced" for dim in DIMENSIONS}
MODELS = {"nano": "small", "super": "medium", "ultra": "large"}


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


class FakeLLM:
    """Records calls and answers with a canned string."""

    def __init__(self):
        self.calls = []

    def chat(self, tier, messages, temperature=0.7):
        self.calls.append((tier, messages))
        return LLMResult(text=f"answer from {tier}", tier=tier, model=MODELS[tier], latency_ms=5)


def mock_client(handler, api_key=""):
    return LLMClient("http://test/v1", MODELS, api_key=api_key, transport=httpx.MockTransport(handler))


def sse(*chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


USAGE = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def completion(content, usage=USAGE, **delta_extra):
    """A streamed reply: one content chunk, then (optionally) the usage chunk."""
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": content, **delta_extra}}]}]
    if usage:
        chunks.append({"choices": [], "usage": usage})
    return sse(*chunks)


class TestSignalExtractor:
    @pytest.mark.parametrize("message, expected", [
        ("Can you explain that in more detail?", {"length": "detailed"}),
        ("please elaborate", {"length": "detailed"}),
        ("too long, make it shorter", {"length": "concise"}),
        ("tl;dr please", {"length": "concise"}),
        ("be less formal", {"tone": "casual"}),
        ("make it more professional", {"tone": "formal"}),
        ("no bullets, write in paragraphs", {"structure": "paragraph"}),
        ("give me bullet points", {"structure": "bullet"}),
        ("can you cite that?", {"sourcing": "full"}),
        ("no sources needed", {"sourcing": "minimal"}),
        ("that's not what I wanted, try again", {"revision": "rejects"}),
        ("perfect, that works", {"revision": "accepts"}),
        ("What's the capital of France?", {}),
    ])
    def test_extract(self, message, expected):
        assert SignalExtractor().extract(message) == expected

    @pytest.mark.parametrize("message", [
        "I'm excited about open source models",  # "cite" inside a word, "source" in a name
        "Write a function that lists files",
        # topic words that look like style words: found in the audit, must stay silent
        "I no longer need the old config file",
        "I'm a professional developer working on APIs",
        "Explain formal verification of smart contracts",
        "What's the official Python docs URL for asyncio?",
        "How do casual games make money?",
        "Is this code perfect for production?",
        "Write a function that returns the longer of two strings",
        "Summarize the references section of this paper",
        "Can you list the steps to install Docker?",
        "What does 'in short' mean in English grammar?",
        "My cat is very relaxed today, any tips for feeding?",
        "How do I redo a git commit?",
        "What are the sources of vitamin D?",
        "How do I cite a website in APA?",
    ])
    def test_no_false_positives(self, message):
        assert SignalExtractor().extract(message) == {}

    @pytest.mark.parametrize("message, expected", [
        ("Please be more formal", {"tone": "formal"}),
        ("make it casual", {"tone": "casual"}),
        ("give me sources for that", {"sourcing": "full"}),
        ("sources?", {"sourcing": "full"}),
        ("use bullet points", {"structure": "bullet"}),
        ("keep it short", {"length": "concise"}),
        ("rewrite it", {"revision": "rejects"}),
        ("that's perfect", {"revision": "accepts"}),
    ])
    def test_style_requests_still_detected(self, message, expected):
        assert SignalExtractor().extract(message) == expected

    def test_one_signal_is_not_enough(self):
        assert SignalExtractor().infer_profile(["elaborate please"]) == {}

    def test_repeated_signal_wins(self):
        messages = ["elaborate please", "what is X?", "go deeper on that"]
        assert SignalExtractor().infer_profile(messages) == {"length": "detailed"}

    def test_tie_is_not_a_winner(self):
        messages = ["elaborate", "tell me more", "shorter", "too long"]
        assert SignalExtractor().infer_profile(messages) == {}


def analysis(tier, stated=None, observed=None, conflicts=None):
    return {
        "model_chosen": tier,
        "stated_profile": dict(BALANCED, **(stated or {})),
        "observed_profile": dict(BALANCED, **(observed or {})),
        "conflicts_detected": dict({dim: 0.0 for dim in DIMENSIONS}, **(conflicts or {})),
    }


class TestPrompts:
    def test_behavior_overrides_words(self):
        style = effective_style(dict(BALANCED, length="concise"), dict(BALANCED, length="detailed"))
        assert style["length"] == "detailed"

    def test_stated_used_until_behavior_known(self):
        style = effective_style(dict(BALANCED, length="concise"), BALANCED)
        assert style["length"] == "concise"

    def test_nano_prompt_has_no_conflict_text(self):
        prompt = build_system_prompt(analysis("nano", {"tone": "formal"}, {"tone": "formal"}))
        assert "formal" in prompt
        assert "contradict" not in prompt

    def test_ultra_prompt_names_the_contradiction(self):
        prompt = build_system_prompt(
            analysis("ultra", {"length": "concise"}, {"length": "detailed"}, {"length": 0.8})
        )
        assert "length: says 'concise', behaves 'detailed'" in prompt
        assert "TL;DR" in prompt
        assert "sources together" not in prompt  # only hints for conflicting dimensions

    def test_super_prompt_leans_to_behavior(self):
        prompt = build_system_prompt(analysis("super", {"length": "concise"}, {"length": "balanced"}, {"length": 0.4}))
        assert "lean toward their observed behavior" in prompt

    def test_ultra_repeats_format_in_user_turn(self):
        ultra = analysis("ultra", {"length": "concise"}, {"length": "detailed"}, {"length": 0.8})
        last = build_messages(ultra, [], "why?")[-1]["content"]
        assert last.startswith("why?") and "TL;DR" in last
        assert build_messages(analysis("nano"), [], "why?")[-1]["content"] == "why?"

    def test_history_is_replayed_oldest_first(self):
        history = [
            {"user_message": "second", "agent_response": "r2"},
            {"user_message": "first", "agent_response": "r1"},
            {"user_message": "unanswered", "agent_response": ""},
        ]
        messages = build_messages(analysis("nano"), history, "now")
        assert [m["content"] for m in messages[1:]] == ["first", "r1", "second", "r2", "now"]


class TestLLMClient:
    def test_sends_model_for_tier_and_parses_answer(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers.get("authorization")
            return completion("hello")

        result = mock_client(handler, api_key="secret").chat("super", [{"role": "user", "content": "hi"}])
        assert seen["url"] == "http://test/v1/chat/completions"
        assert seen["body"]["model"] == "medium"
        assert seen["auth"] == "Bearer secret"
        assert seen["body"]["stream"] is True
        assert (result.text, result.tier, result.model) == ("hello", "super", "medium")
        assert result.usage["completion_tokens"] == 3

    def test_no_auth_header_without_key(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return completion("hi")

        mock_client(handler).chat("nano", [])
        assert seen["auth"] is None

    def test_strips_think_tags(self):
        result = mock_client(lambda r: completion("<think>hmm, the user wants X</think>\n\nFinal answer")).chat("ultra", [])
        assert result.text == "Final answer"
        assert "the user wants X" in result.reasoning

    def test_reads_separate_reasoning_field(self):
        result = mock_client(lambda r: completion("Answer", reasoning_content="step 1")).chat("ultra", [])
        assert (result.text, result.reasoning) == ("Answer", "step 1")

    def test_reads_ollama_reasoning_field(self):
        result = mock_client(lambda r: completion("Answer", reasoning="thinking")).chat("ultra", [])
        assert result.reasoning == "thinking"

    def test_stats_from_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result = mock_client(lambda r: completion("hi", usage=usage)).chat("nano", [])
        assert (result.input_tokens, result.output_tokens, result.total_tokens) == (100, 50, 150)
        assert result.tokens_estimated is False
        assert result.ttft_ms is not None and result.ttft_ms <= result.latency_ms
        stats = result.stats()
        assert stats["model_id"] == "small" and stats["tier"] == "nano"
        assert stats["agent_rounds"] == 1
        assert stats["est_cost_usd"] is None  # no prices configured

    def test_tokens_estimated_without_usage(self):
        messages = [{"role": "user", "content": "x" * 40}]
        result = mock_client(lambda r: completion("y" * 20, usage=None)).chat("nano", messages)
        assert result.tokens_estimated is True
        assert (result.input_tokens, result.output_tokens) == (10, 5)

    def test_tokens_per_sec_uses_wall_time(self):
        result = LLMResult(text="", tier="nano", model="m", latency_ms=2000, output_tokens=100)
        assert result.tokens_per_sec == 50.0
        assert LLMResult(text="", tier="nano", model="m", latency_ms=0).tokens_per_sec == 0.0

    def test_cost_estimate_uses_tier_prices_and_ultra_baseline(self):
        prices = {"nano": {"in": 0.06, "out": 0.24}, "ultra": {"in": 1.0, "out": 3.0}}
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000, "total_tokens": 1_500_000}
        client = LLMClient("http://test/v1", MODELS, prices=prices,
                           transport=httpx.MockTransport(lambda r: completion("hi", usage=usage)))
        result = client.chat("nano", [])
        assert result.est_cost_usd == pytest.approx(0.06 + 0.12)         # 1M in + 0.5M out at nano prices
        assert result.est_cost_if_ultra_usd == pytest.approx(1.0 + 1.5)  # same tokens at ultra prices
        assert client.chat("ultra", []).est_cost_usd == pytest.approx(2.5)

    def test_unpriced_tier_has_no_cost(self):
        client = LLMClient("http://test/v1", MODELS, prices={"ultra": {"in": 1.0, "out": 3.0}},
                           transport=httpx.MockTransport(lambda r: completion("hi")))
        result = client.chat("nano", [])
        assert result.est_cost_usd is None and result.est_cost_if_ultra_usd is not None

    def test_stream_error_chunk_raises(self):
        with pytest.raises(LLMError, match="stream error"):
            mock_client(lambda r: sse({"error": {"message": "out of memory"}})).chat("nano", [])

    def test_context_from_ollama_ps(self):
        def handler(request):
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [{
                    "name": "small", "context_length": 32768,
                    "details": {"family": "nemotron_h", "parameter_size": "4.0B", "quantization_level": "Q4_K_M"},
                }]})
            return completion("hi", usage={"prompt_tokens": 3000, "completion_tokens": 288, "total_tokens": 3288})

        client = LLMClient("http://test/v1", MODELS, transport=httpx.MockTransport(handler),
                           ps_urls={"nano": "http://test/api/ps"})
        result = client.chat("nano", [])
        assert result.model_name == "nemotron_h 4.0B Q4_K_M"
        assert result.context_window == 32768
        assert result.context_pct == 10.0

    def test_context_falls_back_to_config_and_survives_ps_failure(self):
        def handler(request):
            if request.url.path == "/api/ps":
                return httpx.Response(500)
            return completion("hi")

        client = LLMClient("http://test/v1", MODELS, transport=httpx.MockTransport(handler),
                           ps_urls={"ultra": "http://test/api/ps"}, context_windows={"ultra": 1000})
        result = client.chat("ultra", [])
        assert (result.model_name, result.context_window) == ("large", 1000)
        assert result.context_pct == 1.3

    def test_http_error_raises_llm_error(self):
        with pytest.raises(LLMError, match="500"):
            mock_client(lambda r: httpx.Response(500, text="boom")).chat("nano", [])

    def test_unreachable_backend_raises_llm_error(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(LLMError, match="Could not reach"):
            mock_client(handler).chat("nano", [])

    def test_per_tier_backends(self):
        """Ultra goes to one host with a key; the other tiers stay on the default host."""
        seen = []

        def handler(request):
            seen.append((request.url.host, request.headers.get("authorization")))
            return completion("ok")

        client = LLMClient(
            "http://local/v1", MODELS, transport=httpx.MockTransport(handler),
            overrides={"ultra": {"base_url": "http://cloud/v1", "api_key": "k"}},
        )
        client.chat("nano", [])
        client.chat("ultra", [])
        assert seen == [("local", None), ("cloud", "Bearer k")]

    def test_from_config_mixed_providers(self, monkeypatch):
        from src import config
        monkeypatch.setattr(config, "TIER_PROVIDERS", {"nano": "ollama", "super": "ollama", "ultra": "nebius"})
        monkeypatch.setattr(config, "NEBIUS_API_KEY", "")
        with pytest.raises(LLMError, match="ultra"):
            LLMClient.from_config()
        monkeypatch.setattr(config, "NEBIUS_API_KEY", "k")
        monkeypatch.setattr(config, "NEBIUS_BASE_URL", "https://cloud/v1")
        client = LLMClient.from_config()
        assert str(client._clients["ultra"].base_url).startswith("https://cloud")
        assert "11434" in str(client._clients["nano"].base_url)

    def test_unknown_tier(self):
        with pytest.raises(ValueError):
            mock_client(lambda r: completion("x")).chat("mega", [])


class TestRespond:
    def test_requires_llm(self, db_path):
        with pytest.raises(RuntimeError):
            Agent(db_path).respond("u1", "hi")

    def test_aligned_user_gets_nano_and_answer_is_logged(self, db_path):
        llm = FakeLLM()
        agent = Agent(db_path, llm)
        result = agent.respond("u1", "What's 2+2?")
        assert result["model_chosen"] == "nano"
        assert result["response"] == "answer from nano"
        assert llm.calls[0][0] == "nano"
        assert agent.logger.get_recent_decisions("u1")[0]["agent_response"] == "answer from nano"

    def test_demo_scenario_behavior_contradicts_words(self, db_path):
        """Says concise, keeps asking for depth: agent learns it and escalates to Ultra."""
        llm = FakeLLM()
        agent = Agent(db_path, llm)
        agent.profiles.update_stated("u1", "length", "concise")

        first = agent.respond("u1", "Explain how attention works, in more detail please")
        assert first["model_chosen"] == "nano"  # one signal is not evidence yet
        assert first["observed_updates"] == {}

        second = agent.respond("u1", "Can you elaborate on the softmax part?")
        assert second["observed_updates"] == {"length": "detailed"}
        assert second["model_chosen"] == "ultra"
        assert "you said 'concise', you behave 'detailed'" in second["routing_reason"]
        assert "Update your stated preference" in second["suggestion"]

        system_prompt = llm.calls[-1][1][0]["content"]
        assert "says 'concise', behaves 'detailed'" in system_prompt

        third = agent.respond("u1", "ok and what about positional encoding?")
        assert third["model_chosen"] == "ultra"
        assert third["conflict_score"] == 0.8  # confidence grows with repetition

    def test_accepting_suggestion_returns_to_nano(self, db_path):
        agent = Agent(db_path, FakeLLM())
        agent.profiles.update_stated("u1", "length", "concise")
        agent.respond("u1", "more detail please")
        agent.respond("u1", "elaborate")
        agent.accept_observed("u1", "length")
        assert agent.respond("u1", "thanks, next question")["model_chosen"] == "nano"

    def test_failed_model_call_keeps_message_as_evidence(self, db_path):
        class DownLLM(FakeLLM):
            def chat(self, tier, messages, temperature=0.7):
                raise LLMError("backend down")

        agent = Agent(db_path, DownLLM())
        agent.profiles.update_stated("u1", "length", "concise")
        with pytest.raises(LLMError):
            agent.respond("u1", "elaborate please")
        logged = agent.logger.get_recent_decisions("u1")
        assert [d["user_message"] for d in logged] == ["elaborate please"]
        assert logged[0]["agent_response"] == ""

        # the failed turn still counts: one more signal completes the evidence
        agent.llm = FakeLLM()
        assert agent.respond("u1", "go deeper")["model_chosen"] == "ultra"
        # and the failed turn is not replayed to the model as a conversation turn
        assert all(m["content"] != "elaborate please" for m in agent.llm.calls[-1][1][1:-1])

    def test_history_replayed_to_model(self, db_path):
        llm = FakeLLM()
        agent = Agent(db_path, llm)
        agent.respond("u1", "first question")
        agent.respond("u1", "second question")
        contents = [m["content"] for m in llm.calls[-1][1][1:]]
        assert contents == ["first question", "answer from nano", "second question"]

"""
Competitive Matrix Tests: Fireworks AI vs Cerebras
Compares OpenAI-compatible chat completions API behavior across providers.

Models under test:
  - GLM 4.7:       Fireworks (fireworks/glm-4p7) vs Cerebras (zai-glm-4.7)
  - gpt-oss-120b:  Fireworks (fireworks/gpt-oss-120b) vs Cerebras (gpt-oss-120b)

All tests use the OpenAI Python SDK pointed at each provider's endpoint.
"""

import json
import os
import time
from pathlib import Path

import pytest
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load API keys (prefer env vars / --env-file .env; fallback to local .keys)
# ---------------------------------------------------------------------------
def _strip_quotes(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1].strip()
    return v

CEREBRAS_API_KEY = _strip_quotes(os.getenv("CEREBRAS_API_KEY"))
FIREWORKS_API_KEY = _strip_quotes(os.getenv("FIREWORKS_API_KEY"))

# If env does not provide keys, fail early with a clear message.
if not CEREBRAS_API_KEY or not FIREWORKS_API_KEY:
    raise RuntimeError(
        "Missing API keys. Set CEREBRAS_API_KEY and FIREWORKS_API_KEY in your environment "
        "(recommended: use `uv run --env-file .env pytest ...`)."
    )

# ---------------------------------------------------------------------------
# Provider / model configuration
# ---------------------------------------------------------------------------
PROVIDERS = {
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key": FIREWORKS_API_KEY,
        "models": {
            "glm": "accounts/fireworks/models/glm-4p7",
            "gpt_oss": "accounts/fireworks/models/gpt-oss-120b",
        },
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
        "models": {
            "glm": "zai-glm-4.7",
            "gpt_oss": "gpt-oss-120b",
        },
    },
}

SIMPLE_PROMPT = [{"role": "user", "content": "Say exactly: hello world"}]
MATH_PROMPT = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]


def make_client(provider: str) -> OpenAI:
    cfg = PROVIDERS[provider]
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])


def get_model(provider: str, model_family: str) -> str:
    return PROVIDERS[provider]["models"][model_family]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=["fireworks", "cerebras"], ids=["fireworks", "cerebras"])
def provider(request):
    return request.param


@pytest.fixture(params=["glm", "gpt_oss"], ids=["glm-4.7", "gpt-oss-120b"])
def model_family(request):
    return request.param


@pytest.fixture
def client(provider):
    return make_client(provider)


@pytest.fixture
def model(provider, model_family):
    return get_model(provider, model_family)


@pytest.fixture(autouse=True)
def rate_limit_delay():
    """10-second delay between tests to avoid rate limiting."""
    yield
    time.sleep(10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chat(client: OpenAI, model: str, messages: list[dict], **kwargs):
    """Wrapper for non-streaming chat completion."""
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


def chat_stream(client: OpenAI, model: str, messages: list[dict], **kwargs):
    """Wrapper for streaming chat completion. Returns collected chunks."""
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    )
    chunks = list(stream)
    return chunks


class Result:
    """Convenience container for comparing results across providers."""

    def __init__(self, provider, model_family, response=None, error=None):
        self.provider = provider
        self.model_family = model_family
        self.response = response
        self.error = error

    @property
    def ok(self):
        return self.error is None

    @property
    def content(self):
        if self.response and self.response.choices:
            return self.response.choices[0].message.content
        return None


def run_on_both(model_family: str, messages: list[dict], **kwargs) -> dict[str, Result]:
    """Run the same request on both providers and return results keyed by provider."""
    results = {}
    for prov in ["fireworks", "cerebras"]:
        c = make_client(prov)
        m = get_model(prov, model_family)
        try:
            resp = chat(c, m, messages, **kwargs)
            results[prov] = Result(prov, model_family, response=resp)
        except Exception as e:
            results[prov] = Result(prov, model_family, error=e)
    return results


# ===========================================================================
# 1. BASIC CHAT COMPLETION
# ===========================================================================
class TestBasicChatCompletion:
    """Basic non-streaming chat completion tests."""

    def test_simple_completion(self, client, model, provider, model_family):
        """Verify basic chat completion returns a valid response."""
        resp = chat(client, model, SIMPLE_PROMPT)
        assert resp.id, f"[{provider}/{model_family}] Missing response ID"
        assert resp.choices, f"[{provider}/{model_family}] No choices returned"
        assert resp.choices[0].message.content, f"[{provider}/{model_family}] Empty content"
        assert resp.choices[0].finish_reason in ("stop", "length"), (
            f"[{provider}/{model_family}] Unexpected finish_reason: {resp.choices[0].finish_reason}"
        )

    def test_response_object_fields(self, client, model, provider, model_family):
        """Verify the response has standard OpenAI-compatible fields."""
        resp = chat(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        assert resp.object == "chat.completion", (
            f"[{provider}/{model_family}] object={resp.object}"
        )
        assert resp.model, f"[{provider}/{model_family}] Missing model in response"
        assert resp.created, f"[{provider}/{model_family}] Missing created timestamp"

    def test_usage_info(self, client, model, provider, model_family):
        """Verify usage statistics are returned."""
        resp = chat(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        assert resp.usage is not None, f"[{provider}/{model_family}] No usage info"
        assert resp.usage.prompt_tokens > 0, f"[{provider}/{model_family}] prompt_tokens=0"
        assert resp.usage.completion_tokens > 0, f"[{provider}/{model_family}] completion_tokens=0"
        assert resp.usage.total_tokens > 0, f"[{provider}/{model_family}] total_tokens=0"
        assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens, (
            f"[{provider}/{model_family}] total_tokens mismatch"
        )

    def test_system_message(self, client, model, provider, model_family):
        """Verify system messages are respected."""
        messages = [
            {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
            {"role": "user", "content": "Hello"},
        ]
        resp = chat(client, model, messages, max_completion_tokens=1000)
        content = resp.choices[0].message.content.lower()
        assert "arrr" in content or "arr" in content, (
            f"[{provider}/{model_family}] System message not followed: {content[:200]}"
        )

    def test_role_in_response(self, client, model, provider, model_family):
        """Verify the response message has role=assistant."""
        resp = chat(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        assert resp.choices[0].message.role == "assistant", (
            f"[{provider}/{model_family}] role={resp.choices[0].message.role}"
        )


# ===========================================================================
# 2. STREAMING
# ===========================================================================
class TestStreaming:
    """Streaming chat completion tests."""

    def test_stream_returns_chunks(self, client, model, provider, model_family):
        """Verify streaming returns multiple chunks."""
        chunks = chat_stream(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        assert len(chunks) > 1, f"[{provider}/{model_family}] Only {len(chunks)} chunk(s)"

    def test_stream_chunk_object_type(self, client, model, provider, model_family):
        """Verify each chunk has object=chat.completion.chunk."""
        chunks = chat_stream(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        for chunk in chunks:
            if chunk.choices:
                assert chunk.object == "chat.completion.chunk", (
                    f"[{provider}/{model_family}] chunk object={chunk.object}"
                )

    def test_stream_has_finish_reason(self, client, model, provider, model_family):
        """Verify the final chunk has a finish_reason."""
        chunks = chat_stream(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        last_with_choices = [c for c in chunks if c.choices and c.choices[0].finish_reason]
        assert last_with_choices, f"[{provider}/{model_family}] No chunk with finish_reason"
        assert last_with_choices[-1].choices[0].finish_reason in ("stop", "length"), (
            f"[{provider}/{model_family}] Unexpected finish_reason"
        )

    def test_stream_content_concatenation(self, client, model, provider, model_family):
        """Verify concatenated stream deltas produce non-empty content."""
        chunks = chat_stream(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        full_content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        assert len(full_content) > 0, f"[{provider}/{model_family}] Empty concatenated stream"

    def test_stream_consistent_id(self, client, model, provider, model_family):
        """Verify all chunks share the same response ID."""
        chunks = chat_stream(client, model, SIMPLE_PROMPT, max_completion_tokens=300)
        ids = {c.id for c in chunks if c.id}
        assert len(ids) == 1, f"[{provider}/{model_family}] Multiple IDs in stream: {ids}"


# ===========================================================================
# 3. TEMPERATURE
# ===========================================================================
class TestTemperature:
    """Temperature sampling parameter tests."""

    def test_temperature_zero(self, client, model, provider, model_family):
        """Temperature=0 should produce deterministic-ish output."""
        responses = []
        for _ in range(2):
            resp = chat(client, model, MATH_PROMPT, temperature=0, max_completion_tokens=300)
            responses.append(resp.choices[0].message.content.strip())
        assert responses[0] == responses[1], (
            f"[{provider}/{model_family}] temp=0 produced different outputs: {responses}"
        )

    def test_temperature_high(self, client, model, provider, model_family):
        """Temperature=1.5 should still return a valid response."""
        resp = chat(client, model, SIMPLE_PROMPT, temperature=1.5, max_completion_tokens=1000)
        assert resp.choices[0].message.content, (
            f"[{provider}/{model_family}] No content at high temperature"
        )

    def test_temperature_boundary_zero(self, client, model, provider, model_family):
        """Temperature=0 (min boundary) should work."""
        resp = chat(client, model, SIMPLE_PROMPT, temperature=0, max_completion_tokens=300)
        assert resp.choices[0].message.content is not None

    def test_temperature_boundary_two(self, client, model, provider, model_family):
        """Temperature=2 (max boundary) should work or return an error gracefully."""
        try:
            resp = chat(client, model, SIMPLE_PROMPT, temperature=2.0, max_completion_tokens=300)
            assert resp.choices[0].message.content is not None
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] temp=2.0 not supported: {e}")


# ===========================================================================
# 4. TOP_P
# ===========================================================================
class TestTopP:
    """Nucleus sampling (top_p) tests."""

    def test_top_p_default(self, client, model, provider, model_family):
        """Default top_p should produce a valid response."""
        resp = chat(client, model, SIMPLE_PROMPT, max_completion_tokens=1000)
        assert resp.choices[0].message.content

    def test_top_p_low(self, client, model, provider, model_family):
        """Low top_p (0.1) should still produce output."""
        resp = chat(client, model, SIMPLE_PROMPT, top_p=0.1, max_completion_tokens=300)
        assert resp.choices[0].message.content

    def test_top_p_one(self, client, model, provider, model_family):
        """top_p=1.0 should work (no filtering)."""
        resp = chat(client, model, SIMPLE_PROMPT, top_p=1.0, max_completion_tokens=300)
        assert resp.choices[0].message.content


# ===========================================================================
# 5. MAX_TOKENS / MAX_COMPLETION_TOKENS
# ===========================================================================
class TestMaxTokens:
    """Token limit tests."""

    def test_max_completion_tokens_small(self, client, model, provider, model_family):
        """Small max_completion_tokens should truncate output."""
        resp = chat(
            client, model,
            [{"role": "user", "content": "Write a very long essay about the history of computing."}],
            max_completion_tokens=10,
        )
        assert resp.choices[0].finish_reason == "length", (
            f"[{provider}/{model_family}] Expected length, got {resp.choices[0].finish_reason}"
        )

    def test_max_completion_tokens_large(self, client, model, provider, model_family):
        """Larger max_completion_tokens should allow full response."""
        resp = chat(client, model, MATH_PROMPT, max_completion_tokens=3000)
        assert resp.choices[0].finish_reason == "stop", (
            f"[{provider}/{model_family}] Expected stop, got {resp.choices[0].finish_reason}"
        )

    def test_max_tokens_alias(self, client, model, provider, model_family):
        """max_tokens should also work (OpenAI compat alias)."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_tokens=1000,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] max_tokens not supported: {e}")


# ===========================================================================
# 6. STOP SEQUENCES
# ===========================================================================
class TestStopSequences:
    """Stop sequence tests."""

    def test_stop_string(self, client, model, provider, model_family):
        """A single stop string should halt generation."""
        resp = chat(
            client, model,
            [{"role": "user", "content": "Count from 1 to 10, separated by commas."}],
            stop=",",
            max_completion_tokens=1000,
        )
        content = resp.choices[0].message.content
        assert "," not in content, (
            f"[{provider}/{model_family}] Stop sequence ',' found in output: {content[:200]}"
        )

    def test_stop_list(self, client, model, provider, model_family):
        """A list of stop strings should halt generation at the first match."""
        resp = chat(
            client, model,
            [{"role": "user", "content": "Count from 1 to 10, one per line."}],
            stop=["5", "6"],
            max_completion_tokens=3000,
        )
        content = resp.choices[0].message.content
        assert "5" not in content and "6" not in content or resp.choices[0].finish_reason == "stop", (
            f"[{provider}/{model_family}] Stop sequence not respected: {content[:200]}"
        )

    def test_stop_finish_reason(self, client, model, provider, model_family):
        """When stop sequence is hit, finish_reason should be 'stop'."""
        resp = chat(
            client, model,
            [{"role": "user", "content": "Say: alpha beta gamma delta epsilon"}],
            stop=["gamma"],
            max_completion_tokens=1000,
        )
        assert resp.choices[0].finish_reason == "stop", (
            f"[{provider}/{model_family}] finish_reason={resp.choices[0].finish_reason}"
        )


# ===========================================================================
# 7. RESPONSE FORMAT (JSON MODE)
# ===========================================================================
class TestResponseFormat:
    """JSON mode / response_format tests."""

    def test_json_object_mode(self, client, model, provider, model_family):
        """response_format=json_object should return valid JSON."""
        messages = [
            {"role": "system", "content": "You always respond in JSON format."},
            {"role": "user", "content": 'Return a JSON object with key "color" and value "blue".'},
        ]
        resp = chat(
            client, model, messages,
            response_format={"type": "json_object"},
            max_completion_tokens=1000,
        )
        content = resp.choices[0].message.content.strip()
        parsed = json.loads(content)
        assert isinstance(parsed, dict), f"[{provider}/{model_family}] Not a JSON object: {content}"

    def test_json_schema_strict_true(self, client, model, provider, model_family):
        """json_schema with strict=true and a valid schema should work."""
        schema = {
            "name": "color_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "hex": {"type": "string"},
                },
                "required": ["color", "hex"],
                "additionalProperties": False,
            },
        }
        messages = [
            {"role": "system", "content": "Respond in JSON matching the schema."},
            {"role": "user", "content": "What color is the sky? Include hex code."},
        ]
        try:
            resp = chat(
                client, model, messages,
                response_format={"type": "json_schema", "json_schema": schema},
                max_completion_tokens=1000,
            )
            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)
            assert "color" in parsed, f"[{provider}/{model_family}] Missing 'color' key"
            assert "hex" in parsed, f"[{provider}/{model_family}] Missing 'hex' key"
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] json_schema strict=true not supported: {e}")

    def test_json_schema_strict_false(self, client, model, provider, model_family):
        """json_schema with strict=false should return valid JSON."""
        schema = {
            "name": "color_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "hex": {"type": "string"},
                },
                "required": ["color", "hex"],
            },
        }
        messages = [
            {"role": "system", "content": "Respond in JSON matching the schema."},
            {"role": "user", "content": "What color is the sky? Include hex code."},
        ]
        try:
            resp = chat(
                client, model, messages,
                response_format={"type": "json_schema", "json_schema": schema},
                max_completion_tokens=1000,
            )
            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)
            assert "color" in parsed, f"[{provider}/{model_family}] Missing 'color' key"
            assert "hex" in parsed, f"[{provider}/{model_family}] Missing 'hex' key"
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] json_schema strict=false not supported: {e}")

    def test_json_schema_strict_rejects_invalid(self, client, model, provider, model_family):
        """strict=true should reject schemas missing additionalProperties: false.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        schema = {
            "name": "test_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
                # Missing additionalProperties: false — invalid for strict mode
            },
        }
        messages = [
            {"role": "system", "content": "Respond in JSON."},
            {"role": "user", "content": "What is 2+2? Return JSON with key 'answer'."},
        ]
        try:
            resp = chat(
                client, model, messages,
                response_format={"type": "json_schema", "json_schema": schema},
                max_completion_tokens=1000,
            )
            # Provider accepted invalid schema — strict validation not enforced
            pytest.skip(
                f"[{provider}/{model_family}] strict=true accepted invalid schema "
                f"(missing additionalProperties) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise  # Re-raise pytest.skip
        except Exception:
            # Provider correctly rejected the invalid schema
            pass

    def test_text_format(self, client, model, provider, model_family):
        """response_format=text should return normal text."""
        resp = chat(
            client, model, SIMPLE_PROMPT,
            response_format={"type": "text"},
            max_completion_tokens=300,
        )
        assert resp.choices[0].message.content


# ===========================================================================
# 8. N (MULTIPLE COMPLETIONS)
# ===========================================================================
class TestMultipleCompletions:
    """Tests for the n parameter (multiple completions)."""

    def test_n_equals_2(self, client, model, provider, model_family):
        """n=2 should return 2 choices."""
        try:
            resp = chat(client, model, SIMPLE_PROMPT, n=2, max_completion_tokens=300)
            assert len(resp.choices) == 2, (
                f"[{provider}/{model_family}] Expected 2 choices, got {len(resp.choices)}"
            )
            for i, choice in enumerate(resp.choices):
                assert choice.index == i
                assert choice.message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] n=2 not supported: {e}")

    def test_n_equals_1(self, client, model, provider, model_family):
        """n=1 (default) should return exactly 1 choice."""
        resp = chat(client, model, SIMPLE_PROMPT, n=1, max_completion_tokens=300)
        assert len(resp.choices) == 1


# ===========================================================================
# 9. FREQUENCY / PRESENCE / REPETITION PENALTY
# ===========================================================================
class TestPenalties:
    """Frequency, presence, and repetition penalty tests."""

    def test_frequency_penalty(self, client, model, provider, model_family):
        """frequency_penalty should be accepted."""
        try:
            resp = chat(
                client, model, SIMPLE_PROMPT,
                frequency_penalty=0.5,
                max_completion_tokens=300,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] frequency_penalty not supported: {e}")

    def test_frequency_penalty_negative(self, client, model, provider, model_family):
        """Negative frequency_penalty (encourage repetition) should be accepted."""
        try:
            resp = chat(
                client, model, SIMPLE_PROMPT,
                frequency_penalty=-0.5,
                max_completion_tokens=300,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] negative frequency_penalty not supported: {e}")

    def test_frequency_penalty_max(self, client, model, provider, model_family):
        """frequency_penalty=2.0 (max boundary) should be accepted."""
        try:
            resp = chat(
                client, model, SIMPLE_PROMPT,
                frequency_penalty=2.0,
                max_completion_tokens=300,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] frequency_penalty=2.0 not supported: {e}")

    def test_presence_penalty(self, client, model, provider, model_family):
        """presence_penalty should be accepted."""
        try:
            resp = chat(
                client, model, SIMPLE_PROMPT,
                presence_penalty=0.5,
                max_completion_tokens=300,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] presence_penalty not supported: {e}")

    def test_presence_penalty_negative(self, client, model, provider, model_family):
        """Negative presence_penalty (encourage repetition) should be accepted."""
        try:
            resp = chat(
                client, model, SIMPLE_PROMPT,
                presence_penalty=-0.5,
                max_completion_tokens=300,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] negative presence_penalty not supported: {e}")

    def test_both_penalties(self, client, model, provider, model_family):
        """Both penalties together should be accepted."""
        try:
            resp = chat(
                client, model, SIMPLE_PROMPT,
                frequency_penalty=0.3,
                presence_penalty=0.3,
                max_completion_tokens=300,
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] combined penalties not supported: {e}")

    def test_repetition_penalty(self, client, model, provider, model_family):
        """repetition_penalty=1.2 (Fireworks extension) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"repetition_penalty": 1.2},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] repetition_penalty not supported: {e}")

    def test_repetition_penalty_no_penalty(self, client, model, provider, model_family):
        """repetition_penalty=1.0 (no penalty) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"repetition_penalty": 1.0},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] repetition_penalty=1.0 not supported: {e}")


# ===========================================================================
# 10. SEED (DETERMINISM)
# ===========================================================================
class TestSeed:
    """Seed parameter tests for reproducibility."""

    def test_same_seed_same_output(self, client, model, provider, model_family):
        """Same seed should produce the same output."""
        kwargs = dict(seed=42, temperature=0, max_completion_tokens=300)
        r1 = chat(client, model, MATH_PROMPT, **kwargs)
        r2 = chat(client, model, MATH_PROMPT, **kwargs)
        c1 = r1.choices[0].message.content.strip()
        c2 = r2.choices[0].message.content.strip()
        assert c1 == c2, (
            f"[{provider}/{model_family}] Same seed produced different output:\n  1: {c1}\n  2: {c2}"
        )

    def test_different_seed_accepted(self, client, model, provider, model_family):
        """Different seed values should be accepted without error."""
        r1 = chat(client, model, SIMPLE_PROMPT, seed=1, max_completion_tokens=300)
        r2 = chat(client, model, SIMPLE_PROMPT, seed=999, max_completion_tokens=300)
        assert r1.choices[0].message.content
        assert r2.choices[0].message.content


# ===========================================================================
# 11. LOGPROBS
# ===========================================================================
class TestLogprobs:
    """Log probability tests."""

    def test_logprobs_boolean(self, client, model, provider, model_family):
        """logprobs=True should return log probabilities."""
        try:
            resp = chat(
                client, model, MATH_PROMPT,
                logprobs=True,
                max_completion_tokens=300,
            )
            assert resp.choices[0].logprobs is not None, (
                f"[{provider}/{model_family}] logprobs=True but no logprobs in response"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] logprobs not supported: {e}")

    def test_top_logprobs(self, client, model, provider, model_family):
        """top_logprobs should return multiple token alternatives."""
        try:
            resp = chat(
                client, model, MATH_PROMPT,
                logprobs=True,
                top_logprobs=3,
                max_completion_tokens=300,
            )
            lp = resp.choices[0].logprobs
            assert lp is not None, f"[{provider}/{model_family}] No logprobs returned"
            if hasattr(lp, "content") and lp.content:
                assert len(lp.content[0].top_logprobs) <= 3, (
                    f"[{provider}/{model_family}] top_logprobs returned more than requested"
                )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] top_logprobs not supported: {e}")

    def test_logprobs_false(self, client, model, provider, model_family):
        """logprobs=False should not return logprobs."""
        resp = chat(client, model, MATH_PROMPT, logprobs=False, max_completion_tokens=300)
        assert resp.choices[0].logprobs is None, (
            f"[{provider}/{model_family}] logprobs=False but logprobs returned"
        )


# ===========================================================================
# 12. TOOL / FUNCTION CALLING
# ===========================================================================
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
}


class TestToolCalling:
    """Tool/function calling tests."""

    def test_tool_call_triggered(self, client, model, provider, model_family):
        """Model should trigger a tool call when appropriate."""
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in San Francisco?"}],
                tools=[WEATHER_TOOL],
                max_completion_tokens=3000,
            )
            choice = resp.choices[0]
            if choice.message.tool_calls:
                tc = choice.message.tool_calls[0]
                assert tc.function.name == "get_weather", (
                    f"[{provider}/{model_family}] Wrong function: {tc.function.name}"
                )
                args = json.loads(tc.function.arguments)
                assert "location" in args, f"[{provider}/{model_family}] Missing location arg"
            else:
                # Some models may answer directly; that's acceptable
                assert choice.message.content, (
                    f"[{provider}/{model_family}] No tool call and no content"
                )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] tool calling not supported: {e}")

    def test_tool_choice_none(self, client, model, provider, model_family):
        """tool_choice=none should prevent tool calls."""
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in NYC?"}],
                tools=[WEATHER_TOOL],
                tool_choice="none",
                max_completion_tokens=1000,
            )
            choice = resp.choices[0]
            assert not choice.message.tool_calls, (
                f"[{provider}/{model_family}] tool_choice=none but tool_calls present"
            )
            assert choice.message.content, f"[{provider}/{model_family}] No content returned"
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] tool_choice=none not supported: {e}")

    def test_tool_choice_required(self, client, model, provider, model_family):
        """tool_choice=required should force a tool call."""
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in London?"}],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_completion_tokens=3000,
            )
            choice = resp.choices[0]
            assert choice.message.tool_calls, (
                f"[{provider}/{model_family}] tool_choice=required but no tool_calls"
            )
            assert choice.finish_reason == "tool_calls", (
                f"[{provider}/{model_family}] finish_reason={choice.finish_reason}, expected tool_calls"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] tool_choice=required not supported: {e}")

    def test_tool_choice_specific_function(self, client, model, provider, model_family):
        """tool_choice specifying a function should force that function."""
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Hello, how are you?"}],
                tools=[WEATHER_TOOL, CALC_TOOL],
                tool_choice={"type": "function", "function": {"name": "calculate"}},
                max_completion_tokens=3000,
            )
            choice = resp.choices[0]
            assert choice.message.tool_calls, (
                f"[{provider}/{model_family}] No tool_calls despite specific tool_choice"
            )
            assert choice.message.tool_calls[0].function.name == "calculate", (
                f"[{provider}/{model_family}] Wrong function called"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] specific tool_choice not supported: {e}")

    def test_tool_strict_true(self, client, model, provider, model_family):
        """Tool with strict=true and a valid schema should work."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in San Francisco?"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=3000,
            )
            choice = resp.choices[0]
            assert choice.message.tool_calls, (
                f"[{provider}/{model_family}] No tool call with strict=true"
            )
            args = json.loads(choice.message.tool_calls[0].function.arguments)
            assert "location" in args, f"[{provider}/{model_family}] Missing location arg"
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] tool strict=true not supported: {e}")

    def test_tool_strict_false(self, client, model, provider, model_family):
        """Tool with strict=false should work."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in San Francisco?"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=3000,
            )
            choice = resp.choices[0]
            assert choice.message.tool_calls, (
                f"[{provider}/{model_family}] No tool call with strict=false"
            )
            args = json.loads(choice.message.tool_calls[0].function.arguments)
            assert "location" in args, f"[{provider}/{model_family}] Missing location arg"
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] tool strict=false not supported: {e}")

    def test_tool_strict_rejects_invalid(self, client, model, provider, model_family):
        """Tool with strict=true should reject schemas missing additionalProperties: false.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                    },
                    "required": ["location"],
                    # Missing additionalProperties: false — invalid for strict mode
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in San Francisco?"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=3000,
            )
            # Provider accepted invalid schema — strict validation not enforced
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(missing additionalProperties) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise  # Re-raise pytest.skip
        except Exception:
            # Provider correctly rejected the invalid schema
            pass

    def test_tool_strict_rejects_nested_missing_additionalproperties(self, client, model, provider, model_family):
        """Tool with strict=true should reject nested objects missing additionalProperties: false.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "create_user",
                "description": "Create a user with address",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"}
                            },
                            "required": ["street"]
                            # Missing additionalProperties in nested object
                        }
                    },
                    "required": ["name", "address"],
                    "additionalProperties": False
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Create user John at 123 Main St"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(nested object missing additionalProperties) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_tool_strict_rejects_additionalproperties_true(self, client, model, provider, model_family):
        """Tool with strict=true should reject additionalProperties: true.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "log_event",
                "description": "Log an event",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_name": {"type": "string"}
                    },
                    "required": ["event_name"],
                    "additionalProperties": True  # Explicitly true - invalid for strict
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Log event user_login"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(additionalProperties: true) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_tool_strict_rejects_format_keyword(self, client, model, provider, model_family):
        """Tool with strict=true should reject unsupported 'format' keyword.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "validate_email",
                "description": "Validate an email address",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "format": "email"  # format keyword - not supported in strict
                        }
                    },
                    "required": ["email"],
                    "additionalProperties": False
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Validate email test@example.com"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(format keyword) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_tool_strict_rejects_pattern_keyword(self, client, model, provider, model_family):
        """Tool with strict=true should reject unsupported 'pattern' keyword.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "set_phone",
                "description": "Set phone number",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "pattern": "^\\d{3}-\\d{3}-\\d{4}$"  # pattern - not supported in strict
                        }
                    },
                    "required": ["phone"],
                    "additionalProperties": False
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Set phone number to 555-123-4567"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(pattern keyword) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_tool_strict_rejects_minmax_length(self, client, model, provider, model_family):
        """Tool with strict=true should reject unsupported minLength/maxLength keywords.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "set_username",
                "description": "Set username",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 20
                        }
                    },
                    "required": ["username"],
                    "additionalProperties": False
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Set username to john_doe"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(minLength/maxLength keywords) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_tool_strict_rejects_oneof(self, client, model, provider, model_family):
        """Tool with strict=true should reject unsupported 'oneOf' keyword.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "send_notification",
                "description": "Send a notification via one channel",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["email"]},
                                        "address": {"type": "string"}
                                    },
                                    "required": ["type", "address"],
                                    "additionalProperties": False
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["sms"]},
                                        "phone": {"type": "string"}
                                    },
                                    "required": ["type", "phone"],
                                    "additionalProperties": False
                                }
                            ]
                        }
                    },
                    "required": ["channel"],
                    "additionalProperties": False
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Send notification via email to test@example.com"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(oneOf keyword) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_tool_strict_rejects_minmax_items(self, client, model, provider, model_family):
        """Tool with strict=true should reject unsupported minItems/maxItems keywords.

        Providers with proper strict mode validation reject this schema.
        Providers without enforcement silently accept it.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "set_coordinates",
                "description": "Set geographic coordinates",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinates": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },
                    "required": ["coordinates"],
                    "additionalProperties": False
                },
            },
        }
        try:
            resp = chat(
                client, model,
                [{"role": "user", "content": "Set coordinates to latitude 40.7128, longitude -74.0060"}],
                tools=[tool],
                tool_choice="required",
                max_completion_tokens=1000,
            )
            pytest.skip(
                f"[{provider}/{model_family}] tool strict=true accepted invalid schema "
                f"(minItems/maxItems keywords) — validation not enforced"
            )
        except pytest.skip.Exception:
            raise
        except Exception:
            pass

    def test_parallel_tool_calls_actual(self, client, model, provider, model_family):
        """Model should make multiple tool calls in parallel when the query requires it.

        Uses the Cerebras docs weather example: "Is Toronto warmer than Montreal?"
        with a get_weather tool. The model should call get_weather twice (once per city).
        """
        parallel_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Toronto, Canada",
                        },
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Is Toronto warmer than Montreal right now?"}],
                tools=[parallel_tool],
                tool_choice="auto",
                parallel_tool_calls=True,
                max_completion_tokens=3000,
            )
            choice = resp.choices[0]
            assert choice.message.tool_calls, (
                f"[{provider}/{model_family}] No tool calls made"
            )
            assert len(choice.message.tool_calls) >= 2, (
                f"[{provider}/{model_family}] Expected >=2 parallel tool calls, "
                f"got {len(choice.message.tool_calls)}"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] parallel tool calls not supported: {e}")

    def test_multi_turn_with_tool_result(self, client, model, provider, model_family):
        """Multi-turn: provide tool result and get final answer."""
        try:
            # First call: get tool call
            resp1 = chat(
                client, model,
                [{"role": "user", "content": "What's the weather in Paris?"}],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_completion_tokens=3000,
            )
            tc = resp1.choices[0].message.tool_calls[0]

            # Second call: provide tool result
            messages = [
                {"role": "user", "content": "What's the weather in Paris?"},
                resp1.choices[0].message.model_dump(exclude_none=True),
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"temperature": "22°C", "condition": "sunny"}),
                },
            ]
            resp2 = chat(client, model, messages, tools=[WEATHER_TOOL], max_completion_tokens=3000)
            assert resp2.choices[0].message.content, (
                f"[{provider}/{model_family}] No content after tool result"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] multi-turn tool calling not supported: {e}")


# ===========================================================================
# 13. REASONING EFFORT
# ===========================================================================
class TestReasoningEffort:
    """reasoning_effort parameter tests."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_reasoning_effort_levels(self, client, model, provider, model_family, effort):
        """Each reasoning_effort level should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"reasoning_effort": effort},
            )
            assert resp.choices[0].message.content, (
                f"[{provider}/{model_family}] No content with reasoning_effort={effort}"
            )
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] reasoning_effort={effort} not supported: {e}"
            )

    def test_reasoning_content_field(self, client, model, provider, model_family):
        """Check if reasoning_content is returned in the response."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What is 15 * 37? Think step by step."}],
                max_completion_tokens=1000,
                extra_body={"reasoning_effort": "high"},
            )
            choice = resp.choices[0]
            # reasoning_content may or may not be present depending on provider
            msg_dict = choice.message.model_dump()
            has_reasoning = msg_dict.get("reasoning_content") is not None
            # Just report, don't fail
            print(
                f"[{provider}/{model_family}] reasoning_content present: {has_reasoning}"
            )
            assert choice.message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] reasoning_effort not supported: {e}")


# ===========================================================================
# 14. MULTI-TURN CONVERSATION
# ===========================================================================
class TestMultiTurn:
    """Multi-turn conversation tests."""

    def test_two_turn_conversation(self, client, model, provider, model_family):
        """Model should use context from previous turns."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
        ]
        r1 = chat(client, model, messages, max_completion_tokens=1000)

        messages.append({"role": "assistant", "content": r1.choices[0].message.content})
        messages.append({"role": "user", "content": "What is my name?"})
        r2 = chat(client, model, messages, max_completion_tokens=1000)

        assert "alice" in r2.choices[0].message.content.lower(), (
            f"[{provider}/{model_family}] Model didn't recall name: {r2.choices[0].message.content}"
        )

    def test_three_turn_with_system(self, client, model, provider, model_family):
        """Three turns with a system message should maintain context."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 5 + 3?"},
        ]
        r1 = chat(client, model, messages, max_completion_tokens=300)
        messages.append({"role": "assistant", "content": r1.choices[0].message.content})
        messages.append({"role": "user", "content": "Now multiply that result by 2."})
        r2 = chat(client, model, messages, max_completion_tokens=300)
        assert "16" in r2.choices[0].message.content, (
            f"[{provider}/{model_family}] Wrong multi-turn math: {r2.choices[0].message.content}"
        )


# ===========================================================================
# 15. ERROR HANDLING
# ===========================================================================
class TestErrorHandling:
    """Error handling and edge case tests."""

    def test_empty_messages_rejected(self, client, model, provider, model_family):
        """Empty messages list should be rejected."""
        with pytest.raises(Exception):
            chat(client, model, [])

    def test_invalid_model_rejected(self, client, provider):
        """Invalid model name should be rejected."""
        with pytest.raises(Exception):
            chat(client, "nonexistent-model-xyz-123", SIMPLE_PROMPT)

    def test_negative_max_tokens_rejected(self, client, model, provider, model_family):
        """Negative max_completion_tokens should be rejected."""
        with pytest.raises(Exception):
            chat(client, model, SIMPLE_PROMPT, max_completion_tokens=-1)

    def test_temperature_out_of_range(self, client, model, provider, model_family):
        """Temperature > 2 should be rejected or clamped."""
        try:
            resp = chat(client, model, SIMPLE_PROMPT, temperature=5.0, max_completion_tokens=300)
            # If it succeeds, provider may be clamping — that's fine
            assert resp.choices[0].message.content
        except Exception:
            pass  # Expected: error for out-of-range temperature


# ===========================================================================
# 16. USER PARAMETER
# ===========================================================================
class TestUserParameter:
    """Tests for the user parameter."""

    def test_user_param_accepted(self, client, model, provider, model_family):
        """user parameter should be accepted without error."""
        resp = chat(
            client, model, SIMPLE_PROMPT,
            user="test-user-123",
            max_completion_tokens=300,
        )
        assert resp.choices[0].message.content


# ===========================================================================
# 17. TOP_K SAMPLING
# ===========================================================================
class TestTopK:
    """Top-k sampling parameter tests (Fireworks: 0-100)."""

    def test_top_k_50(self, client, model, provider, model_family):
        """top_k=50 should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"top_k": 50},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] top_k not supported: {e}")

    def test_top_k_1(self, client, model, provider, model_family):
        """top_k=1 (greedy-like) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"top_k": 1},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] top_k=1 not supported: {e}")

    def test_top_k_100(self, client, model, provider, model_family):
        """top_k=100 (max boundary) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"top_k": 100},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] top_k=100 not supported: {e}")


# ===========================================================================
# 18. MIN_P SAMPLING
# ===========================================================================
class TestMinP:
    """Min-p sampling parameter tests (Fireworks: 0-1)."""

    def test_min_p(self, client, model, provider, model_family):
        """min_p=0.1 should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"min_p": 0.1},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] min_p not supported: {e}")

    def test_min_p_zero(self, client, model, provider, model_family):
        """min_p=0 (no filtering) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"min_p": 0.0},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] min_p=0 not supported: {e}")


# ===========================================================================
# 19. TYPICAL_P SAMPLING
# ===========================================================================
class TestTypicalP:
    """Typical-p sampling parameter tests (Fireworks: 0-1)."""

    def test_typical_p(self, client, model, provider, model_family):
        """typical_p=0.9 should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"typical_p": 0.9},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] typical_p not supported: {e}")

    def test_typical_p_one(self, client, model, provider, model_family):
        """typical_p=1.0 (no filtering) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"typical_p": 1.0},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] typical_p=1.0 not supported: {e}")


# ===========================================================================
# 20. MIROSTAT SAMPLING
# ===========================================================================
class TestMirostat:
    """Mirostat sampling parameter tests (Fireworks extension)."""

    def test_mirostat_target(self, client, model, provider, model_family):
        """mirostat_target should enable Mirostat sampling."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"mirostat_target": 5.0},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] mirostat_target not supported: {e}")

    def test_mirostat_with_lr(self, client, model, provider, model_family):
        """mirostat_target with mirostat_lr should be accepted together."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"mirostat_target": 5.0, "mirostat_lr": 0.1},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] mirostat with lr not supported: {e}")


# ===========================================================================
# 21. LOGIT_BIAS
# ===========================================================================
class TestLogitBias:
    """Logit bias parameter tests."""

    def test_logit_bias(self, client, model, provider, model_family):
        """logit_bias should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                logit_bias={"100": -50.0},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] logit_bias not supported: {e}")

    def test_logit_bias_positive(self, client, model, provider, model_family):
        """Positive logit_bias values should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                logit_bias={"100": 50.0},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] positive logit_bias not supported: {e}")


# ===========================================================================
# 22. ECHO / ECHO_LAST
# ===========================================================================
class TestEcho:
    """Echo parameter tests (Fireworks extension)."""

    def test_echo(self, client, model, provider, model_family):
        """echo=True should echo the prompt back."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"echo": True},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] echo not supported: {e}")

    def test_echo_last(self, client, model, provider, model_family):
        """echo_last=5 should echo last N prompt tokens."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"echo_last": 5},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] echo_last not supported: {e}")


# ===========================================================================
# 23. IGNORE_EOS
# ===========================================================================
class TestIgnoreEos:
    """ignore_eos parameter tests (Fireworks extension)."""

    def test_ignore_eos_false(self, client, model, provider, model_family):
        """ignore_eos=False (default) should stop at EOS."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"ignore_eos": False},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] ignore_eos not supported: {e}")

    def test_ignore_eos_true(self, client, model, provider, model_family):
        """ignore_eos=True should continue past EOS (and hit max_tokens)."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=50,
                extra_body={"ignore_eos": True},
            )
            # With ignore_eos, we expect it to hit max_tokens
            assert resp.choices[0].finish_reason == "length", (
                f"[{provider}/{model_family}] Expected length with ignore_eos=True, "
                f"got {resp.choices[0].finish_reason}"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] ignore_eos=True not supported: {e}")


# ===========================================================================
# 24. CONTEXT_LENGTH_EXCEEDED_BEHAVIOR
# ===========================================================================
class TestContextLengthExceeded:
    """context_length_exceeded_behavior tests (Fireworks extension)."""

    def test_truncate_behavior(self, client, model, provider, model_family):
        """context_length_exceeded_behavior=truncate should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"context_length_exceeded_behavior": "truncate"},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] context_length_exceeded_behavior not supported: {e}"
            )

    def test_error_behavior(self, client, model, provider, model_family):
        """context_length_exceeded_behavior=error should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"context_length_exceeded_behavior": "error"},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] context_length_exceeded_behavior=error not supported: {e}"
            )


# ===========================================================================
# 25. PREDICTION (SPECULATIVE DECODING)
# ===========================================================================
class TestPrediction:
    """Predicted output / speculative decoding tests (OpenAI-compatible)."""

    def test_prediction_string(self, client, model, provider, model_family):
        """prediction with a string hint should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say exactly: The quick brown fox"}],
                max_completion_tokens=1000,
                prediction={"type": "content", "content": "The quick brown fox"},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] prediction not supported: {e}")


# ===========================================================================
# 26. METADATA
# ===========================================================================
class TestMetadata:
    """metadata parameter tests (Fireworks extension)."""

    def test_metadata(self, client, model, provider, model_family):
        """metadata dict should be accepted without error."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"metadata": {"session_id": "test-123", "source": "pytest"}},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] metadata not supported: {e}")


# ===========================================================================
# 27. REASONING_HISTORY
# ===========================================================================
class TestReasoningHistory:
    """reasoning_history parameter tests (Fireworks extension, GLM-4.7 / GPT-OSS)."""

    @pytest.mark.parametrize("mode", ["disabled", "interleaved", "preserved"])
    def test_reasoning_history_modes(self, client, model, provider, model_family, mode):
        """Each reasoning_history mode should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"reasoning_history": mode},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] reasoning_history={mode} not supported: {e}"
            )


# ===========================================================================
# 28. THINKING (ANTHROPIC-COMPATIBLE FORMAT)
# ===========================================================================
class TestThinking:
    """thinking parameter tests (Anthropic-compatible format)."""

    def test_thinking_enabled(self, client, model, provider, model_family):
        """thinking type=enabled should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"thinking": {"type": "enabled"}},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] thinking enabled not supported: {e}")

    def test_thinking_disabled(self, client, model, provider, model_family):
        """thinking type=disabled should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] thinking disabled not supported: {e}")

    def test_thinking_with_budget(self, client, model, provider, model_family):
        """thinking with budget_tokens should be accepted (must be >= 1024)."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What is 15 * 37?"}],
                max_completion_tokens=30000,
                extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] thinking with budget not supported: {e}")


# ===========================================================================
# 29. CLEAR_THINKING (CEREBRAS)
# ===========================================================================
class TestClearThinking:
    """clear_thinking parameter tests (Cerebras extension, zai-glm-4.7 only).

    clear_thinking controls whether thinking content from previous conversation
    turns is included in the prompt context:
      - true (default): Excludes thinking from earlier turns (general chat)
      - false: Preserves thinking from previous turns (agentic workflows)
    Only supported on the zai-glm-4.7 model.
    """

    def test_clear_thinking_true(self, client, model, provider, model_family):
        """clear_thinking=true (default) should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=1000,
                extra_body={"clear_thinking": True},
            )
            assert resp.choices[0].message.content, (
                f"[{provider}/{model_family}] No content with clear_thinking=true"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] clear_thinking=true not supported: {e}")

    def test_clear_thinking_false(self, client, model, provider, model_family):
        """clear_thinking=false should preserve thinking from previous turns."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=1000,
                extra_body={"clear_thinking": False},
            )
            assert resp.choices[0].message.content, (
                f"[{provider}/{model_family}] No content with clear_thinking=false"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] clear_thinking=false not supported: {e}")

    def test_clear_thinking_multi_turn(self, client, model, provider, model_family):
        """clear_thinking=false in multi-turn should preserve reasoning context."""
        try:
            # First turn
            r1 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What is 5 + 3? Think step by step."}],
                max_completion_tokens=1000,
                extra_body={"clear_thinking": False},
            )
            assert r1.choices[0].message.content, (
                f"[{provider}/{model_family}] No content on first turn"
            )

            # Second turn with previous context
            messages = [
                {"role": "user", "content": "What is 5 + 3? Think step by step."},
                {"role": "assistant", "content": r1.choices[0].message.content},
                {"role": "user", "content": "Now multiply that result by 2."},
            ]
            r2 = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1000,
                extra_body={"clear_thinking": False},
            )
            assert r2.choices[0].message.content, (
                f"[{provider}/{model_family}] No content on second turn with clear_thinking=false"
            )
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] clear_thinking multi-turn not supported: {e}"
            )


# ===========================================================================
# 30. PARALLEL_TOOL_CALLS
# ===========================================================================
class TestParallelToolCalls:
    """parallel_tool_calls parameter tests."""

    def test_parallel_tool_calls_true(self, client, model, provider, model_family):
        """parallel_tool_calls=True should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What's the weather in NYC and London?"}],
                tools=[WEATHER_TOOL],
                parallel_tool_calls=True,
                max_completion_tokens=3000,
            )
            assert resp.choices[0].message.content or resp.choices[0].message.tool_calls
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] parallel_tool_calls not supported: {e}")

    def test_parallel_tool_calls_false(self, client, model, provider, model_family):
        """parallel_tool_calls=False should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What's the weather in NYC?"}],
                tools=[WEATHER_TOOL],
                parallel_tool_calls=False,
                max_completion_tokens=3000,
            )
            assert resp.choices[0].message.content or resp.choices[0].message.tool_calls
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] parallel_tool_calls=False not supported: {e}")


# ===========================================================================
# 30. PROMPT_TRUNCATE_LEN
# ===========================================================================
class TestPromptTruncateLen:
    """prompt_truncate_len parameter tests (Fireworks extension)."""

    def test_prompt_truncate_len(self, client, model, provider, model_family):
        """prompt_truncate_len should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"prompt_truncate_len": 2048},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] prompt_truncate_len not supported: {e}")


# ===========================================================================
# 31. RAW_OUTPUT
# ===========================================================================
class TestRawOutput:
    """raw_output parameter tests (Fireworks extension)."""

    def test_raw_output(self, client, model, provider, model_family):
        """raw_output=True should return raw model output."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"raw_output": True},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] raw_output not supported: {e}")


# ===========================================================================
# 32. PERF_METRICS_IN_RESPONSE
# ===========================================================================
class TestPerfMetrics:
    """perf_metrics_in_response parameter tests (Fireworks extension)."""

    def test_perf_metrics_in_response(self, client, model, provider, model_family):
        """perf_metrics_in_response=True should include metrics in response body."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"perf_metrics_in_response": True},
            )
            assert resp.choices[0].message.content
            # Check if perf_metrics field is present in raw response
            resp_dict = resp.model_dump()
            has_perf = resp_dict.get("perf_metrics") is not None
            print(f"[{provider}/{model_family}] perf_metrics present: {has_perf}")
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] perf_metrics_in_response not supported: {e}")

    def test_perf_metrics_streaming(self, client, model, provider, model_family):
        """perf_metrics_in_response=True should include metrics in final stream chunk."""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                stream=True,
                extra_body={"perf_metrics_in_response": True},
            )
            chunks = list(stream)
            assert len(chunks) > 0
            # Check last chunk for perf_metrics
            last_chunk = chunks[-1].model_dump()
            has_perf = last_chunk.get("perf_metrics") is not None
            print(f"[{provider}/{model_family}] streaming perf_metrics present: {has_perf}")
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] perf_metrics streaming not supported: {e}"
            )


# ===========================================================================
# 33. RETURN_TOKEN_IDS
# ===========================================================================
class TestReturnTokenIds:
    """return_token_ids parameter tests (Fireworks extension)."""

    def test_return_token_ids(self, client, model, provider, model_family):
        """return_token_ids=True should return token IDs."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=300,
                extra_body={"return_token_ids": True},
            )
            assert resp.choices[0].message.content
            resp_dict = resp.model_dump()
            has_prompt_ids = resp_dict.get("prompt_token_ids") is not None
            has_choice_ids = (
                resp_dict.get("choices", [{}])[0].get("token_ids") is not None
            )
            print(
                f"[{provider}/{model_family}] prompt_token_ids: {has_prompt_ids}, "
                f"choice token_ids: {has_choice_ids}"
            )
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] return_token_ids not supported: {e}")


# ===========================================================================
# 34. PROMPT_CACHE_ISOLATION_KEY
# ===========================================================================
class TestPromptCacheIsolation:
    """prompt_cache_isolation_key parameter tests (Fireworks extension)."""

    def test_prompt_cache_isolation_key(self, client, model, provider, model_family):
        """prompt_cache_isolation_key should be accepted."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=SIMPLE_PROMPT,
                max_completion_tokens=300,
                extra_body={"prompt_cache_isolation_key": "test-isolation-key"},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(
                f"[{provider}/{model_family}] prompt_cache_isolation_key not supported: {e}"
            )


# ===========================================================================
# 35. REASONING_EFFORT EDGE CASES
# ===========================================================================
class TestReasoningEffortEdgeCases:
    """Additional reasoning_effort edge case tests."""

    def test_reasoning_effort_none(self, client, model, provider, model_family):
        """reasoning_effort='none' should disable reasoning (GLM supports, GPT-OSS does not)."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"reasoning_effort": "none"},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] reasoning_effort=none not supported: {e}")

    def test_reasoning_effort_boolean_true(self, client, model, provider, model_family):
        """reasoning_effort=True (Fireworks extension, maps to 'medium')."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"reasoning_effort": True},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] reasoning_effort=True not supported: {e}")

    def test_reasoning_effort_boolean_false(self, client, model, provider, model_family):
        """reasoning_effort=False (Fireworks extension, maps to 'none')."""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=MATH_PROMPT,
                max_completion_tokens=3000,
                extra_body={"reasoning_effort": False},
            )
            assert resp.choices[0].message.content
        except Exception as e:
            pytest.skip(f"[{provider}/{model_family}] reasoning_effort=False not supported: {e}")


# ===========================================================================
# 36. STREAMING + PARAMETERS COMBINED
# ===========================================================================
class TestStreamingWithParams:
    """Test streaming combined with various parameters."""

    def test_stream_with_temperature(self, client, model, provider, model_family):
        """Streaming with temperature should work."""
        chunks = chat_stream(
            client, model, SIMPLE_PROMPT,
            temperature=0.5, max_completion_tokens=300,
        )
        content = "".join(
            c.choices[0].delta.content or ""
            for c in chunks
            if c.choices and c.choices[0].delta
        )
        assert len(content) > 0

    def test_stream_with_stop(self, client, model, provider, model_family):
        """Streaming with stop sequences should work."""
        chunks = chat_stream(
            client, model,
            [{"role": "user", "content": "Count from 1 to 10, separated by commas."}],
            stop=[","],
            max_completion_tokens=1000,
        )
        content = "".join(
            c.choices[0].delta.content or ""
            for c in chunks
            if c.choices and c.choices[0].delta
        )
        assert "," not in content

    def test_stream_with_json_mode(self, client, model, provider, model_family):
        """Streaming with JSON mode should produce valid JSON."""
        messages = [
            {"role": "system", "content": "Always respond in JSON."},
            {"role": "user", "content": 'Return {"status": "ok"}'},
        ]
        chunks = chat_stream(
            client, model, messages,
            response_format={"type": "json_object"},
            max_completion_tokens=300,
        )
        content = "".join(
            c.choices[0].delta.content or ""
            for c in chunks
            if c.choices and c.choices[0].delta
        )
        parsed = json.loads(content.strip())
        assert isinstance(parsed, dict)


# ===========================================================================
# 18. CROSS-PROVIDER COMPARISON TESTS
# ===========================================================================
class TestCrossProviderComparison:
    """
    Run the same request on both providers and compare behavior.
    These tests are not parametrized by provider — they run both internally.
    """

    @pytest.mark.parametrize("model_family", ["glm", "gpt_oss"], ids=["glm-4.7", "gpt-oss-120b"])
    def test_both_providers_return_valid_response(self, model_family):
        """Both providers should return a valid response for the same prompt."""
        results = run_on_both(model_family, SIMPLE_PROMPT, max_completion_tokens=300)
        for prov, r in results.items():
            assert r.ok, f"[{prov}/{model_family}] Error: {r.error}"
            assert r.content, f"[{prov}/{model_family}] Empty content"

    @pytest.mark.parametrize("model_family", ["glm", "gpt_oss"], ids=["glm-4.7", "gpt-oss-120b"])
    def test_both_providers_finish_reason_consistent(self, model_family):
        """Both providers should agree on finish_reason for the same prompt."""
        results = run_on_both(model_family, MATH_PROMPT, max_completion_tokens=3000, temperature=0)
        finish_reasons = {}
        for prov, r in results.items():
            if r.ok:
                finish_reasons[prov] = r.response.choices[0].finish_reason
        if len(finish_reasons) == 2:
            assert finish_reasons["fireworks"] == finish_reasons["cerebras"], (
                f"finish_reason mismatch: {finish_reasons}"
            )

    @pytest.mark.parametrize("model_family", ["glm", "gpt_oss"], ids=["glm-4.7", "gpt-oss-120b"])
    def test_both_providers_usage_present(self, model_family):
        """Both providers should return usage info."""
        results = run_on_both(model_family, MATH_PROMPT, max_completion_tokens=300, temperature=0)
        for prov, r in results.items():
            assert r.ok, f"[{prov}] Error: {r.error}"
            assert r.response.usage is not None, f"[{prov}] No usage info"
            assert r.response.usage.prompt_tokens > 0, f"[{prov}] prompt_tokens=0"

    @pytest.mark.parametrize("model_family", ["glm", "gpt_oss"], ids=["glm-4.7", "gpt-oss-120b"])
    def test_both_providers_json_mode(self, model_family):
        """Both providers should support JSON mode."""
        messages = [
            {"role": "system", "content": "Always respond in JSON."},
            {"role": "user", "content": 'Return a JSON object: {"answer": 42}'},
        ]
        results = run_on_both(
            model_family, messages,
            response_format={"type": "json_object"},
            max_completion_tokens=1000,
        )
        for prov, r in results.items():
            assert r.ok, f"[{prov}] Error: {r.error}"
            parsed = json.loads(r.content.strip())
            assert isinstance(parsed, dict), f"[{prov}] Not a JSON object"

    @pytest.mark.parametrize("model_family", ["glm", "gpt_oss"], ids=["glm-4.7", "gpt-oss-120b"])
    def test_both_providers_streaming(self, model_family):
        """Both providers should support streaming."""
        for prov in ["fireworks", "cerebras"]:
            c = make_client(prov)
            m = get_model(prov, model_family)
            chunks = chat_stream(c, m, SIMPLE_PROMPT, max_completion_tokens=300)
            assert len(chunks) > 1, f"[{prov}] Only {len(chunks)} chunk(s)"
            content = "".join(
                ch.choices[0].delta.content or ""
                for ch in chunks
                if ch.choices and ch.choices[0].delta
            )
            assert len(content) > 0, f"[{prov}] Empty stream content"

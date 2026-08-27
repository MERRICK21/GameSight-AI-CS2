"""Tests for DeepSeek JSON parsing without network access."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gamesight.llm.client import DeepSeekClient, LLMClientError


class FakeCompletions:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=self.finish_reason,
                message=SimpleNamespace(content=self.content),
            )],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ),
        )


def _client(completions: FakeCompletions) -> DeepSeekClient:
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return DeepSeekClient(api_key="test-not-secret", client=fake, model="test-model")


def test_deepseek_client_requests_json_and_returns_usage() -> None:
    completions = FakeCompletions('{"answer":"grounded"}')
    result = _client(completions).generate_json("Return JSON", "Evidence")
    assert result.content == {"answer": "grounded"}
    assert result.usage.total_tokens == 15
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["extra_body"]["thinking"]["type"] == "disabled"


def test_deepseek_client_rejects_invalid_json() -> None:
    with pytest.raises(LLMClientError, match="invalid JSON"):
        _client(FakeCompletions("not-json")).generate_json("JSON", "Evidence")


def test_deepseek_client_rejects_truncated_output() -> None:
    with pytest.raises(LLMClientError, match="truncated"):
        _client(FakeCompletions("{}", "length")).generate_json("JSON", "Evidence")


def test_missing_key_is_unavailable_without_secret_in_error() -> None:
    client = DeepSeekClient(api_key=None, model="test-model")
    client._api_key = None
    assert not client.available
    with pytest.raises(LLMClientError, match="DEEPSEEK_API_KEY"):
        client.generate_json("JSON", "Evidence")

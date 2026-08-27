"""DeepSeek-first JSON generation with a reserved Ollama adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from gamesight.llm.models import JsonGenerationResult, LLMUsage


class LLMClientError(RuntimeError):
    """Safe, key-free error surfaced to the application fallback path."""


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else text
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    if not text:
        raise LLMClientError("The LLM returned empty JSON content")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMClientError("The LLM returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LLMClientError("The LLM JSON response must be an object")
    return value


class JsonLLMClient(ABC):
    @property
    @abstractmethod
    def provider(self) -> str:
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> JsonGenerationResult:
        pass


class DeepSeekClient(JsonLLMClient):
    """OpenAI-compatible DeepSeek Chat Completions adapter.

    The API key is read from ``DEEPSEEK_API_KEY`` unless explicitly injected.
    It is never included in diagnostics, exceptions or serialized output.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 45.0,
        max_tokens: int = 1800,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self._model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self._base_url = (base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com",
        )).rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self._client = client

    @property
    def provider(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        return bool(self._api_key or self._client is not None)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMClientError("DEEPSEEK_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMClientError(
                "DeepSeek support requires the openai Python package"
            ) from exc
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self.timeout_sec,
            max_retries=1,
        )
        return self._client

    def generate_json(self, system_prompt: str, user_prompt: str) -> JsonGenerationResult:
        started = time.perf_counter()
        try:
            response = self._get_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=self.max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise LLMClientError("DeepSeek JSON output was truncated")
            content = _parse_json_object(choice.message.content)
            raw_usage = getattr(response, "usage", None)
            usage = LLMUsage(
                prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
            )
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"DeepSeek request failed: {type(exc).__name__}") from exc
        return JsonGenerationResult(
            content=content,
            provider=self.provider,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=usage,
        )


class OllamaClient(JsonLLMClient):
    """Reserved local provider using Ollama's JSON chat endpoint."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 90.0,
    ) -> None:
        self._model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self._base_url = (base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434",
        )).rstrip("/")
        self.timeout_sec = timeout_sec

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        return True

    def generate_json(self, system_prompt: str, user_prompt: str) -> JsonGenerationResult:
        started = time.perf_counter()
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = _parse_json_object((body.get("message") or {}).get("content"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LLMClientError(f"Ollama request failed: {type(exc).__name__}") from exc
        return JsonGenerationResult(
            content=content,
            provider=self.provider,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=LLMUsage(
                prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(body.get("eval_count", 0) or 0),
                total_tokens=(
                    int(body.get("prompt_eval_count", 0) or 0)
                    + int(body.get("eval_count", 0) or 0)
                ),
            ),
        )

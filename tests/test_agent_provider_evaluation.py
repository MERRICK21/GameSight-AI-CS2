"""Network-free tests for the opt-in real-provider benchmark layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gamesight.evaluation.agent_provider_cli import main as provider_cli_main
from gamesight.evaluation.agent_fixture import load_agent_eval_suite
from gamesight.evaluation.agent_provider import (
    ProviderPricing,
    run_provider_eval_suite,
)
from gamesight.llm.client import JsonLLMClient
from gamesight.llm.models import JsonGenerationResult, LLMUsage


PROVIDER_SUITE = Path("evaluation/agent_provider_v1.json")


class AutonomousProviderDouble(JsonLLMClient):
    """Chooses tools from the live prompt instead of replaying fixture turns."""

    def __init__(self) -> None:
        self._suggestion_id: str | None = None

    @property
    def provider(self) -> str:
        return "provider_double"

    @property
    def model(self) -> str:
        return "autonomous-v1"

    @property
    def available(self) -> bool:
        return True

    def generate_json(self, system_prompt: str, user_prompt: str) -> JsonGenerationResult:
        prompt = json.loads(user_prompt)
        observations = prompt["tool_observations"]
        if not observations:
            self._suggestion_id = prompt["eligible_suggestion_ids"][0]["suggestion_id"]
            content = {
                "status": "tool_calls",
                "tool_calls": [
                    {"call_id": "candidates", "tool_name": "list_coaching_candidates", "arguments": {}},
                    {"call_id": "evidence", "tool_name": "get_round_evidence", "arguments": {"round_id": "round_001"}},
                    {"call_id": "context", "tool_name": "get_decision_context", "arguments": {"round_id": "round_001"}},
                    {"call_id": "knowledge", "tool_name": "search_knowledge", "arguments": {"suggestion_id": self._suggestion_id, "query": "contact evidence decision quality", "top_k": 1}},
                ],
                "final": None,
            }
        else:
            search = next(
                item for item in observations if item["tool_name"] == "search_knowledge"
            )
            chunk_id = search["data"]["results"][0]["knowledge_chunk_id"]
            chinese = "Simplified Chinese" in system_prompt
            if chinese:
                reasoning = "该次接触构成有证据支持的复盘窗口，不能自动判定为操作失误。"
                action = "复盘接触前的掩体利用和准星位置。"
                summary = "现有证据支持一次有边界的复盘。"
                strengths = ["录像保留了可审计的接触证据。"]
                weaknesses = ["部分决策上下文仍然不可用。"]
                drills = ["使用已链接的接触片段复盘掩体。"]
                focus = ["基于证据复盘首次接触。"]
            else:
                reasoning = "The contact is an evidence-backed review window, not an automatic error."
                action = "Review cover and crosshair placement before the contact."
                summary = "The available evidence supports a bounded review."
                strengths = ["The replay retains auditable contact evidence."]
                weaknesses = ["Some decision context remains unavailable."]
                drills = ["Review the linked contact from available cover."]
                focus = ["Evidence-based contact review."]
            content = {
                "status": "final",
                "tool_calls": [],
                "final": {
                    "suggestions": [{
                        "source_suggestion_id": self._suggestion_id,
                        "reasoning": reasoning,
                        "action": action,
                        "knowledge_chunk_ids": [chunk_id],
                        "evaluation_basis": "decision_quality",
                    }],
                    "summary": {
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "practice_drills": drills,
                        "focus_areas": focus,
                        "overall_assessment": summary,
                        "knowledge_chunk_ids": [chunk_id],
                    },
                },
            }
        return JsonGenerationResult(
            content=content,
            provider=self.provider,
            model=self.model,
            latency_ms=11,
            usage=LLMUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
        )


def test_provider_suite_separates_planning_and_preflight_cases() -> None:
    suite = load_agent_eval_suite(PROVIDER_SUITE)
    assert len(suite.cases) == 7
    assert sum(case.expectation.expect_llm_called for case in suite.cases) == 4
    assert all("provider" in case.tags for case in suite.cases)


def test_repeated_provider_report_measures_stability_tokens_and_cost() -> None:
    report = run_provider_eval_suite(
        load_agent_eval_suite(PROVIDER_SUITE),
        AutonomousProviderDouble(),
        repetitions=2,
        pricing=ProviderPricing(
            input_usd_per_million=1.0,
            output_usd_per_million=2.0,
            price_source="https://example.test/pricing",
            price_verified_at="2026-09-01",
        ),
    )
    metrics = report.metrics
    assert metrics.trial_count == 14
    assert metrics.provider_call_trials == 8
    assert metrics.total_llm_calls == 16
    assert metrics.contract_metrics.pass_rate == 1.0
    assert metrics.case_pass_consistency == 1.0
    assert metrics.tool_path_stability == 1.0
    assert metrics.outcome_stability == 1.0
    assert metrics.total_prompt_tokens == 480
    assert metrics.total_completion_tokens == 320
    assert metrics.total_tokens == 800
    assert metrics.estimated_cost_usd == 0.00112
    assert metrics.p50_latency_ms == 22
    assert metrics.p95_latency_ms == 22


def test_provider_report_does_not_serialize_prompts_or_generated_prose() -> None:
    report = run_provider_eval_suite(
        load_agent_eval_suite(PROVIDER_SUITE), AutonomousProviderDouble(),
    )
    encoded = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    assert "system_prompt" not in encoded
    assert "user_prompt" not in encoded
    assert "private/eval-frame.png" not in encoded
    assert "The contact is an evidence-backed" not in encoded
    assert "Contract pass rate | 100.0%" in report.to_markdown()


def test_provider_cli_requires_explicit_call_acknowledgement(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gamesight-agent-provider-eval"])
    with pytest.raises(SystemExit) as exc:
        provider_cli_main()
    assert exc.value.code == 2

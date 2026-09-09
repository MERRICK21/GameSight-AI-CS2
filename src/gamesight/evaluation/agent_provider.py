"""Opt-in repeated evaluation against a real JSON LLM provider.

Unlike the deterministic CI suite, this module measures model planning variance,
latency and token usage.  It stores no prompts, retrieved text or generated
coaching prose in its report.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict, Field

from gamesight.coach.agent_engine import AgentCoachConfig, SingleAgentCoach
from gamesight.evaluation.agent import (
    AgentEvalCase,
    AgentEvalCaseResult,
    AgentEvalMetrics,
    AgentEvalSuite,
    RecordingCoachToolRegistry,
    aggregate_agent_results,
    evaluate_agent_case,
)
from gamesight.evaluation.agent_fixture import prepare_agent_eval_case
from gamesight.llm.client import JsonLLMClient
from gamesight.llm.models import JsonGenerationResult


class ProviderPricing(BaseModel):
    """Caller-supplied current pricing; no patch-sensitive rate is hard-coded."""

    model_config = ConfigDict(extra="forbid")

    input_usd_per_million: float | None = Field(default=None, ge=0.0)
    output_usd_per_million: float | None = Field(default=None, ge=0.0)
    price_source: str | None = Field(default=None, max_length=500)
    price_verified_at: str | None = Field(default=None, max_length=32)


class ProviderTrialResult(BaseModel):
    case_id: str
    repetition: int = Field(ge=1)
    result: AgentEvalCaseResult


class ProviderEvalMetrics(BaseModel):
    contract_metrics: AgentEvalMetrics
    trial_count: int
    provider_call_trials: int
    total_llm_calls: int
    successful_agent_rate: float
    case_pass_consistency: float
    tool_path_stability: float
    outcome_stability: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None = None


class ProviderAgentEvalReport(BaseModel):
    schema_version: str = "1.0"
    suite_name: str
    provider: str
    model: str
    repetitions: int
    pricing: ProviderPricing
    metrics: ProviderEvalMetrics
    trials: list[ProviderTrialResult]

    def to_markdown(self) -> str:
        m = self.metrics
        c = m.contract_metrics
        cost = (
            f"${m.estimated_cost_usd:.6f}"
            if m.estimated_cost_usd is not None else "not calculated"
        )
        lines = [
            f"# Provider Agent evaluation: {self.suite_name}",
            "",
            f"Provider/model: `{self.provider}/{self.model}`  ",
            f"Repetitions: {self.repetitions}",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Contract pass rate | {c.pass_rate:.1%} |",
            f"| Task completion | {c.task_completion_rate:.1%} |",
            f"| Fallback accuracy | {c.fallback_accuracy:.1%} |",
            f"| Tool precision / recall | {c.tool_selection_precision:.1%} / {c.tool_selection_recall:.1%} |",
            f"| Argument validity | {c.tool_argument_validity:.1%} |",
            f"| Citation coverage | {c.citation_coverage:.1%} |",
            f"| Successful Agent rate | {m.successful_agent_rate:.1%} |",
            f"| Case pass consistency | {m.case_pass_consistency:.1%} |",
            f"| Tool-path stability | {m.tool_path_stability:.1%} |",
            f"| Outcome stability | {m.outcome_stability:.1%} |",
            f"| Latency p50 / p95 | {m.p50_latency_ms:.0f} / {m.p95_latency_ms:.0f} ms |",
            f"| Prompt / completion tokens | {m.total_prompt_tokens} / {m.total_completion_tokens} |",
            f"| Estimated cost | {cost} |",
            "",
            "## Trials",
            "",
            "| Case | Run | Result | Stop/fallback | Tools | Tokens | Latency |",
            "|---|---:|---|---|---|---:|---:|",
        ]
        for trial in self.trials:
            result = trial.result
            state = "PASS" if result.passed else "FAIL"
            reason = result.fallback_reason or result.stop_reason or "-"
            tools = ", ".join(item.value for item in result.tool_names) or "-"
            lines.append(
                f"| `{trial.case_id}` | {trial.repetition} | {state} | "
                f"`{reason}` | {tools} | {result.total_tokens} | {result.latency_ms} ms |"
            )
        return "\n".join(lines) + "\n"


class CountingLLMClient(JsonLLMClient):
    """Count provider calls without retaining prompts or completions."""

    def __init__(self, delegate: JsonLLMClient) -> None:
        self.delegate = delegate
        self.call_count = 0

    @property
    def provider(self) -> str:
        return self.delegate.provider

    @property
    def model(self) -> str:
        return self.delegate.model

    @property
    def available(self) -> bool:
        return self.delegate.available

    def generate_json(self, system_prompt: str, user_prompt: str) -> JsonGenerationResult:
        self.call_count += 1
        return self.delegate.generate_json(system_prompt, user_prompt)


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _dominant_fraction(values: list[tuple]) -> float:
    if not values:
        return 1.0
    return Counter(values).most_common(1)[0][1] / len(values)


def run_provider_eval_suite(
    suite: AgentEvalSuite,
    llm: JsonLLMClient,
    *,
    repetitions: int = 1,
    pricing: ProviderPricing | None = None,
) -> ProviderAgentEvalReport:
    """Run real provider trials sequentially to keep cost/rate use explicit."""
    if repetitions < 1 or repetitions > 20:
        raise ValueError("repetitions must be between 1 and 20")
    pricing = pricing or ProviderPricing()
    trials: list[ProviderTrialResult] = []
    expanded_cases: list[AgentEvalCase] = []

    for case in suite.cases:
        for repetition in range(1, repetitions + 1):
            prepared = prepare_agent_eval_case(case)
            counted = CountingLLMClient(llm)
            registry = RecordingCoachToolRegistry()
            coach = SingleAgentCoach(
                prepared.retriever,
                counted,
                base_coach=prepared.base_coach,
                locale=case.scenario.locale,
                registry=registry,
                config=AgentCoachConfig(
                    max_iterations=case.scenario.max_iterations,
                    max_tool_calls=case.scenario.max_tool_calls,
                    max_observation_chars=case.scenario.max_observation_chars,
                ),
            )
            run = coach.run(prepared.analysis, prepared.report)
            result = evaluate_agent_case(
                case, run, registry.records, llm_calls=counted.call_count,
            )
            trials.append(ProviderTrialResult(
                case_id=case.case_id,
                repetition=repetition,
                result=result,
            ))
            expanded_cases.append(case)

    results = [trial.result for trial in trials]
    contract = aggregate_agent_results(
        f"{suite.name} ({llm.provider}/{llm.model})", expanded_cases, results,
    ).metrics
    by_case: dict[str, list[AgentEvalCaseResult]] = defaultdict(list)
    for trial in trials:
        by_case[trial.case_id].append(trial.result)
    pass_consistency = sum(
        len({result.passed for result in values}) == 1
        for values in by_case.values()
    ) / len(by_case)
    tool_stability = sum(
        _dominant_fraction([
            tuple(item.value for item in result.tool_names) for result in values
        ])
        for values in by_case.values()
    ) / len(by_case)
    outcome_stability = sum(
        _dominant_fraction([(
            result.passed,
            result.actual_mode,
            result.fallback_reason,
            result.stop_reason,
            tuple(item.value for item in result.tool_names),
        ) for result in values])
        for values in by_case.values()
    ) / len(by_case)
    prompt_tokens = sum(result.prompt_tokens for result in results)
    completion_tokens = sum(result.completion_tokens for result in results)
    estimated_cost = None
    if (
        pricing.input_usd_per_million is not None
        and pricing.output_usd_per_million is not None
    ):
        estimated_cost = (
            prompt_tokens * pricing.input_usd_per_million
            + completion_tokens * pricing.output_usd_per_million
        ) / 1_000_000
    call_trials = [result for result in results if result.llm_calls > 0]
    successful_agent = [
        result for result in call_trials
        if result.actual_mode == "agent_llm" and result.stop_reason == "completed"
    ]
    latencies = [result.latency_ms for result in call_trials]
    metrics = ProviderEvalMetrics(
        contract_metrics=contract,
        trial_count=len(results),
        provider_call_trials=len(call_trials),
        total_llm_calls=sum(result.llm_calls for result in results),
        successful_agent_rate=(
            len(successful_agent) / len(call_trials) if call_trials else 1.0
        ),
        case_pass_consistency=pass_consistency,
        tool_path_stability=tool_stability,
        outcome_stability=outcome_stability,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_tokens=sum(result.total_tokens for result in results),
        estimated_cost_usd=estimated_cost,
    )
    return ProviderAgentEvalReport(
        suite_name=suite.name,
        provider=llm.provider,
        model=llm.model,
        repetitions=repetitions,
        pricing=pricing,
        metrics=metrics,
        trials=trials,
    )


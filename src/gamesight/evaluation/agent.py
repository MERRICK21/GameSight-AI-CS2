"""Provider-neutral offline evaluation for the bounded Replay Coach Agent.

The evaluator scores observable behaviour only: tool selection, argument
validity, completion/fallback state, citation coverage, guardrail decisions and
runtime telemetry.  It never asks an evaluator LLM to grade another LLM.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gamesight.agent.models import CoachToolName
from gamesight.agent.tools import AgentToolContext, CoachToolRegistry
from gamesight.coach.models import CoachRun


class AgentEvalExpectation(BaseModel):
    """Machine-checkable contract for one offline Agent scenario."""

    model_config = ConfigDict(extra="forbid")

    expected_mode: Literal["rules", "agent_llm"]
    expected_fallback_reason: str | None = None
    expected_stop_reason: str | None = None
    required_tools: set[CoachToolName] = Field(default_factory=set)
    allowed_tools: set[CoachToolName] = Field(default_factory=set)
    min_tool_calls: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=12, ge=0)
    expected_tool_failures: int | None = Field(default=None, ge=0)
    expected_invalid_arguments: int | None = Field(default=None, ge=0)
    min_accepted_enrichments: int = Field(default=0, ge=0)
    min_rejected_enrichments: int = Field(default=0, ge=0)
    require_agent_citations: bool = False
    expect_llm_called: bool = True
    expected_output_language: Literal["en", "zh-CN"] | None = None
    forbidden_output_patterns: list[str] = Field(default_factory=list)


class AgentEvalScenario(BaseModel):
    """Small synthetic replay state used by the reproducible fixture runner."""

    model_config = ConfigDict(extra="forbid")

    analysis_complete: bool = True
    llm_available: bool = True
    include_contact_evidence: bool = True
    knowledge_profile: Literal[
        "normal", "empty", "strategic", "hard_rule", "stale_dynamic",
        "prompt_injection",
    ] = "normal"
    locale: Literal["en", "zh-CN"] = "en"
    max_iterations: int = Field(default=3, ge=1, le=8)
    max_tool_calls: int = Field(default=12, ge=1, le=32)
    max_observation_chars: int = Field(default=48_000, ge=1, le=200_000)


class AgentEvalScriptStep(BaseModel):
    """One deterministic LLM response or provider error in a fixture."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["response", "final", "error"]
    response: dict | None = None
    variant: str | None = None


class AgentEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,79}$")
    description: str = Field(min_length=3, max_length=500)
    tags: set[str] = Field(default_factory=set)
    scenario: AgentEvalScenario = Field(default_factory=AgentEvalScenario)
    script: list[AgentEvalScriptStep] = Field(default_factory=list)
    expectation: AgentEvalExpectation


class AgentEvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    name: str
    description: str = ""
    cases: list[AgentEvalCase] = Field(min_length=1)


class AgentToolCallRecord(BaseModel):
    tool_name: CoachToolName
    ok: bool
    error: str | None = None


class AgentEvalCaseResult(BaseModel):
    case_id: str
    passed: bool
    checks: dict[str, bool]
    failures: list[str] = Field(default_factory=list)
    actual_mode: str
    fallback_reason: str | None = None
    stop_reason: str | None = None
    tool_calls: int = 0
    tool_failures: int = 0
    invalid_arguments: int = 0
    tool_names: list[CoachToolName] = Field(default_factory=list)
    tool_precision: float = 1.0
    tool_recall: float = 1.0
    citation_coverage: float = 1.0
    llm_calls: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tags: set[str] = Field(default_factory=set)


class AgentEvalMetrics(BaseModel):
    case_count: int
    passed_cases: int
    pass_rate: float
    task_completion_rate: float
    fallback_accuracy: float
    tool_selection_precision: float
    tool_selection_recall: float
    tool_argument_validity: float
    tool_execution_success: float
    citation_coverage: float
    guardrail_pass_rate: float
    average_latency_ms: float
    average_total_tokens: float


class AgentEvalReport(BaseModel):
    suite_name: str
    metrics: AgentEvalMetrics
    cases: list[AgentEvalCaseResult]

    def to_markdown(self) -> str:
        m = self.metrics
        lines = [
            f"# Agent offline evaluation: {self.suite_name}",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Cases passed | {m.passed_cases}/{m.case_count} |",
            f"| Pass rate | {m.pass_rate:.1%} |",
            f"| Task completion | {m.task_completion_rate:.1%} |",
            f"| Fallback accuracy | {m.fallback_accuracy:.1%} |",
            f"| Tool precision | {m.tool_selection_precision:.1%} |",
            f"| Tool recall | {m.tool_selection_recall:.1%} |",
            f"| Argument validity | {m.tool_argument_validity:.1%} |",
            f"| Tool execution success | {m.tool_execution_success:.1%} |",
            f"| Citation coverage | {m.citation_coverage:.1%} |",
            f"| Guardrail pass rate | {m.guardrail_pass_rate:.1%} |",
            f"| Average latency | {m.average_latency_ms:.1f} ms |",
            f"| Average tokens | {m.average_total_tokens:.1f} |",
            "",
            "## Cases",
            "",
            "| Case | Result | Stop/fallback | Tools |",
            "|---|---|---|---:|",
        ]
        for result in self.cases:
            state = "PASS" if result.passed else "FAIL"
            reason = result.fallback_reason or result.stop_reason or "-"
            lines.append(
                f"| `{result.case_id}` | {state} | `{reason}` | {result.tool_calls} |"
            )
            for failure in result.failures:
                lines.append(f"\n- `{result.case_id}`: {failure}")
        return "\n".join(lines) + "\n"


class RecordingCoachToolRegistry(CoachToolRegistry):
    """Production registry plus privacy-safe call outcomes for evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[AgentToolCallRecord] = []

    def execute(
        self,
        call_id: str,
        tool_name: CoachToolName,
        arguments: dict,
        context: AgentToolContext,
    ):
        observation = super().execute(call_id, tool_name, arguments, context)
        self.records.append(AgentToolCallRecord(
            tool_name=tool_name,
            ok=observation.ok,
            error=observation.error,
        ))
        return observation


def _check(
    checks: dict[str, bool],
    failures: list[str],
    name: str,
    condition: bool,
    detail: str,
) -> None:
    checks[name] = bool(condition)
    if not condition:
        failures.append(detail)


def evaluate_agent_case(
    case: AgentEvalCase,
    run: CoachRun,
    records: list[AgentToolCallRecord],
    *,
    llm_calls: int,
) -> AgentEvalCaseResult:
    """Score one completed run without using a subjective LLM judge."""
    expected = case.expectation
    diagnostics = run.diagnostics
    actual_tools = {record.tool_name for record in records}
    allowed = expected.allowed_tools or expected.required_tools
    tool_precision = (
        len(actual_tools & allowed) / len(actual_tools) if actual_tools else 1.0
    )
    tool_recall = (
        len(actual_tools & expected.required_tools) / len(expected.required_tools)
        if expected.required_tools else 1.0
    )
    agent_suggestions = [
        item for item in run.suggestions if item.generated_by.endswith("_agent")
    ]
    cited_agent_suggestions = [
        item for item in agent_suggestions if item.knowledge_citations
    ]
    citation_coverage = (
        len(cited_agent_suggestions) / len(agent_suggestions)
        if agent_suggestions else (0.0 if expected.require_agent_citations else 1.0)
    )
    invalid_arguments = sum(
        record.error == "invalid_tool_arguments" for record in records
    )
    output_text = " ".join([
        *(item.reasoning for item in run.suggestions),
        *(item.action for item in run.suggestions),
        run.summary.overall_assessment,
        *run.summary.strengths,
        *run.summary.weaknesses,
        *run.summary.practice_drills,
        *run.summary.focus_areas,
    ]).lower()

    checks: dict[str, bool] = {}
    failures: list[str] = []
    _check(checks, failures, "mode", diagnostics.mode == expected.expected_mode,
           f"mode={diagnostics.mode!r}, expected {expected.expected_mode!r}")
    _check(
        checks, failures, "fallback_reason",
        diagnostics.fallback_reason == expected.expected_fallback_reason,
        f"fallback={diagnostics.fallback_reason!r}, expected {expected.expected_fallback_reason!r}",
    )
    _check(
        checks, failures, "stop_reason",
        diagnostics.agent_stop_reason == expected.expected_stop_reason,
        f"stop={diagnostics.agent_stop_reason!r}, expected {expected.expected_stop_reason!r}",
    )
    _check(checks, failures, "required_tools", tool_recall == 1.0,
           "one or more required tools were not called")
    _check(checks, failures, "allowed_tools", tool_precision == 1.0,
           "one or more tools were outside the case allowlist")
    _check(
        checks, failures, "tool_call_budget",
        expected.min_tool_calls <= diagnostics.agent_tool_calls <= expected.max_tool_calls,
        f"tool_calls={diagnostics.agent_tool_calls} outside expected range",
    )
    if expected.expected_tool_failures is not None:
        _check(
            checks, failures, "tool_failures",
            diagnostics.agent_tool_failures == expected.expected_tool_failures,
            f"tool_failures={diagnostics.agent_tool_failures}, expected {expected.expected_tool_failures}",
        )
    if expected.expected_invalid_arguments is not None:
        _check(
            checks, failures, "invalid_arguments",
            invalid_arguments == expected.expected_invalid_arguments,
            f"invalid_arguments={invalid_arguments}, expected {expected.expected_invalid_arguments}",
        )
    _check(
        checks, failures, "accepted_enrichments",
        diagnostics.accepted_enrichments >= expected.min_accepted_enrichments,
        f"accepted_enrichments={diagnostics.accepted_enrichments} below minimum",
    )
    _check(
        checks, failures, "rejected_enrichments",
        diagnostics.rejected_enrichments >= expected.min_rejected_enrichments,
        f"rejected_enrichments={diagnostics.rejected_enrichments} below minimum",
    )
    _check(
        checks, failures, "citations",
        not expected.require_agent_citations or citation_coverage == 1.0,
        "an accepted Agent suggestion has no retrieved citation",
    )
    _check(
        checks, failures, "llm_call_state",
        (llm_calls > 0) == expected.expect_llm_called,
        f"llm_calls={llm_calls} does not match expectation",
    )
    forbidden_found = [
        pattern for pattern in expected.forbidden_output_patterns
        if pattern.lower() in output_text
    ]
    _check(checks, failures, "forbidden_output", not forbidden_found,
           f"forbidden output patterns found: {forbidden_found}")
    if expected.expected_output_language == "zh-CN":
        han_count = sum("\u4e00" <= char <= "\u9fff" for char in output_text)
        _check(checks, failures, "output_language", han_count >= 4,
               "accepted output does not contain enough Simplified Chinese text")
    elif expected.expected_output_language == "en":
        latin_count = sum("a" <= char <= "z" for char in output_text)
        _check(checks, failures, "output_language", latin_count >= 12,
               "accepted output does not contain enough English text")

    return AgentEvalCaseResult(
        case_id=case.case_id,
        passed=all(checks.values()),
        checks=checks,
        failures=failures,
        actual_mode=diagnostics.mode,
        fallback_reason=diagnostics.fallback_reason,
        stop_reason=diagnostics.agent_stop_reason,
        tool_calls=diagnostics.agent_tool_calls,
        tool_failures=diagnostics.agent_tool_failures,
        invalid_arguments=invalid_arguments,
        tool_names=[record.tool_name for record in records],
        tool_precision=tool_precision,
        tool_recall=tool_recall,
        citation_coverage=citation_coverage,
        llm_calls=llm_calls,
        latency_ms=diagnostics.latency_ms,
        prompt_tokens=diagnostics.prompt_tokens,
        completion_tokens=diagnostics.completion_tokens,
        total_tokens=diagnostics.total_tokens,
        tags=case.tags,
    )


def aggregate_agent_results(
    suite_name: str,
    cases: list[AgentEvalCase],
    results: list[AgentEvalCaseResult],
) -> AgentEvalReport:
    """Aggregate macro quality and micro execution metrics."""
    if len(cases) != len(results):
        raise ValueError("cases and results must have equal length")
    count = len(results)
    if not count:
        raise ValueError("at least one Agent evaluation result is required")
    by_id = {case.case_id: case for case in cases}
    completion_cases = [
        result for result in results
        if by_id[result.case_id].expectation.expected_stop_reason == "completed"
    ]
    fallback_matches = [
        result.checks.get("fallback_reason", False) for result in results
    ]
    total_calls = sum(result.tool_calls for result in results)
    total_failures = sum(result.tool_failures for result in results)
    total_invalid = sum(result.invalid_arguments for result in results)
    citation_cases = [
        result for result in results
        if by_id[result.case_id].expectation.require_agent_citations
    ]
    guardrail_cases = [result for result in results if "guardrail" in result.tags]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 1.0

    metrics = AgentEvalMetrics(
        case_count=count,
        passed_cases=sum(result.passed for result in results),
        pass_rate=sum(result.passed for result in results) / count,
        task_completion_rate=mean([
            float(result.stop_reason == "completed") for result in completion_cases
        ]),
        fallback_accuracy=mean([float(value) for value in fallback_matches]),
        tool_selection_precision=mean([result.tool_precision for result in results]),
        tool_selection_recall=mean([result.tool_recall for result in results]),
        tool_argument_validity=(
            (total_calls - total_invalid) / total_calls if total_calls else 1.0
        ),
        tool_execution_success=(
            (total_calls - total_failures) / total_calls if total_calls else 1.0
        ),
        citation_coverage=mean([
            result.citation_coverage for result in citation_cases
        ]),
        guardrail_pass_rate=mean([
            float(result.passed) for result in guardrail_cases
        ]),
        average_latency_ms=sum(result.latency_ms for result in results) / count,
        average_total_tokens=sum(result.total_tokens for result in results) / count,
    )
    return AgentEvalReport(suite_name=suite_name, metrics=metrics, cases=results)

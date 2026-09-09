"""Versioned, deterministic evaluation of the Replay Coach Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gamesight.evaluation.agent import AgentEvalSuite
from gamesight.evaluation.agent_fixture import (
    load_agent_eval_suite,
    run_agent_eval_suite,
)


SUITE_PATH = Path("evaluation/agent_offline_v1.json")


def test_agent_eval_suite_is_versioned_and_has_unique_cases() -> None:
    suite = load_agent_eval_suite(SUITE_PATH)
    assert suite.schema_version == "1.0"
    assert len(suite.cases) == 26
    assert len({case.case_id for case in suite.cases}) == len(suite.cases)
    assert {"guardrail", "preflight", "tools"} <= {
        tag for case in suite.cases for tag in case.tags
    }


def test_complete_offline_agent_suite_passes_without_network() -> None:
    report = run_agent_eval_suite(load_agent_eval_suite(SUITE_PATH))
    assert report.metrics.case_count == 26
    assert report.metrics.passed_cases == 26
    assert report.metrics.pass_rate == 1.0
    assert report.metrics.task_completion_rate == 1.0
    assert report.metrics.fallback_accuracy == 1.0
    assert report.metrics.tool_selection_precision == 1.0
    assert report.metrics.tool_selection_recall == 1.0
    assert report.metrics.citation_coverage == 1.0
    assert report.metrics.guardrail_pass_rate == 1.0
    # The suite deliberately includes one invalid-argument recovery case and
    # several failed lookups, so these micro metrics must remain below 100%.
    assert 0.9 < report.metrics.tool_argument_validity < 1.0
    assert 0.8 < report.metrics.tool_execution_success < 1.0


def test_agent_eval_report_is_json_and_markdown_serializable() -> None:
    report = run_agent_eval_suite(load_agent_eval_suite(SUITE_PATH))
    encoded = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    markdown = report.to_markdown()
    assert "happy_path_full_tool_chain" in encoded
    assert "Cases passed | 26/26" in markdown
    assert "Guardrail pass rate | 100.0%" in markdown


def test_agent_eval_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    value = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    value["cases"].append(value["cases"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_agent_eval_suite(path)


def test_agent_eval_schema_rejects_unknown_expectation_fields() -> None:
    value = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    value["cases"][0]["expectation"]["subjective_score"] = 0.9
    with pytest.raises(ValueError):
        AgentEvalSuite.model_validate(value)

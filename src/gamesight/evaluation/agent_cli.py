"""CLI for the deterministic Replay Coach Agent evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gamesight.evaluation.agent_fixture import (
    load_agent_eval_suite,
    run_agent_eval_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline Agent benchmark")
    parser.add_argument(
        "suite", nargs="?", default="evaluation/agent_offline_v1.json",
        help="Path to a versioned Agent evaluation JSON suite",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    parser.add_argument("--output", help="Also write the report to this path")
    parser.add_argument(
        "--min-pass-rate", type=float, default=1.0,
        help="Return a failing exit code below this pass rate (default: 1.0)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")

    report = run_agent_eval_suite(load_agent_eval_suite(args.suite))
    rendered = (
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
        if args.json else report.to_markdown()
    )
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
    return 0 if report.metrics.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Explicit opt-in CLI for real-provider Replay Coach Agent evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gamesight.evaluation.agent_fixture import load_agent_eval_suite
from gamesight.evaluation.agent_provider import ProviderPricing, run_provider_eval_suite
from gamesight.llm.client import DeepSeekClient, OllamaClient


def _optional_rate(cli_value: float | None, env_name: str) -> float | None:
    if cli_value is not None:
        return cli_value
    raw = os.getenv(env_name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be numeric") from exc
    if value < 0:
        raise ValueError(f"{env_name} must not be negative")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an opt-in Agent benchmark against DeepSeek or Ollama",
    )
    parser.add_argument(
        "suite", nargs="?", default="evaluation/agent_provider_v1.json",
    )
    parser.add_argument("--provider", choices=["deepseek", "ollama"], default="deepseek")
    parser.add_argument("--model", help="Override the provider model")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument(
        "--allow-provider-calls", action="store_true",
        help="Required acknowledgement that this command may send fixture prompts and incur cost",
    )
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--price-source")
    parser.add_argument("--price-verified-at")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args()
    if not args.allow_provider_calls:
        parser.error("--allow-provider-calls is required for real-provider evaluation")
    if not 1 <= args.repetitions <= 20:
        parser.error("--repetitions must be between 1 and 20")
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")
    try:
        pricing = ProviderPricing(
            input_usd_per_million=_optional_rate(
                args.input_usd_per_million, "DEEPSEEK_INPUT_USD_PER_MILLION",
            ),
            output_usd_per_million=_optional_rate(
                args.output_usd_per_million, "DEEPSEEK_OUTPUT_USD_PER_MILLION",
            ),
            price_source=args.price_source,
            price_verified_at=args.price_verified_at,
        )
    except ValueError as exc:
        parser.error(str(exc))
    llm = (
        DeepSeekClient(
            model=args.model, timeout_sec=args.timeout_sec,
        )
        if args.provider == "deepseek"
        else OllamaClient(model=args.model, timeout_sec=args.timeout_sec)
    )
    if not llm.available:
        parser.error("selected provider is unavailable; configure its API key/service")

    report = run_provider_eval_suite(
        load_agent_eval_suite(args.suite),
        llm,
        repetitions=args.repetitions,
        pricing=pricing,
    )
    rendered = (
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
        if args.json else report.to_markdown()
    )
    print(rendered)
    if args.output:
        Path(args.output).write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8",
        )
    return 0 if report.metrics.contract_metrics.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())


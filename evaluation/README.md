# Replay Coach Agent offline evaluation

`agent_offline_v1.json` is a versioned, network-free contract suite for the
bounded Replay Coach Agent. It runs scripted model turns through the real Agent
runtime, read-only tool registry, RAG retrieval and final coaching validators.

Run it from the repository root:

```powershell
gamesight-agent-eval evaluation/agent_offline_v1.json
```

Use `--json` for machine-readable output or `--output artifacts/agent-eval.md`
to retain a local report. The command exits non-zero when the pass rate is below
`--min-pass-rate` (100% by default), making it suitable for CI.

The initial suite covers:

- normal full-tool and minimal-tool completion;
- tool selection, invalid arguments and recoverable lookup failures;
- duplicate call IDs and fixed iteration/tool/observation budgets;
- incomplete analysis, unavailable providers, empty knowledge and no evidence;
- unknown citations, numeric hallucinations and outcome-bias rejection;
- tactical-principle wording, dynamic-data freshness and prompt injection.

No judge LLM is used. Scores come from typed traces, tool outcomes, immutable
citations and deterministic guardrail results. A future provider run may reuse
the same scenario contracts, but API latency/cost benchmarks should remain an
explicit opt-in job rather than part of ordinary CI.

## Real-provider benchmark

`agent_provider_v1.json` is a smaller seven-case suite for actual model planning.
Four cases ask the provider to inspect candidates, replay evidence, decision
context and knowledge; three verify that preflight failures spend no tokens.
Unlike the scripted suite, the provider decides every tool call itself.

Real calls require an explicit acknowledgement and are never run by CI:

```powershell
gamesight-agent-provider-eval evaluation/agent_provider_v1.json `
  --provider deepseek `
  --repetitions 3 `
  --allow-provider-calls `
  --output artifacts/deepseek-agent-eval.md
```

The API key is read only from `DEEPSEEK_API_KEY`. The report stores aggregate
metrics and privacy-safe traces, not prompts, retrieved passages or generated
coaching prose. To estimate cost, pass the currently verified provider prices:

```powershell
--input-usd-per-million <current-rate> `
--output-usd-per-million <current-rate> `
--price-source <source-url> `
--price-verified-at <YYYY-MM-DD>
```

No price is hard-coded because provider rates are version-sensitive. Repetitions
default to one to avoid accidental spend; use at least three when measuring tool
path and outcome stability.

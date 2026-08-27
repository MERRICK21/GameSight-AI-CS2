# GameSight AI for CS2 — Project Journey and Interview Narrative

## One-line project summary

GameSight AI turns a first-person CS2 recording into auditable rounds, personal
combat events, evidence clips and context-aware coaching by combining deterministic
computer vision, a four-layer RAG knowledge base, an evidence-constrained LLM and
a bounded single Agent with read-only tools.

## Why the project evolved this way

The project did not begin as an “LLM application.” It began with a harder and more
fundamental question: can an ordinary first-person recording be converted into a
reliable, reviewable representation of a match without access to a `.dem` file?

That framing led to a deliberate order of work:

1. Establish typed domain boundaries and testable Sprints.
2. Make video/HUD perception and temporal reconstruction reliable.
3. Preserve evidence and uncertainty instead of hiding detection errors.
4. Add coaching only after the underlying facts became auditable.
5. Add RAG/LLM only as an editor of verified facts, not as an event detector.
6. Add a single Agent only after tools, permissions, validation and fallback paths existed.

## Phase 1 — Sprint-driven engineering foundation

The first implementation was divided into independently testable Sprints:

- Video ingestion, validation and frame sampling.
- HUD perception with region-specific extractors.
- Temporal event detection and round aggregation.
- Optional YOLO player detection and faction classification.
- Object tracking and evidence linking.
- Typed timeline serialization and JSON export.
- Evidence-backed match reports.
- Streamlit product integration and bilingual UI.
- Rule-based coaching, screenshots and review clips.

The important architectural choice was to pass typed Pydantic models between
perception, event reasoning, reporting and coaching instead of sharing arbitrary
dictionaries. That made later RAG and Agent tools possible without rewriting the
entire pipeline.

## Phase 2 — From “detecting pixels” to reconstructing a match

Early versions produced plausible but inaccurate results: one 19-minute recording
could collapse into one round, a persistent kill-feed row could become multiple
kills, or creator watermarks and overlays could influence scene metrics.

The solution was not a larger model alone. The pipeline combined:

- Native scoreboard and timer continuity for round boundaries.
- Native local-player kill highlighting for personal kill attribution.
- Health HUD and selected player-card transitions for death attribution.
- Temporal debounce, persistence fingerprints and sparse/high-rate two-stage sampling.
- Explicit masks and robust tile statistics for creator overlays and relocated UI noise.
- Match-level fixtures that preserve known 20/10 and 25/14 recording ground truths.

This stage established a principle used throughout the project: a detected event
must have a timestamp, confidence, source and reviewable evidence; unavailable
information remains unavailable.

## Phase 3 — Rule-based coach before LLM

The first coach was deterministic. It could identify review windows for flashes,
scope episodes, enemy contacts, likely firefights and deaths, but it intentionally
abstained when map, side, weapon, economy or position evidence was missing.

This exposed an important product problem: natural-language advice can sound
convincing even when its premise is wrong. Examples included calling a player
“passive” because first contact occupied a large percentage of a short round, or
criticizing crosshair stability before any realistic engagement occurred.

The response was to treat round duration, first-contact time and view motion as
descriptive signals rather than automatic tactical verdicts.

## Phase 4 — Evidence-constrained RAG + LLM

The LLM was deliberately removed from event discovery. The two-stage design became:

```text
Video/HUD perception
  -> typed events and round context
  -> deterministic rule suggestions
  -> situation-aware knowledge retrieval
  -> LLM language enrichment
  -> citation/context/version/output gates
  -> deterministic fallback when rejected
```

DeepSeek is the first hosted provider and Ollama remains a local adapter. Chinese
and English passages use multilingual MiniLM embeddings with persistent Chroma
storage.

### Why ordinary chunking was insufficient

A single Markdown corpus can blur the distinction between:

- “CT wins by successfully defusing the planted bomb” — a hard mechanic.
- “In a 5v4, avoid unnecessary isolated fights” — a high-probability principle.

The knowledge base was therefore split into four physical collections:

- `game_rules`
- `tactical_fundamentals`
- `situation_decisions`
- `dynamic_game_data`

Each chunk also carries one of three strengths:

- `hard_rule`
- `strategic_principle`
- `contextual_recommendation`

Soft principles cannot be rewritten as absolute “must/never/禁止/必须” claims.
Patch-sensitive economy data requires stable IDs, sources and verification dates;
stale or unverified values are excluded from retrieval.

### Decision quality instead of outcome bias

Every LLM suggestion declares `evaluation_basis=decision_quality`. A favorable
kill or round result cannot prove that the original action was correct, and a
failed execution cannot by itself prove that the decision was poor. The coach
evaluates the information available at action time.

## Phase 5 — Single Agent with controlled tool use

The project did not jump directly to a multi-Agent framework. A single Agent was
introduced only after the underlying tools and safety boundaries were mature.
It uses the existing provider-neutral JSON LLM interface and Pydantic models, with
no LangChain/LangGraph dependency.

The Agent has five read-only tools:

1. `get_match_overview`
2. `list_coaching_candidates`
3. `get_round_evidence`
4. `get_decision_context`
5. `search_knowledge`

It has no filesystem, shell, network, event-mutation or report-write tool. The
runtime is bounded to three model iterations and twelve tool calls. Tool arguments
are schema validated, call IDs must be unique, observations have a size budget,
and final suggestions may cite only knowledge returned for the same immutable
suggestion ID.

The Agent can decide which round evidence and context it needs, then construct a
situation-specific retrieval query. It still cannot change detected timestamps,
confidence, category or evidence. The existing numeric, context, rule-strength,
version-freshness and outcome-bias validators run after the Agent finishes. Any
failure falls back to the deterministic coach and remains visible in diagnostics.

## Current architecture

```text
CS2 POV recording
  -> ingestion and adaptive sampling
  -> HUD / first-person / optional YOLO perception
  -> temporal event engine and round reconstruction
  -> evidence report + deterministic RuleBasedCoach
  -> Single Replay Coach Agent
       -> match overview tool
       -> rule-candidate tool
       -> round-evidence tool
       -> decision-context tool
       -> four-layer RAG search tool
  -> evidence and policy validators
  -> bilingual Streamlit report or deterministic fallback
```

## Resume-ready bullets

- Built an end-to-end CS2 first-person replay analysis system combining OpenCV,
  OCR, optional YOLO inference, temporal event reconstruction, evidence clips and
  bilingual Streamlit reporting.
- Designed a four-layer RAG architecture with multilingual MiniLM embeddings and
  Chroma persistence, separating hard game rules, tactical principles, situation
  decisions and patch-sensitive game data.
- Implemented a DeepSeek-first/Ollama-compatible structured LLM layer with citation
  allowlists, unknown-number rejection, context abstention, dynamic-data freshness
  checks and deterministic fallback.
- Developed a bounded single Agent with five schema-validated read-only tools,
  explicit iteration/call budgets, immutable suggestion IDs and complete tool-use
  diagnostics; avoided unnecessary multi-Agent/framework complexity.
- Added decision-quality evaluation that rejects hindsight reasoning such as
  treating a kill or round win as proof that an action was strategically correct.
- Maintained a 534-test regression suite covering perception, match-level K/D,
  RAG retrieval, LLM validation, Agent permissions, tool traces and fallback paths.

## Interview narrative

“I started the project as a video-understanding problem rather than an LLM demo.
The first challenge was reconstructing trustworthy rounds and personal K/D from
ordinary recordings, so I built typed perception and temporal reasoning modules
with evidence links and match-level regression fixtures. Once the facts were
auditable, I added a deterministic coach and learned that plausible advice can
still be wrong when context is missing.

I then introduced RAG, but did not put every rule into one generic corpus. I split
hard mechanics, tactical principles, situation decisions and version-sensitive
economy data, and added policy metadata that controls how strongly the LLM may
phrase a conclusion. The LLM is an evidence-constrained editor, not an event
detector, and every unsupported output falls back to rules.

Finally, I implemented a single tool-using Agent. It can decide which match,
round-context and knowledge tools to call, but it cannot access files, execute
commands or modify detected events. Its loop is bounded and fully observable, and
the same validators still run after it finishes. I chose a native typed runtime
instead of adding an Agent framework because the workflow did not yet justify that
dependency. This gave me practical experience with Agent design as a permissions,
state and reliability problem—not just a prompt-engineering problem.”

## Honest limitations and next steps

- The Agent is a bounded single-model loop, not a multi-Agent system.
- It operates on already-extracted evidence and does not autonomously inspect raw video.
- Alive counts, bomb state, kit and detailed economy remain unavailable unless the
  perception layer or user supplies reliable evidence.
- The next meaningful step is an offline Agent evaluation set covering tool choice,
  argument correctness, citation grounding, abstention and end-to-end coaching quality.
- Multi-Agent separation should be considered only if independent evidence,
  retrieval and critic roles produce measurable quality gains over this baseline.

# GameSight AI for CS2

[![Python CI](https://github.com/MERRICK21/GameSight-AI-CS2/actions/workflows/ci.yml/badge.svg)](https://github.com/MERRICK21/GameSight-AI-CS2/actions/workflows/ci.yml)

GameSight AI for CS2 is a modular, evidence-driven system for analysing first-person Counter-Strike 2 gameplay recordings. It is designed to evolve from video ingestion and structured event extraction into a CV-assisted review report.

## Current status

The end-to-end Streamlit application is implemented. It reconstructs rounds
from the native CS2 scoreboard, measures first-person visual effects, can use
an optional CS2 T/CT detector to locate enemy contacts, and produces evidence
screenshots, short H.264 review clips, reports, and grounded coaching cards.
Personal K/D remains unavailable unless it can be attributed from native game
signals; creator watermarks and recording-provider IDs are ignored.

Long recordings use a two-stage path: a complete 2 FPS round/OCR scan is
followed by high-rate analysis only around candidate combat and visual-effect
windows. The Diagnostics tab can reject false personal events, add missed
kills/deaths, rebuild the report immediately, and export correction labels.
Coaching context is explicit: unsupported clock, weapon, economy, utility,
map, or position inputs stay unknown and cannot silently become tactical advice.

## Planned pipeline

`video input -> preprocessing -> HUD parsing + detection -> tracking -> event fusion -> report -> web demo`

## Layout

- `src/gamesight/`: application package and module boundaries.
- `configs/config.example.yaml`: safe local configuration template.
- `data/`: local datasets and annotations (not committed).
- `artifacts/`: generated clips, frames, and reports (not committed).
- `tests/`: future unit and integration tests.
- `docs/`: architecture and data-contract documentation.

## Quick start

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Register the local package with `pip install -e .`.
4. Run `streamlit run src/gamesight/web/app.py`.

The optional enemy-contact feature expects local weights at
`models/yolov10n_cs2.pt`. See `docs/CS2_ENEMY_MODEL.md` for its constraints.

## Evidence-constrained RAG + LLM coach

The optional coach enrichment path uses multilingual MiniLM embeddings and a
local persistent Chroma index. DeepSeek is the first supported hosted provider;
Ollama remains available as a local adapter. The video pipeline and event
detection stay deterministic, and the API is called only after the user clicks
the RAG generation button.

Set the DeepSeek key in the process environment before starting Streamlit:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
streamlit run src/gamesight/web/app.py
```

Knowledge files can be uploaded as Markdown, text, or DOCX. The built-in
coaching evidence policy is always indexed; a local `cs2_basic_rule.docx` is
also included automatically when present. Uploaded sources, the local manual,
and Chroma data remain local and are ignored by Git. The first run downloads
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` into
`models/huggingface/`.

Knowledge is stored in four separate collections:

- `game_rules`: durable mechanics and victory conditions.
- `tactical_fundamentals`: general principles that retain qualifiers and exceptions.
- `situation_decisions`: replay decisions such as save, retake, post-plant and low-time states.
- `dynamic_game_data`: patch-sensitive prices, rewards and economy values with verification metadata.

Numbered sections in ordinary DOCX paragraphs are recognized as headings. Dynamic
facts without a source URL and current verification date are not used. Retrieval
reserves space for situation decisions before generic tactical or weapon knowledge.

For Ollama, choose it in the sidebar and make sure the local service is running.
The model defaults to `qwen3:8b` and can be changed in the UI or with
`OLLAMA_MODEL`.

The LLM is an evidence-constrained editor rather than an event detector: it may
rewrite only existing suggestion IDs, must cite retrieved chunks, cannot alter
timestamps/evidence/confidence, and is rejected if it invents numeric facts or
context-dependent pace claims. Tactical principles cannot be rewritten as hard
prohibitions, and a favorable kill/win result cannot prove that the original
decision was sound. Invalid output automatically falls back to the rule-based
coach. Provider/model, per-layer retrieval counts, accepted/rejected
enrichments, latency, token usage, and fallback reason are shown in diagnostics.

The local index is also reproducible from the command line:

```powershell
gamesight-knowledge build docs/COACHING_EVIDENCE_POLICY.md --reset
gamesight-knowledge query "闪光弹后如何恢复视野"
```

## Design principles

- Keep perception, temporal reasoning, reporting, and UI independently replaceable.
- Pass typed domain objects between modules rather than unstructured dictionaries.
- Preserve timestamps, confidence scores, and evidence for every derived event.
- Treat input quality and unsupported formats as explicit, reportable conditions.

## Current limitations

- Native health-HUD transitions attribute POV deaths, while native local-kill
  highlighting attributes personal kills without reading player names. Exact
  K/D coaching activates only when both native HUD paths are available.
- Generated event clips are silent.
- Native round-clock, weapon/economy/utility, map-position, minimap, and `.dem`
  evidence extraction are future work; context-dependent coaching abstains until
  those inputs are available.

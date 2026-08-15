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

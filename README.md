# GameSight AI for CS2

GameSight AI for CS2 is a modular, evidence-driven system for analysing first-person Counter-Strike 2 gameplay recordings. It is designed to evolve from video ingestion and structured event extraction into a CV-assisted review report.

## Sprint 0 status

This repository contains the project skeleton only. The interfaces are intentionally empty: no models, downloads, or inference logic are included yet.

## Planned pipeline

`video input -> preprocessing -> HUD parsing + detection -> tracking -> event fusion -> report -> web demo`

## Layout

- `src/gamesight/`: application package and module boundaries.
- `configs/config.example.yaml`: safe local configuration template.
- `data/`: local datasets and annotations (not committed).
- `artifacts/`: generated clips, frames, and reports (not committed).
- `tests/`: future unit and integration tests.
- `docs/`: architecture and data-contract documentation.

## Quick start (when implementation begins)

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Register the local package with `pip install -e .`.
4. Copy `configs/config.example.yaml` to `configs/config.local.yaml` and adapt local paths.

## Design principles

- Keep perception, temporal reasoning, reporting, and UI independently replaceable.
- Pass typed domain objects between modules rather than unstructured dictionaries.
- Preserve timestamps, confidence scores, and evidence for every derived event.
- Treat input quality and unsupported formats as explicit, reportable conditions.

## Current non-goals

Sprint 0 does not implement YOLO, ByteTrack, OCR, VLM/LLM calls, Streamlit UI, or video processing.

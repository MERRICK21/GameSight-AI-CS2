# Project Status

## Current Status

**Sprint 3 -- Event Engine** (in progress)


**Sprint 3 -- Event Engine** (in progress)

---

## Sprint Outline

### Sprint 0 -- Project Initialization

- Scaffold, dependencies, domain models, config

### Sprint 1 -- Video Ingestion

- Task 1: VideoReader (OpenCVVideoReader)
- Task 2: FrameSampler
- Task 3: Normalizer (VideoPreprocessor)
- Task 4: Validator + Reporter

### Sprint 2 -- HUD Perception Layer

- Task 1: HUD Layout Parser (HudRegion, HudLayoutProfile, CS2 profile, Registry)
- Task 2: HUD State Extractor (RegionExtractor ABC, 5 extractors, CS2HudParser)
- Task 3: Pipeline Integration (VideoAnalysisPipeline, end-to-end wiring)

### Sprint 3 -- Event Engine

- Task 1: Round Boundary Detector (state machine, debounce, min duration) -- 26 tests
- Task 2: Kill Event Detector (HP-drop death, kill-feed-edge kill) -- 20 tests

Convert HudState sequences into structured GameEvent objects.

- Task 1: Round Boundary Detector (round_start / round_end from state transitions + debouncing)
- Task 2: Kill Event Detector (kill / death from kill-feed activity changes)
- Task 3: Bomb Event Detector (bomb_planted / bomb_defused)
- Task 4: Event Aggregator (merge into RoundAnalysis, produce timeline)

### Sprint 4 -- Object Detection

YOLO-based player detection on frame crops.

- Task 1: YOLO model integration (load + inference)
- Task 2: Player detector (enemy / teammate classification)
- Task 3: Detection pipeline integration with frame sampling

### Sprint 5 -- Tracking

Multi-object tracking across frames.

- Task 1: ByteTrack / SORT integration
- Task 2: Track ID assignment + cross-frame association
- Task 3: Tracking pipeline integration

### Sprint 6 -- Timeline JSON

Merge events + tracks into structured, serialisable timeline.

- Task 1: Timeline data model
- Task 2: Timeline builder (events + tracks -> sorted timeline)
- Task 3: JSON export + evidence linking

### Sprint 7 -- LLM Evidence-based Report

Feed structured timeline to LLM for natural-language analysis.

- Task 1: Evidence formatter (structured data -> LLM prompt)
- Task 2: LLM report generator (OpenAI API)
- Task 3: Report output (structured text + stats)

### Sprint 8 -- Streamlit Demo

Interactive web demo for the full pipeline.

- Task 1: Video upload + processing UI
- Task 2: Timeline visualization
- Task 3: Report display

---

## Completed

### Sprint 0 -- Project Initialization

- Project scaffold (`src/gamesight/`)
- `pyproject.toml`, `README.md`, `requirements.txt`
- Domain models, config skeleton
- GitHub repo established

### Sprint 1 -- Video Ingestion

- Task 1: VideoReader (metadata, error handling, DI)
- Task 2: FrameSampler (arbitrary FPS, seek, edge cases) -- 13 tests
- Task 3: Normalizer (resolution / aspect / FPS quality) -- 12 tests
- Task 4: Validator + Reporter (file validation, ingestion report) -- 28 tests

### Sprint 2 -- HUD Perception Layer

- Task 1: HUD Layout Parser (HudRegion, HudLayoutProfile, CS2 16:9, Registry) -- 31 tests
- Task 2: HUD State Extractor (5 extractors, CS2HudParser) -- 42 tests
- Task 3: Pipeline Integration (VideoAnalysisPipeline) -- 10 tests

**Sprint 2 total: 83 tests, all passing**

### Sprint 3 -- Event Engine

- Task 1: Round Boundary Detector (state machine, debounce, min duration) -- 26 tests
- Task 2: Kill Event Detector (HP-drop death, kill-feed-edge kill) -- 20 tests

- Task 1: Round Boundary Detector
  - `RoundBoundaryDetector` implements `EventEngine` ABC
  - State machine: idle -> candidate_start -> in_round -> candidate_end -> idle
  - Configurable debounce window (flicker suppression)
  - Minimum round duration enforcement
  - `finalize()` support for mid-video truncation
  - Evidence-tagged GameEvent output
  - 26 tests, all passing

---

## Next

**Sprint 3 Task 2** -- Kill Event Detector

Detect PLAYER_KILL / PLAYER_DEATH events from kill-feed activity transitions
and HUD state changes.

---

## Notes

- 1 pre-existing test failure in `test_validator.py::test_accepts_custom_extensions` (unrelated)
- HUD coordinates / colour ranges calibrated against CS2 16:9 standard layout
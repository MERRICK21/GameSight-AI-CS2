# Project Status

## Current Status

**Sprint 2 — HUD Perception Layer** (in progress)

---

## Completed

### Sprint 0 — Project Initialization

- Project scaffold (`src/gamesight/`)
- `pyproject.toml`, `README.md`, `requirements.txt`
- Domain models, config skeleton
- GitHub repo established

### Sprint 1 — Video Ingestion

- Sprint 1 Task 1 — `VideoReader` (OpenCVVideoReader)
  - Metadata reading (FPS, width, height, duration, FourCC)
  - `VideoReadError`, dependency injection for testing
  - Full unit test coverage
- Sprint 1 Task 2 — `FrameSampler`
  - Arbitrary sample FPS, accurate timestamps, seek
  - Edge cases: 60fps to 10fps, sample_fps > native, missing FPS, empty video, mid-stream read failure, release(), dtype/shape validation
  - 13 tests, all passing
- Sprint 1 Task 3 — `Normalizer` (VideoPreprocessor)
  - Resolution, aspect-ratio, and FPS quality assessment
  - `QualityDiagnostic` dataclass
  - 12 tests, all passing
- Sprint 1 Task 4 — `Validator` + `Reporter`
  - Video file validation (extension, resolution, FPS checks)
  - Ingestion quality report with sampling plan
  - 28 tests (27 pass, 1 pre-existing failure in custom extensions test)

### Sprint 2 — HUD Perception Layer

- Sprint 2 Task 1 — `HUD Layout Parser`
  - `HudRegion` domain model (normalized coordinates, `to_pixel()`)
  - `HudLayoutProfile` domain model (region lookup, serialisable)
  - CS2 standard 16:9 built-in profile (7 HUD regions)
  - `HudProfileRegistry` with pre-loaded defaults
  - 31 tests, all passing
- Sprint 2 Task 2 — `HUD State Extractor`
  - `RegionExtractor` ABC (per-region extraction contract)
  - `CrosshairExtractor` (intensity-variance-based crosshair detection)
  - `HPBarExtractor` (colour-threshold HP estimate, armour detection)
  - `KillFeedExtractor` (bright-pixel activity detection)
  - `MoneyExtractor` (text visibility detection)
  - `RoundInfoExtractor` (round-active / timer-visible detection)
  - `CS2HudParser` implements `HudParser` ABC, delegates to region extractors
  - 42 tests, all passing (29 extractor + 13 parser)

---

## Next

**Sprint 2 Task 3** — End-to-end pipeline integration

Connect `VideoReader` → `FrameSampler` → `CS2HudParser` in the orchestration layer.
Run a real CS2 video through the pipeline and validate HUD state output.

---

## Notes

- 1 pre-existing test failure in `test_validator.py::test_accepts_custom_extensions` (unrelated to current sprint)
- HUD coordinates and colour ranges are calibrated against standard CS2 16:9 layout; may need fine-tuning with real screenshots

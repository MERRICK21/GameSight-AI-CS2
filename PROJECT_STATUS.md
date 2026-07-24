# Project Status

## Current Status

**Sprint 2 -- HUD Perception Layer** (complete)

---

## Completed

### Sprint 0 -- Project Initialization

- Project scaffold (src/gamesight/)
- pyproject.toml, README.md, equirements.txt
- Domain models, config skeleton
- GitHub repo established

### Sprint 1 -- Video Ingestion

- Sprint 1 Task 1 -- VideoReader (OpenCVVideoReader)
  - Metadata reading (FPS, width, height, duration, FourCC)
  - VideoReadError, dependency injection for testing
  - Full unit test coverage
- Sprint 1 Task 2 -- FrameSampler
  - Arbitrary sample FPS, accurate timestamps, seek
  - Edge cases: 60fps to 10fps, sample_fps > native, missing FPS, empty video
  - 13 tests, all passing
- Sprint 1 Task 3 -- Normalizer (VideoPreprocessor)
  - Resolution, aspect-ratio, and FPS quality assessment
  - QualityDiagnostic dataclass
  - 12 tests, all passing
- Sprint 1 Task 4 -- Validator + Reporter
  - Video file validation (extension, resolution, FPS checks)
  - Ingestion quality report with sampling plan
  - 28 tests

### Sprint 2 -- HUD Perception Layer

- Sprint 2 Task 1 -- HUD Layout Parser
  - HudRegion domain model (normalized coordinates, 	o_pixel())
  - HudLayoutProfile domain model (region lookup, serialisable)
  - CS2 standard 16:9 built-in profile (7 HUD regions)
  - HudProfileRegistry with pre-loaded defaults
  - 31 tests, all passing
- Sprint 2 Task 2 -- HUD State Extractor
  - RegionExtractor ABC (per-region extraction contract)
  - CrosshairExtractor (intensity-variance-based crosshair detection)
  - HPBarExtractor (colour-threshold HP estimate, armour detection)
  - KillFeedExtractor (bright-pixel activity detection)
  - MoneyExtractor (text visibility detection)
  - RoundInfoExtractor (round-active / timer-visible detection)
  - CS2HudParser implements HudParser ABC, delegates to region extractors
  - 42 tests, all passing
- Sprint 2 Task 3 -- Pipeline Integration
  - VideoAnalysisPipeline implementing AnalysisPipeline ABC
  - Wires VideoReader -> CS2HudParser end-to-end
  - Per-frame HUD state collection + summary statistics
  - Dependency injection for testability (mock reader, real parser)
  - Error resilience (captures reader failures as warnings)
  - 10 integration tests, all passing

---

## Next

**Sprint 3 -- Event Engine**

Parse collected HudState sequences into structured GameEvent objects:
round boundaries, kills, deaths, bomb events.

---

## Notes

- 1 pre-existing test failure in 	est_validator.py::test_accepts_custom_extensions (unrelated)
- HUD coordinates / colour ranges calibrated against CS2 16:9 standard layout
- Sprint 2 total: 83 tests, all passing

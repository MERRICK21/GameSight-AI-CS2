# Project Status

## Current Status

**Sprint 5 -- Tracking** (in progress)

---

## Sprint Outline

### Sprint 0 -- Project Initialization

- Scaffold, dependencies, domain models, config

### Sprint 1 -- Video Ingestion

- Task 1: VideoReader, Task 2: FrameSampler, Task 3: Normalizer, Task 4: Validator + Reporter

### Sprint 2 -- HUD Perception Layer

- Task 1: HUD Layout Parser, Task 2: HUD State Extractor, Task 3: Pipeline Integration

### Sprint 3 -- Event Engine

- Task 1: RoundBoundaryDetector, Task 2: KillEventDetector, Task 3: Skipped, Task 4: EventAggregator

### Sprint 4 -- Object Detection

- Task 1: YOLODetector, Task 2: PlayerClassifier, Task 3: Pipeline Integration

### Sprint 5 -- Tracking

Multi-object tracking across frames.

- Task 1: IOUTracker (IOU-based multi-object tracker)
- Task 2: Track lifecycle (ID assignment, creation, termination, re-identification)
- Task 3: Tracking pipeline integration

### Sprint 6 -- Timeline JSON

- Task 1: Timeline data model, Task 2: Timeline builder, Task 3: JSON export

### Sprint 7 -- LLM Evidence-based Report

- Task 1: Evidence formatter, Task 2: LLM report generator, Task 3: Report output

### Sprint 8 -- Streamlit Demo

- Task 1: Video upload + processing UI, Task 2: Timeline visualization, Task 3: Report display

---

## Completed

### Sprint 0 -- Project Initialization

### Sprint 1 -- Video Ingestion (53 tests)

### Sprint 2 -- HUD Perception Layer (83 tests)

### Sprint 3 -- Event Engine

- Task 1: RoundBoundaryDetector (state machine, debounce) -- 26 tests
- Task 2: KillEventDetector (HP death, kill-feed kill) -- 20 tests
- Task 3: Skipped (bomb not in kill feed)
- Task 4: EventAggregator (events -> RoundAnalysis) -- 12 tests

**Sprint 3 total: 58 tests**

### Sprint 4 -- Object Detection

- Task 1: YOLODetector (model load + inference, DI) -- 12 tests
- Task 2: PlayerClassifier (BGR colour enemy/teammate) -- 11 tests
- Task 3: Pipeline integration (detector + classifier in pipeline) -- 16 tests

**Sprint 4 total: 39 tests**

**Cumulative: 209 tests, all passing**

---

## Notes

- 1 pre-existing test failure in `test_validator.py::test_accepts_custom_extensions` (unrelated)
- HUD coordinates / colour ranges calibrated against CS2 16:9 standard layout
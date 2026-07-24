# Project Status

## Current Status

**Sprint 6 — Timeline JSON** (next)

---

## Completed

### Sprint 0 — Project Initialization
### Sprint 1 — Video Ingestion (53 tests)
### Sprint 2 — HUD Perception Layer (83 tests)

### Sprint 3 — Event Engine (58 tests)

- RoundBoundaryDetector (state machine, debounce)
- KillEventDetector (HP death, kill-feed kill)
- EventAggregator (events → RoundAnalysis)

### Sprint 4 — Object Detection (39 tests)

- YOLODetector (model load + inference, DI)
- PlayerClassifier (BGR colour enemy/teammate)
- Pipeline integration

### Sprint 5 — Tracking (22 + 19 tests)

- IOUTracker (IOU greedy matching, track lifecycle)
- Pipeline integration

---

## Cumulative: 234 tests, all passing

Pipeline architecture:

```
VideoInput → VideoReader → VideoFrame
    ├── CS2HudParser → HudState (minimap, crosshair, HP, armour, kill feed, money, round info)
    └── YOLODetector → PlayerClassifier (enemy/teammate) → Detection[]
            └── IOUTracker → Track[] (ID, lifecycle)
    ↓
VideoAnalysisPipeline → AnalysisResult
    ↓
Event Engine (RoundBoundaryDetector + KillEventDetector) → GameEvent[]
    ↓
EventAggregator → RoundAnalysis[] (structured timeline)
```

## Next: Sprint 6 — Timeline JSON

- Timeline data model serialisation
- Timeline builder (events + tracks → sorted timeline)
- JSON export + evidence linking

## Sprint 7 — LLM Evidence-based Report

## Sprint 8 — Streamlit Demo
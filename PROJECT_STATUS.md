# Project Status

## Current Status

**All sprints complete!** 🎉

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

### Sprint 6 — Timeline JSON (41 tests)

- `serialization/` module: `timeline.py` + `exporter.py`
- `MatchTimeline`, `RoundTimeline`, `TimelineEvent`, `TrackSummary`, `EvidenceRef` (Pydantic)
- `TimelineBuilder` (AnalysisResult + Tracks → MatchTimeline)
- `TimelineExporter` (JSON serialisation + file export)
- Convenience functions: `timeline_to_json()`, `export_timeline()`

### Sprint 7 — LLM Evidence-based Report (41 tests)

- `reporting/models.py`: `MatchReport`, `RoundReport`, `ReportFinding`, `EvidenceLink`, `RoundStats`, `MatchOverview`
- `reporting/builder.py`: `EvidenceReportBuilder` — per-round stats, findings with evidence links
- `reporting/generator.py`: `EvidenceReportGenerator` — implements `ReportGenerator.generate()` → JSON-safe dict
- Every finding carries explicit `EvidenceLink` references (frame, timestamp, source)
- Categories: combat, movement, utility, teamplay, round_flow
- Severities: info, warning, critical

### Sprint 8 — Streamlit Demo (27 tests)

- `web/demo.py`: `generate_demo_events()`, `generate_demo_tracks()`, `run_demo_pipeline()`
- `web/app.py`: Full Streamlit app — sidebar config, video upload, pipeline runner, 5 result tabs
- Tabs: Overview (metrics + round table), Timeline (expandable findings), Report (full document), Evidence Explorer (all links), Raw JSON (download)
- Demo Mode with synthetic 5-round CS2 match data for instant pipeline demonstration
- CSS-styled finding cards with severity colour coding
- Download buttons for Timeline JSON and Report JSON

---

## Cumulative: 343 tests, all passing (1 pre-existing failure in validator)

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
    ↓
TimelineBuilder → MatchTimeline → TimelineExporter → JSON
    ↓
EvidenceReportBuilder → MatchReport → EvidenceReportGenerator → dict/JSON
    ↓
Streamlit App (web/app.py + web/demo.py)
```

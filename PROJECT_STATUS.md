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
- `TimelineBuilder` (AnalysisResult + Tracks → MatchTimeline)
- `TimelineExporter` (JSON serialisation + file export)

### Sprint 7 — LLM Evidence-based Report (41 tests)

- `reporting/` module: `MatchReport`, `RoundReport`, `ReportFinding`, `EvidenceLink`
- `EvidenceReportBuilder` — per-round stats, findings with evidence links
- `EvidenceReportGenerator` — implements `ReportGenerator.generate()`

### Sprint 8 — Streamlit Demo (27 tests)

- `web/demo.py` + `web/app.py` — full Streamlit app with 5 tabs

### Sprint 9 — Internationalization (10 tests)

- `i18n/` module: `I18nLoader` with JSON locale files
- English (`en.json`) + Simplified Chinese (`zh-CN.json`)
- Language selector in Streamlit sidebar
- All user-facing strings translatable with `{key}` interpolation
- Missing key fallback prevents UI breakage

### Sprint 10 — AI Coach (12 tests)

- `coach/` module: `CoachSuggestion` model + `CoachEngine` ABC + `RuleBasedCoach`
- 6 coaching categories: aim, positioning, game_sense, utility, economy, teamplay
- 6 rule-based analysis rules: death-heavy rounds, aggressive rounds, late/early enemy contact, no-combat rounds, high combat density
- Every suggestion includes: timestamp, round_id, reasoning, action, confidence, evidence links
- `CoachEngine` ABC designed for future LLM replacement

### Sprint 11 — Evidence Screenshots (11 tests)

- `evidence/` module: `EvidenceImage` model + `ScreenshotExtractor` ABC + `OpenCVScreenshotExtractor`
- Extract PNG screenshots at event frame indices via OpenCV seek+read
- Configurable `max_screenshots` limit, fallback frame computation
- Screenshots displayed alongside timeline events and coach suggestions
- Dependency injection via mock `cv2` for testing

---

## Cumulative: 376 tests, all passing (1 pre-existing failure in validator)

## Pipeline architecture

```
VideoInput → VideoReader → VideoFrame
    ├── CS2HudParser → HudState
    └── YOLODetector → PlayerClassifier → Detection[]
            └── IOUTracker → Track[]
    ↓
Event Engine → GameEvent[] → EventAggregator → RoundAnalysis[]
    ↓
TimelineBuilder → MatchTimeline → JSON
EvidenceReportBuilder → MatchReport → JSON
    ↓
RuleBasedCoach → CoachSuggestion[] (evidence-based coaching)
OpenCVScreenshotExtractor → EvidenceImage[] (event screenshots)
    ↓
Streamlit App (i18n EN/ZH, 6 tabs: Overview, Timeline, Report, Evidence, AI Coach, JSON)
```

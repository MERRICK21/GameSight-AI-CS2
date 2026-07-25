# Project Status -- GameSight AI for CS2

## Current Status

**Active development** -- GPU acceleration enabled, all 396 tests passing.

---

## Latest Fixes (2026-07-25)

### GPU Acceleration Enabled
- **Replaced PyTorch CPU-only** (`torch 2.13.0+cpu`) with CUDA 12.4 version (`torch 2.6.0+cu124`)
- **RTX 3090 Ti (24GB)** now detected and usable: `torch.cuda.is_available() == True`
- **EasyOCR** switched from `gpu=False` to `gpu=True` in `ocr.py` (ScoreReader + PlayerNameReader)
- **YOLO detector** now explicitly loads model with `.to('cuda')` for GPU inference

### Performance Optimization
- **Video reader** changed from `CAP_PROP_POS_FRAMES` seeking to sequential read + `grab()` skipping
  - Avoids slow keyframe-seeking overhead for h264/h265 codecs
  - `grab()` only parses frame headers without full decode

### Code Bugs Fixed
- **`server.py:496`** -- syntax error: literal `` `n `` characters replaced with proper newlines in `CS2HudParser()` call
- **`run_analysis.py:73-80`** -- wrong API usage: `CS2HudParser(extractors=[...])` fixed to `CS2HudParser(CS2_STANDARD_16X9, {...})`
- **`RoundStats.deaths_detected`** -- missing field re-added to model (was replaced by `player_died: bool` without updating coach engine)
- **`builder.py`** -- now passes `deaths_detected=deaths` count alongside `player_died`
- **Test defaults aligned**: `RoundBoundaryDetector` debounce (8) and min_round_duration (15.0) tests updated
- **`test_accepts_custom_extensions`** -- validator test fixed (file-existence check was blocking custom extension acceptance)

---

## Completed Sprints

### Sprint 0 -- Project Initialization
### Sprint 1 -- Video Ingestion (53 tests)
- OpenCVVideoReader: metadata inspection + frame sampling
- VideoValidator: format/resolution/fps checks
- VideoPreprocessor: quality diagnostics

### Sprint 2 -- HUD Perception Layer (83 tests)
- CS2HudParser: per-region delegation to extractors
- 5 extractors: Crosshair, HPBar, KillFeed, Money, RoundInfo
- CS2_STANDARD_16X9 profile with 7 HUD regions, normalized coordinates

### Sprint 3 -- Event Engine (58 tests)
- RoundBoundaryDetector: state machine with debounce + min duration
- KillEventDetector: HP drop (death) + kill-feed rising edge (kill)
- EventAggregator: GameEvent[] -> RoundAnalysis[] timeline

### Sprint 4 -- Object Detection (39 tests)
- YOLODetector: YOLO-backed player detection with DI for testing
- PlayerClassifier: BGR colour enemy/teammate classification
- Pipeline integration

### Sprint 5 -- Tracking (41 tests)
- IOUTracker: IOU greedy matching with track lifecycle
- Track creation, association, and termination after lost frames

### Sprint 6 -- Timeline JSON (41 tests)
- `serialization/` module: `MatchTimeline`, `RoundTimeline`, `TimelineEvent`, `TrackSummary`, `EvidenceRef`
- `TimelineBuilder`: AnalysisResult + Tracks -> MatchTimeline
- `TimelineExporter`: JSON serialisation + file export

### Sprint 7 -- Evidence Report (41 tests)
- `reporting/` module: `MatchReport`, `RoundReport`, `ReportFinding`, `EvidenceLink`, `RoundStats`, `MatchOverview`
- `EvidenceReportBuilder`: per-round stats + evidence-grounded findings
- `EvidenceReportGenerator`: implements `ReportGenerator.generate()` -> JSON-safe dict

### Sprint 8 -- Streamlit Demo (27 tests)
- `web/demo.py`: synthetic CS2 match data generators
- `web/app.py`: full Streamlit app -- upload, pipeline, 7 result tabs

### Sprint 9 -- Internationalization (10 tests)
- `i18n/` module: `I18nLoader` with `en.json` + `zh-CN.json`
- Language selector in Streamlit sidebar
- All UI text + coach reasoning + report findings translatable

### Sprint 10 -- AI Coach (12 tests)
- `coach/` module: `CoachSuggestion`, `CoachSummary`, `CoachEngine` ABC, `RuleBasedCoach`
- 10 rules: death-heavy, aggressive, late/early contact, no-combat, density, K/D trend, survival pattern, momentum, consistency
- Post-match summary: strengths, weaknesses, practice drills, focus areas

### Sprint 11 -- Evidence Screenshots (11 tests)
- `evidence/` module: `EvidenceImage` + `OpenCVScreenshotExtractor`
- Screenshot extraction at event frame indices via OpenCV
- Displayed alongside timeline events and coach suggestions

### UX Refinements
- **Player filter**: skip spectating frames after death, player name input
- **Live Analysis tab**: single-screenshot tactical advice (HP, armour, crosshair, kill-feed)
- **Score-based round detection**: `RoundInfoExtractor` now detects CT (blue) and T (yellow) score colours
- **Full Chinese localization**: all UI labels, coach advice, report findings, summary headings
- **Upload limit**: 500MB via `.streamlit/config.toml`
- **Performance**: OCR sparse sampling (every 30 frames), 5-min timeout, ETA display
- **GPU acceleration**: PyTorch CUDA 12.4, EasyOCR GPU, YOLO CUDA device, sequential video reader

---

## Pipeline Architecture

```
VideoInput -> OpenCVVideoReader -> VideoFrame
    +-> CS2HudParser -> HudState (crosshair, HP, armour, kill feed, money, round info + scores)
    +-> [YOLODetector -> PlayerClassifier -> IOUTracker -> Track[]]  (optional, GPU-accelerated)
    -> Event Engine -> GameEvent[] (round_start/end, player_kill, player_death, enemy_first_visible)
    -> EventAggregator -> RoundAnalysis[]
    -> TimelineBuilder -> MatchTimeline -> JSON export
EvidenceReportBuilder -> MatchReport -> JSON export
RuleBasedCoach -> CoachSuggestion[] + CoachSummary
OpenCVScreenshotExtractor -> EvidenceImage[]
    -> Streamlit App (i18n EN/ZH, 7 tabs: Overview, Timeline, Report, Evidence, AI Coach, Live, JSON)
```

---

## Test Summary

**396 passing**, 0 failures.

| Module | Tests |
|--------|-------|
| Video Ingestion + Preprocessing | 53 |
| HUD Perception | 83 |
| Event Engine | 58 |
| Object Detection | 39 |
| Tracking | 41 |
| Timeline JSON | 41 |
| Evidence Report | 41 |
| Streamlit Demo | 27 |
| Internationalization | 10 |
| AI Coach | 12 |
| Evidence Screenshots | 11 |

---

## Known Issues & Planned Improvements

### P0 -- Must Fix
- [ ] **Round detection accuracy**: score-colour heuristic needs field validation with real CS2 footage. Blue/yellow thresholds may need per-map calibration.
- [ ] **Speed with OCR enabled**: EasyOCR is now GPU-accelerated but still has overhead; consider Tesseract or on-demand OCR only on keyframes.

### P1 -- Should Do
- [ ] **AI Coach specificity**: current suggestions are template-based. With OCR reading player names + kill-feed text, suggestions can reference specific weapons, opponents, and map positions.
- [ ] **Real YOLO integration**: YOLO detector exists (GPU-ready) but needs testing with real CS2 video footage.
- [ ] **Player name OCR matching**: use `PlayerNameReader` to verify whose POV is active, enabling true player-ID filtering.
- [ ] **Weapon detection**: OCR on the bottom-right weapon/utility area to know what guns and utility the player has.
- [ ] **Minimap analysis**: track player dot on minimap for positioning, rotation speed, and map control analysis.

### P2 -- Nice to Have
- [ ] **Multi-map support**: HUD profiles for different maps (Inferno, Mirage, etc.) -- currently only 16:9 generic.
- [ ] **GIF/video clips for key events**: export short video segments around kills/deaths for review.
- [ ] **Comparison mode**: compare two matches side by side to track improvement.
- [ ] **CS2 demo file (.dem) support**: parse demo files directly instead of relying on video recordings.
- [ ] **LLM narrative report**: swap `RuleBasedCoach` with an LLM-backed engine for more natural, contextual advice.
- [ ] **Heatmap generation**: death locations, kill locations, movement paths overlaid on map images.

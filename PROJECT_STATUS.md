# Project Status -- GameSight AI for CS2

## Current Status

**Active development** -- GPU acceleration enabled, full test suite passing.

---

## Latest Fixes (2026-08-13)

### Real-video accuracy
- **Round detection validated on `test_video/test1.mp4`**: native scoreboard OCR reconstructs the complete 0:0 to 13:5 score sequence and all 18 rounds at 2 analysis FPS.
- **OCR noise rejection**: score changes must increase the total by exactly one and remain stable across two reads; portraits and separator artefacts are ignored.
- **False personal combat totals disabled**: generic kill-feed brightness and HUD colour estimates no longer produce fabricated kills/deaths.
- **Watermark exclusion clarified**: creator IDs and video overlays are not treated as CS2 player identity or gameplay evidence.
- **Representative keyframes restored**: three interior gameplay frames per round (up to 54), plus verified visual moments, remain available in the Live Analysis tab even without personal combat events.
- **First-person visual analysis added**: per-round flash exposure, scoped time, view-motion score, and stationary-view ratio are computed from the central gameplay viewport.
- **Creator overlays excluded by design**: motion analysis crops HUD edges and the bottom watermark band; names such as “幽羽” are never used as player identity or gameplay evidence.
- **Evidence-based first-person coaching**: substantial flash exposure and long continuous scope holds produce localized, auditable suggestions with source-frame links.
- **Motion inference corrected**: viewport activity is descriptive only; opening traversal, turning, and weapon switching no longer generate unsupported “aim not settled” advice.
- **Multiple moments per round**: continuous flash and scope episodes retain their own timestamps/evidence, alongside three post-opening/mid/late representative frames per round.
- **Real engagement windows added**: the optional CS2-specific detector distinguishes native T/CT character classes and only creates a review event when the detected faction opposes the POV player's native bottom-centre team HUD.
- **Teammates no longer trigger combat advice**: validation around 50–100 seconds of `test1.mp4` retained T models as teammates and generated evidence only for CT appearances.
- **Grounded encounter coaching**: up to three confirmed enemy-visible windows per round prompt a review of the surrounding two seconds (pre-aim, cover, disengage/trade conditions) without claiming aim quality from one frame.
- **Evidence screenshots prioritised**: confirmed engagement/flash/scope frames now consume screenshot capacity before generic phase samples, so late gunfights are not dropped by the 72-image limit.
- **10 FPS long-video path accelerated**: visual effects retain the selected rate, while round HUD parsing runs at 2 FPS, only the required round-info region is parsed, first-person analysis uses a 640px working frame, and score OCR runs every 2 seconds.
- **Real-video performance validated**: `test1.mp4` (1162 seconds) completed the 10 FPS core pass without engagement detection in about 123 seconds while still reconstructing all 18 rounds.
- **Repository hygiene**: local recordings in `test_video/` and optional model weights in `models/` are excluded from Git so large or licensed assets are not uploaded.

### Web application
- **Upload capacity increased to 2 GB** with chunked temporary-file copying and a 30-minute real-analysis timeout.
- **Streamlit Python 3.11 compilation errors fixed** in translated f-strings.
- **Current limitation is explicit**: personal kills/deaths stay unavailable until native CS2 HUD signals can be attributed reliably.
- **Unavailable data is no longer shown as zero**: K/D fields render as `—`, and combat-based coaching is withheld instead of generating false "no combat" advice.

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
- **Upload limit**: 2GB via `.streamlit/config.toml`
- **Performance**: split-rate visual/HUD processing, 2-second score OCR cadence, 30-minute timeout, ETA display
- **GPU acceleration**: PyTorch CUDA 12.4, EasyOCR GPU, YOLO CUDA device, sequential video reader

---

## Pipeline Architecture

```
VideoInput -> OpenCVVideoReader -> VideoFrame
    +-> CS2HudParser -> HudState (crosshair, HP, armour, kill feed, money, round info + scores)
    +-> FirstPersonAnalyzer -> flash/scope/view-motion + native POV team samples
    +-> [CS2FactionDetector -> T/CT bodies -> opposing-faction samples]  (optional, GPU-accelerated)
    -> Event Engine -> GameEvent[] (round_start/end, enemy_first_visible, engagement_candidate, visual moments)
    -> EventAggregator -> RoundAnalysis[]
    -> TimelineBuilder -> MatchTimeline -> JSON export
EvidenceReportBuilder -> MatchReport -> JSON export
RuleBasedCoach -> CoachSuggestion[] + CoachSummary
OpenCVScreenshotExtractor -> EvidenceImage[]
    -> Streamlit App (i18n EN/ZH, 7 tabs: Overview, Timeline, Report, Evidence, AI Coach, Live, JSON)
```

---

## Test Summary

**411 passing**, 0 failures.

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
- [x] **Round detection accuracy for the current 16:9 recording**: validated at 18/18 rounds from the native 0:0 to 13:5 scoreboard sequence.
- [ ] **Speed with OCR enabled**: EasyOCR is now GPU-accelerated but still has overhead; consider Tesseract or on-demand OCR only on keyframes.

### P1 -- Should Do
- [ ] **AI Coach specificity**: flash/scope and confirmed engagement-window suggestions are now evidence-based; map positions and weapon-aware advice remain future work.
- [x] **Real CS2 model integration**: optional GPU inference has been validated on real footage for T/CT separation and enemy-contact evidence. The non-commercial model remains an uncommitted local dependency; see `docs/CS2_ENEMY_MODEL.md`.
- [ ] **Native personal combat attribution**: use CS2 kill-feed highlighting, HP/death transitions, and first-person HUD state; ignore creator watermarks.
- [ ] **Weapon detection**: OCR on the bottom-right weapon/utility area to know what guns and utility the player has.
- [ ] **Minimap analysis**: track player dot on minimap for positioning, rotation speed, and map control analysis.

### P2 -- Nice to Have
- [ ] **Multi-map support**: HUD profiles for different maps (Inferno, Mirage, etc.) -- currently only 16:9 generic.
- [ ] **GIF/video clips for key events**: export short video segments around kills/deaths for review.
- [ ] **Comparison mode**: compare two matches side by side to track improvement.
- [ ] **CS2 demo file (.dem) support**: parse demo files directly instead of relying on video recordings.
- [ ] **LLM narrative report**: swap `RuleBasedCoach` with an LLM-backed engine for more natural, contextual advice.
- [ ] **Heatmap generation**: death locations, kill locations, movement paths overlaid on map images.

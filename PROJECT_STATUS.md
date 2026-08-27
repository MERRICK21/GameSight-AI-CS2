# Project Status -- GameSight AI for CS2

## Current Status

**Active development** -- GPU acceleration enabled, full test suite passing.

---

## Latest Improvements (2026-08-27)

### Single-Agent + evidence-constrained RAG/LLM coach
- **Confirmed architecture implemented**: DeepSeek is the first hosted LLM provider, Ollama is preserved as a functional local adapter, and the coaching workflow now runs through a bounded single Agent. Multi-Agent orchestration remains deliberately out of scope until evaluation demonstrates a need.
- **Native single-Agent runtime**: implemented without introducing LangChain/LangGraph or a new framework dependency. One provider-neutral JSON model performs up to three iterations and twelve tool calls, with typed turn schemas, unique call IDs, observation-size limits and deterministic failure states.
- **Five read-only Agent tools**: `get_match_overview`, `list_coaching_candidates`, `get_round_evidence`, `get_decision_context`, and `search_knowledge`. There is intentionally no filesystem, shell, network, event-mutation or report-write tool.
- **Suggestion-bound tool retrieval**: the Agent must call `search_knowledge` for each suggestion it enriches; retrieved chunks are bound to immutable suggestion IDs and the existing citation allowlist prevents cross-suggestion or unobserved citations.
- **Agent observability and fallback**: diagnostics expose iterations, tool calls, tool failures, tools used, token/latency totals and stop reason. Invalid turns, duplicate IDs, bad arguments, exhausted budgets, final-without-retrieval and rejected claims all fall back to `RuleBasedCoach` without losing the underlying analysis.
- **Provider-neutral LLM boundary**: typed JSON clients expose availability, provider/model identity, token usage and latency without serializing API keys. DeepSeek uses its OpenAI-compatible Chat Completions endpoint with JSON mode; the key is read only from `DEEPSEEK_API_KEY`.
- **Multilingual retrieval stack**: `paraphrase-multilingual-MiniLM-L12-v2` supplies normalized Chinese/English embeddings and Chroma stores caller-owned vectors in a persistent local index. A deterministic in-memory store supports fast unit tests.
- **DOCX/Markdown/text ingestion**: uploaded coaching material is converted into stable, heading-aware chunks with content-derived IDs. Reports expose privacy-safe URIs such as `upload://guide.docx`, never a user machine's absolute path.
- **Four-layer CS2 knowledge architecture**: the local index is physically separated into `game_rules`, `tactical_fundamentals`, `situation_decisions`, and `dynamic_game_data`. Ordinary-paragraph numbered headings in `cs2_basic_rule.docx` are promoted into semantic sections before chunking; the current manual yields 112 structured chunks rather than one undifferentiated article.
- **Hard rules are distinct from coaching principles**: every chunk carries `hard_rule`, `strategic_principle`, or `contextual_recommendation`. Hard mechanics may be categorical, while tactical principles must remain qualified and situation advice requires matching observed context. Absolute wording such as “must/never/禁止/必须” is rejected when only soft guidance was cited.
- **Version-sensitive data registry**: reward/loss-bonus and utility-price facts use stable knowledge IDs, `version_sensitive=true`, verification dates and direct source URLs. Unverified, expired or stale dynamic facts are excluded from retrieval so cached economy numbers cannot silently become permanent rules.
- **Situation-first retrieval**: a typed `DecisionContext` supports side, alive counts, bomb state, weapon, kit, time, economy, utility and map context while preserving unknowns. Retrieval reserves candidates for `situation_decisions`, so a 1v4 post-plant decision rule is not displaced by generic weapon descriptions.
- **Decision-quality guardrail**: LLM suggestions must declare `evaluation_basis=decision_quality`; generations that infer a decision was correct merely because a kill, win or successful outcome occurred are rejected. Advice is evaluated from information available at action time, not hindsight.
- **Explicit two-step RAG**: existing rule suggestions form the evidence query, relevant knowledge passages are retrieved, and the LLM may only enrich supplied suggestion IDs. The LLM does not detect kills, deaths, rounds, enemies or other game events.
- **Hard evidence gates**: LLM output is schema-validated; citations must be drawn from the retrieved subset; original round, timestamp, category, confidence and video evidence are immutable; unseen numeric claims and unsupported passive/slow/rushed pace verdicts are rejected.
- **Graceful fallback**: missing keys, incomplete analysis, an empty index, low retrieval scores, request failures, invalid/truncated JSON and rejected claims all retain the deterministic rule-based coach instead of failing the video result.
- **Auditable web integration**: RAG is opt-in and calls the selected model only after an explicit button click. The UI shows provider/model, four-layer chunk coverage, indexed and retrieved chunks, accepted/rejected enrichments, latency, token use, fallback reason and per-card citations with localized layer/rule-strength/version labels.
- **Local-first privacy**: the built-in evidence policy is always available, user knowledge and Chroma vectors stay under Git-ignored local paths, and API secrets are not accepted through report fields or written to diagnostics.
- **Reproducible tooling and evaluation**: `gamesight-knowledge` builds/queries the index from the CLI; a provider-neutral evaluator reports retrieval Recall@K and MRR; fake LLM/embedding tests cover grounding and fallback without network calls.
- **Environment validation**: the new Python dependencies import successfully and a real local Chroma round trip passes. The first MiniLM artifact download is still pending because the current execution environment returned empty Hugging Face cache files after TLS interception; Windows trust-store support and an isolated `models/huggingface/` cache are in place so the project does not rely on disabling certificate checks or deleting the user's global cache.
- **Local CS2 manual integrated**: the 20,050-character `cs2_basic_rule.docx` manual is detected automatically when present and indexed together with the repository evidence policy. The third-party/source document and Word lock files remain Git-ignored, while Markdown/text/DOCX uploads can extend the same local index.
- **Regression status**: compilation and the complete suite pass with **534 tests**. Agent regressions additionally cover valid multi-turn tool use, no-tool final rejection, the fixed permission allowlist, private asset-path redaction, invalid argument handling, untrusted-tool-output prompting and post-Agent outcome-bias rejection.

### Resume / interview talking points
- Designed a production-style **two-step RAG pipeline** that separates deterministic CV/event evidence from probabilistic LLM language generation.
- Implemented **multilingual semantic retrieval** with MiniLM, persistent Chroma storage, stable chunk identities, citation traceability and offline Recall@K/MRR evaluation.
- Built a **DeepSeek-first, provider-agnostic LLM layer** with an Ollama adapter, structured-output validation, token/latency telemetry and secret-safe configuration.
- Added **hallucination controls** that preserve source evidence, reject unknown citations and numeric claims, abstain when tactical context is missing, and automatically fall back to deterministic rules.
- Built a **policy-aware retrieval layer** that encodes rule strength and temporal validity instead of treating all Markdown passages as equally authoritative.
- Added **context-aware decision evaluation**: situation retrieval is prioritized and the LLM is prevented from confusing favorable outcomes with sound decisions.
- Implemented a **bounded single-Agent control loop** with explicit permissions, tool budgets, state observations, trace telemetry and deterministic fallback—without overstating the project as a multi-Agent system.
- Integrated the feature into a bilingual Streamlit product with explicit cost control, local knowledge ingestion, privacy-safe source metadata and user-visible diagnostics.
- Documented the full engineering journey and interview narrative in English and Chinese (`docs/PROJECT_JOURNEY.md` and `docs/PROJECT_JOURNEY_ZH.md`), spanning Sprint decomposition, real-video perception, evidence design, RAG/LLM constraints and Agent tool use.

---

## Latest Improvements (2026-08-15)

### `test2` native K/D regression fixed
- **Verified match total restored to 25/14**: a complete 10 FPS native-HUD pass over `test_video/test2.mp4` now returns 25 POV kills and 14 POV deaths across 22 rounds, matching the supplied ground truth.
- **Sequential multi-kills no longer collapse**: the local kill-feed tracker now counts a fresh native row inserted below an existing row even when the feed remains continuously visible. This recovers the four-kill final round while visual fingerprints still merge persistence/fade frames.
- **White HP digits no longer create false deaths**: low-saturation health glyphs require multiple text-shaped components and edge structure; bright flat patches and normal white HUD themes no longer flicker into death events.
- **Top player-card death evidence added**: the native orange-selected roster card provides an independent life-state signal when a demo/recording immediately switches to another first-person HUD after death. The detector selects the left or right card per round from native highlight evidence, so the halftime side swap is handled without a fixed timestamp or player-name OCR.
- **Creator overlays remain out of scope**: the lower-centre key display and relocated voice widget do not overlap either K/D evidence region and are not used for player identity.
- **Regression verification**: the focused K/D/report/coach tests pass, the full suite is now **505 passing**, and both a full 10 FPS pass and the website-equivalent `2 FPS base + 10 FPS candidate windows` pass produced exactly `25 kills / 14 deaths`.

### Creator-overlay and damaged-frame robustness
- **`test_video/test2.mp4` added as a local HUD/noise regression source**: the 1914x1080, 30 FPS, 1059-second recording contains a persistent lower-centre keyboard/mouse display and a relocated transient voice widget. The recording remains local and is not committed.
- **Creator input display excluded from camera-motion metrics**: the confirmed lower-centre overlay is masked only for descriptive view-motion calculation. It does not suppress native health, timer, kill-feed, weapon, death, or engagement evidence.
- **Relocated UI noise is rejected generically**: motion is now calculated from a robust 6x8 tile distribution, discarding the most locally changing 20% of tiles. Voice widgets, subtitles, chat popups, and animated watermarks can no longer dominate a scene-wide movement score merely because their position changed.
- **Noise diagnostics are auditable**: each first-person sample records a `localized_overlay_score`; match metadata reports how many sampled frames contained strongly localized changes and confirms that the creator-overlay motion filter was active.
- **Single-frame decode recovery added**: a failed compressed-video sample now triggers a bounded seek to the next sample instead of silently ending the entire match. Three consecutive failures still stop safely, mark the analysis incomplete, disable K/D/coaching conclusions, and show a localized partial-decode warning.
- **Real-video validation completed**: `test2` scanned through 1059.0s at 2 FPS with 0 decode failures and 91.88% non-flash native-health coverage; 16.42% of samples exposed localized overlay changes. The `test1` compatibility scan still reached 1161.5s with 0 decode failures and 98.52% health coverage.

### Match-level regression and continuous integration
- **Lightweight ground-truth fixture added**: `tests/fixtures/test1_native_signals.json` stores anonymized round, native kill-highlight, HUD-presence, and artefact signals rather than video frames, names, watermarks, or the 1.3 GB source recording.
- **Cross-rate K/D regression added**: the fixture reconstructs 18 rounds, 20 kills, and 10 deaths at match level; separate 10 FPS and 2 FPS scenarios retain the required round-11 and open round-18 kills while rejecting six one-frame outline artefacts.
- **GitHub Actions CI added**: every push and pull request runs the complete pytest suite on Python 3.11, compiles `src/` and `tests/`, and validates both locale JSON files.
- **CI dependency set isolated**: `requirements-ci.txt` excludes GPU, YOLO, EasyOCR, Streamlit, local weights, and source recordings so ordinary correctness checks remain fast and reproducible on GitHub-hosted runners.
- **README CI visibility added**: the repository now exposes the Python CI status badge at the top of the README.

### Prioritized roadmap
- [x] **1. Match-level accuracy regression**: preserve round count, K/D, sparse-sampling anchors, multi-kill behavior, open final rounds, and false-outline rejection in fast automated fixtures. Expand with additional authorized recordings and HUD variants over time.
- [x] **2. GitHub Actions quality gate**: run tests, compilation, locale validation, and signal-level regressions automatically on pushes and pull requests.
- [x] **3. Two-stage video analysis**: the full recording is scanned at no more than 2 FPS for round/OCR integrity; only candidate kill, death, flash, scope, fire/damage, and enemy-contact windows are decoded at the selected higher FPS.
- [x] **4. Detection diagnostics and correction UI**: personal combat events expose round, timestamp, confidence, method and source; users can reject false positives, add missed kills/deaths by round-relative time, immediately rebuild reports/coaching, and export correction labels.
- [x] **5. Context-aware coaching inputs**: native side evidence is represented explicitly while clock, weapon, economy, utility, map and position remain unknown until evidenced. Encounter coaching states the missing inputs and abstains from unsupported pace, buy, rotation, utility and position verdicts.

### Adaptive analysis, auditability and coaching context
- **Round reconstruction is isolated from high-rate work**: scoreboard OCR and native round HUD signals always traverse the complete recording at the stable base rate, preventing a 10 FPS setting from turning a timeout into a false partial match.
- **Candidate-only refinement added**: high-rate first-person analysis runs only around auditable signals and uses windowed sequential decoding; quiet footage is skipped with OpenCV `grab()` rather than decoded twice.
- **Mixed-rate metrics corrected**: flash, scope, viewport-motion and stationary-view summaries use time-weighted samples so candidate windows cannot distort whole-round percentages.
- **Diagnostics tab added**: raw personal kill/death claims remain immutable while accepted/rejected labels and manually added misses produce a corrected analysis view.
- **Corrections propagate immediately**: match totals, per-round reports and AI Coach output are rebuilt after each edit; label JSON retains source recording, event evidence, confidence and acceptance state for future fixtures.
- **Typed context evidence added**: each round carries timestamped native player-side, clock and first-person weapon observations; money, utility, map and position remain explicitly unknown until evidenced.
- **Coach evidence gate added**: encounter reasoning lists known and unavailable context and refuses to infer tactical conclusions that require missing inputs.

### Native round clock and held-weapon context
- **Round-clock OCR shares the existing sparse score pass**: the centre-top native `M:SS` timer is read every two seconds without adding another full-video OCR schedule; GPU initialization falls back to CPU when needed.
- **Clock continuity is mandatory**: a value enters coaching context only after at least two readings decrease consistently with elapsed video time. Single reads, static text and implausible jumps are discarded, and observed round duration is never converted into a game clock.
- **First-person held view is the primary weapon source**: the new classifier interface supports pistol, rifle, SMG, shotgun, sniper, knife, grenade and C4 categories, while the native bottom-right equipment highlight is auxiliary only and cannot independently create a category.
- **Initial built-in recognition is intentionally conservative**: native circular scope geometry establishes a sniper category, and the distinctive held C4 display/keypad establishes C4. Other categories remain unknown until a trained held-view backend or validated templates are supplied; absence of the optional equipment panel does not reduce primary evidence.
- **Auditable context detail added to AI Coach**: each covered round displays the first verified clock, number of continuous clock reads, localized weapon categories and weapon-evidence frame count.

### Economy, utility and map-position context
- **Native money OCR added**: a tightly cropped bottom-left digit pass runs every eight seconds using the already-loaded OCR reader. Values must be within CS2's `$0-$16,000` range, divisible by `$50`, and repeat consistently inside the same round before becoming economy context.
- **Ambiguous dollar-glyph reads abstain**: tokens such as `$450 -> 5450` are rejected instead of being reported as `$5,450`; a single high-confidence value is still insufficient without a stable second reading.
- **Native location text added**: OCR reads only the bounded label beneath the minimap, then matches it against a controlled native-location vocabulary. Creator names, watermarks and arbitrary text never become positions.
- **Conservative Mirage resolution added**: the match is identified as Mirage only from a strong native anchor such as `Palace Interior`/`Snipers Nest`, or multiple independent supporting anchors. Damaged A/B site glyphs with tied matches are rejected.
- **Localized map evidence added**: verified money, held utility and `map / native position` now appear in the AI Coach context table in English or Simplified Chinese.
- **Held-utility evidence channel added**: flashbang, smoke, HE, Molotov, incendiary and decoy categories are accepted from a high-confidence first-person held-view classifier; a bottom-right inventory highlight may only corroborate the same result and cannot create one. Because the supplied recording does not reliably expose the inventory panel and no trained held-grenade weights are bundled, utility remains unknown on frames without direct held-item evidence rather than inferring ownership from flash/smoke effects.
- **Performance remains bounded**: money/location OCR is limited to active first-person frames at an eight-second cadence and reuses the score reader, avoiding another model load or high-rate full-video pass.

---

## Latest Fixes (2026-08-14)

### Native kill-feed identity and deduplication
- **Single-frame kill-feed artefacts rejected without breaking sparse sampling**: an uncorroborated outline normally persists across at least two distinct sampled frames; at 1--2 FPS, a single observation is retained only when native-frame geometry is exceptionally strong (`>=0.97`). Optional engagement detection remains corroborating rather than mandatory.
- **Persistent rows no longer become extra kills**: a local red-highlighted feed row that fades, briefly misses detection, or moves upward is tracked as one event instead of being counted again from a fixed time-gap heuristic.
- **Ownership remains native and content-independent**: CS2's red local-player outline is still the only attribution signal. Row glyph crops are temporary visual fingerprints used solely for deduplication; player names, creator watermarks, and OCR text are not used to decide ownership.
- **Close multi-kills remain separate**: stacked red-highlighted rows are extracted independently, including shared-border layouts that OpenCV would otherwise join into one contour.
- **Lower-contrast complete outlines are retained**: a geometrically complete native row with a dark feed interior and HUD glyphs can confirm a candidate even when the older aggregate line score is below its cutoff.
- **Ground-truth video revalidated**: `test_video/test1.mp4` reconstructs 18 rounds, 20 unique POV kills, and 10 POV deaths. Round 1 now contains the single supplied kill rather than two observations of the same persistent row.
- **Sparse-sampling regression revalidated**: with engagement detection disabled, both 10 FPS and 2 FPS analysis return the supplied 20-kill total. At 2 FPS, the round-11 kill is retained at 867.5s while six one-frame outline artefacts remain rejected; the 18-round / 20-kill / 10-death ground truth is preserved.
- **Open final-round attribution fixed**: the last OCR round can legitimately have `end_sec=None` when recording stops before the next confirmed score. Native kills are now retained in that open interval; the round-18 AWP kill is detected at 1120.7s with 10 FPS sampling (1122.5s at 2 FPS) and assigned to `round_018`.
- **Right-edge feed geometry tightened**: only red row candidates aligned to the native kill-feed edge receive fingerprints, preventing an ordinary teammate row below a persistent local highlight from becoming another kill.

### Evidence-gated AI coaching
- **Invalid contact-time ratio removed**: first enemy contact is no longer divided by the observed round duration, so a short round cannot make an ordinary spawn-to-contact route look artificially passive or slow.
- **Map-pace verdicts now require missing context**: without map, side, spawn, route, and round-phase evidence, the coach records contact timing but explicitly declines to judge pace.
- **CS2 review knowledge expanded**: encounter and death cards now guide review of counter-strafe timing, expected head line, body exposure, cover, peek information, utility/trade support, and repeated peeks without claiming those mechanics failed from a single frame.
- **Native K/D no longer suppresses visual coaching**: exact HUD-attributed kills/deaths and first-person engagement, flash, scope, and death-clip analysis now run together; up to three real contact/fight windows per round remain available.
- **Neutral observations stay neutral in summaries**: contact timestamps, visual encounter windows, and no-combat records do not automatically become match-level game-sense or teamplay weaknesses.
- **Evidence policy documented**: `docs/COACHING_EVIDENCE_POLICY.md` records which CS2 coaching topics are currently supported, review-only, or blocked pending better evidence.

### Single-screenshot HUD advice
- **Armour presence is no longer a numeric value**: the former boolean-to-`0/1` conversion that made every screenshot appear low-armour has been removed.
- **Screenshot-only armour OCR added**: the native shield value is read conservatively on uploaded screenshots and selected keyframes without adding neural OCR cost to long-video sampling. `test_video/test1.png` now reports 100 armour.
- **Economy-aware armour wording**: unknown or 100 armour produces no purchase warning; values below 60 produce a next-round reminder to refill armour and ensure a helmet when the economy allows.
- **First-round equipment respected**: 100 armour without a helmet is not criticised, since the $800 pistol-round start cannot buy the $1000 armour-and-helmet package.
- **Single-frame inference tightened**: visible crosshair and global kill-feed content are descriptive only; they no longer imply stable aim, stopped movement, or a nearby elimination, and the default action no longer tells the player to hold without tactical context.

### Complete Chinese interface and coach summary
- **Post-match summary moved to the top**: the AI Coach tab now opens with assessment, strengths, weaknesses, focus areas, and drills before individual event cards.
- **Full-combat summary localized**: K/D observations, positioning/game-sense review notes, drills, focus areas, defaults, and overall assessments now use the active locale instead of hard-coded English.
- **Language changes rebuild cached results**: switching between English and Simplified Chinese regenerates report findings, coach suggestions, summary text, screenshot advice, and status labels without rerunning the video pipeline.
- **Remaining visible labels localized**: analysis modes, screenshot-debug controls, HUD region names, severity labels, Yes/No values, evidence captions, uploaded-image captions, frame buttons, and ETA text now follow the selected language.
- **Translation parity guarded by tests**: English and Simplified Chinese locale files must expose the same complete key set; the Streamlit layout is also checked so the summary remains above event cards.

---

## Latest Fixes (2026-08-13)

### Real-video accuracy
- **Round detection validated on `test_video/test1.mp4`**: native scoreboard OCR reconstructs the complete 0:0 to 13:5 score sequence and all 18 rounds at 2 analysis FPS.
- **OCR noise rejection**: score changes must increase the total by exactly one and remain stable across two reads; portraits and separator artefacts are ignored.
- **False personal combat totals disabled**: generic kill-feed brightness no longer fabricates personal kills; deaths are counted only from conservative native health-HUD transitions.
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
- **Contact versus likely fight classification**: an enemy-visible window is now upgraded to `likely_firefight` only when a nearby transient muzzle-flash or two-sided damage-overlay candidate is also present; enemy visibility alone remains `visual_contact`.
- **Opening-phase false advice avoided**: fire/damage candidates are only used near an opposing-faction sighting, so normal spawn movement and weapon switching do not independently create combat advice.
- **Real-video combat-signal calibration**: on `test1.mp4`, visual candidates in the 80--100 second validation segment clustered in the actual 87--97 second engagement sequence; the first 87.0 second enemy contact paired with an 87.5 second shot candidate.
- **Event review clips added**: up to 12 priority engagement/visual-moment events receive silent, browser-compatible H.264 clips covering two seconds before through three seconds after the trigger.
- **Clip-backed coaching**: matching event clips appear in both Evidence and AI Coach tabs, while reports expose per-round encounter/likely-fight counts and candidate-window metadata.
- **Encounter timing metrics**: each contact now records first/last visible timestamps, sampled-frame count, observed visibility span, first shot/damage candidate timestamps, and shot offset relative to first visibility.
- **Phase-aware review prompts**: contact-only, incoming-damage-only, pre-contact fire, immediate return fire, and delayed fire each receive a different evidence-cautious review checklist.
- **First-contact timestamps corrected**: encounter and first-visible events now begin at the earliest opposing-faction sample rather than the later highest-confidence detection frame.
- **Duplicate coach evidence fixed**: an exact event match and timestamp fallback can no longer render the same screenshot twice.
- **Native POV death attribution added**: a stable native health/armour HUD followed by a short sustained disappearance creates a traceable death event; watermarks and creator IDs remain outside the measured region.
- **Flash and edit rejection**: full-screen flash frames are ignored, long HUD absences are rejected, and a one-sample loss is accepted only when a separate damage-overlay candidate corroborates it.
- **Partial combat capabilities**: the UI can now show evidence-backed deaths and survival time while keeping unsupported personal kills as `—`; K/D coaching remains disabled.
- **Death-specific review cards and clips**: native-HUD death transitions produce positioning review prompts and are prioritised for screenshots/short clips.
- **Full-video death calibration**: on `test1.mp4`, all 18 rounds remained correct, native HUD coverage was 98.52%, and 10 clear POV death transitions were identified without treating flash blindness as death.
- **Native POV kill attribution added**: the native red local-kill outline is detected geometrically without reading player names; YOLO enemy and POV-fire signals now corroborate evidence but cannot veto a native HUD-owned kill at sparse sampling rates.
- **Warm-map false-positive rejection**: local-kill outlines must contain a thin, continuous high-fill chroma line beside a dark native feed row containing the white weapon/text glyphs; wood and orange wall textures are rejected.
- **Full-video kill calibration**: `test1.mp4` yields 20 native POV kills after content-aware row tracking, matching the supplied 20-kill ground truth. The former 1136.0s wall-texture false positive is rejected automatically.
- **Native K/D restored**: when both native local-kill highlights and native health-HUD death transitions are available, the UI and coach may use the attributed totals; this recording now reports the expected 20/10.
- **Evidence screenshots prioritised**: confirmed engagement/flash/scope frames now consume screenshot capacity before generic phase samples, so late gunfights are not dropped by the 72-image limit.
- **10 FPS long-video path accelerated**: visual effects retain the selected rate, while round HUD parsing runs at 2 FPS, only the required round-info region is parsed, first-person analysis uses a 640px working frame, and score OCR runs every 2 seconds.
- **YOLO rate decoupled from visual FPS**: enemy inference is capped at 2 FPS even when flash/scope sampling is set to 10 FPS, preventing the 30-minute timeout that previously returned only the first four rounds.
- **Partial results can no longer masquerade as a full match**: timeout output disables K/D totals, marks the round count with `*`, shows a prominent partial-result error, and records the processed timestamp.
- **Real-video performance validated**: `test1.mp4` (1162 seconds) completed the 10 FPS core pass without engagement detection in about 123 seconds while still reconstructing all 18 rounds.
- **Repository hygiene**: local recordings in `test_video/` and optional model weights in `models/` are excluded from Git so large or licensed assets are not uploaded.

### Web application
- **Upload capacity increased to 2 GB** with chunked temporary-file copying and a 30-minute real-analysis timeout.
- **Streamlit Python 3.11 compilation errors fixed** in translated f-strings.
- **Current limitation is explicit**: native K/D requires supported CS2 local-kill highlighting and health-HUD transitions; unsupported/custom HUD signals remain unavailable rather than inferred.
- **Unavailable data is no longer shown as zero**: unsupported kills or deaths render as `—`; exact K/D coaching activates only when both native attribution paths are available.

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

### Sprint 10 -- AI Coach
- `coach/` module: `CoachSuggestion`, `CoachSummary`, `CoachEngine` ABC, `RuleBasedCoach`
- Evidence-gated rules for native combat, neutral contact timing, encounter/likely-fight clips, flash/scope episodes, survival patterns, and post-match summaries
- First-contact timing is descriptive only; pace judgments require map, side, spawn, route, and round-phase context
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
    +-> FirstPersonAnalyzer -> flash/scope/view-motion + native POV team + held weapon evidence
    +-> ScoreReader -> native score/clock + sparse money/location OCR
    +-> ContextBuilder -> stable economy + held utility + verified map/position
    +-> NativeStatusDetector -> bottom health HUD + selected top player card -> POV death events
    +-> NativeKillDetector -> local-kill outline + POV fire + enemy engagement -> conservative POV kill events
    +-> [CS2FactionDetector -> T/CT bodies -> opposing-faction samples]  (optional, GPU-accelerated)
    -> Event Engine -> GameEvent[] (round_start/end, enemy_first_visible, engagement_candidate, visual moments)
    -> EventAggregator -> RoundAnalysis[]
    -> TimelineBuilder -> MatchTimeline -> JSON export
EvidenceReportBuilder -> MatchReport -> JSON export
RuleBasedCoach -> immutable coaching candidates
SingleAgentCoach -> read-only evidence/context/RAG tools -> validated CoachSuggestion[] + CoachSummary
OpenCVScreenshotExtractor -> EvidenceImage[]
EvidenceClipExtractor -> EvidenceClip[] (H.264, event -2s to +3s)
    -> Streamlit App (i18n EN/ZH, 8 tabs: Overview, Timeline, Report, Evidence, AI Coach, Live, Diagnostics, JSON)
```

---

## Test Summary

**534 passing**, 0 failures.

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
| Match-level Regression | 4 |

---

## Known Issues & Planned Improvements

### P0 -- Must Fix
- [x] **Round detection accuracy for the current 16:9 recording**: validated at 18/18 rounds from the native 0:0 to 13:5 scoreboard sequence.
- [ ] **Speed with OCR enabled**: EasyOCR is now GPU-accelerated but still has overhead; consider Tesseract or on-demand OCR only on keyframes.

### P1 -- Should Do
- [ ] **AI Coach specificity**: flash/scope, enemy-contact and likely-firefight suggestions are now evidence-based and clip-backed; map positions and weapon-aware advice remain future work.
- [x] **Real CS2 model integration**: optional GPU inference has been validated on real footage for T/CT separation and enemy-contact evidence. The non-commercial model remains an uncommitted local dependency; see `docs/CS2_ENEMY_MODEL.md`.
- [x] **Native personal combat attribution**: native HUD death transitions and native local-kill highlight persistence are implemented and calibrated to the supplied 20/10 ground truth; encounter/fire signals remain corroborating evidence.
- [ ] **Weapon detection**: OCR on the bottom-right weapon/utility area to know what guns and utility the player has.
- [ ] **Minimap analysis**: track player dot on minimap for positioning, rotation speed, and map control analysis.

### P2 -- Nice to Have
- [ ] **Multi-map support**: HUD profiles for different maps (Inferno, Mirage, etc.) -- currently only 16:9 generic.
- [x] **Video clips for key events**: export silent H.264 review segments around engagement and first-person visual events.
- [ ] **Clip audio and download controls**: optionally preserve source audio and provide an explicit per-event download action.
- [ ] **Comparison mode**: compare two matches side by side to track improvement.
- [ ] **CS2 demo file (.dem) support**: parse demo files directly instead of relying on video recordings.
- [x] **Optional single-Agent narrative coach**: DeepSeek-first/Ollama-compatible tool use is evidence constrained, citation backed, context gated and falls back to `RuleBasedCoach`.
- [ ] **Heatmap generation**: death locations, kill locations, movement paths overlaid on map images.

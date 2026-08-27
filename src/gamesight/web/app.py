"""GameSight AI for CS2 - Streamlit Web Application.

Usage: streamlit run src/gamesight/web/app.py
"""

from __future__ import annotations

import io, json, os, shutil, tempfile, time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from gamesight.coach.engine import RuleBasedCoach
from gamesight.coach.models import CoachRun
from gamesight.coach.rag_engine import EvidenceBoundRagCoach
from gamesight.domain.models import AnalysisResult, EventType, GameEvent, RoundAnalysis, VideoInput
from gamesight.events.aggregator import aggregate_events
from gamesight.events.detectors import KillEventDetector, RoundBoundaryDetector
from gamesight.events.ocr_detector import OCRRoundDetector
from gamesight.evidence.extractor import (
    EvidenceClipExtractor,
    OpenCVScreenshotExtractor,
    build_round_keyframe_events,
)
from gamesight.i18n.loader import I18nLoader
from gamesight.knowledge.embeddings import (
    DEFAULT_MULTILINGUAL_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from gamesight.knowledge.indexing import index_documents
from gamesight.knowledge.dynamic_data import (
    durable_game_rule_documents,
    dynamic_game_documents,
)
from gamesight.knowledge.loaders import (
    load_knowledge_bytes,
    load_knowledge_document,
)
from gamesight.knowledge.retriever import KnowledgeRetriever
from gamesight.knowledge.store import LayeredChromaKnowledgeStore
from gamesight.llm.client import DeepSeekClient, OllamaClient
from gamesight.ingestion.video_reader import OpenCVVideoReader
from gamesight.orchestration.adaptive import (
    build_refinement_windows, merge_samples,
)
from gamesight.live.analyzer import LiveAnalyzer
from gamesight.perception.extractors import (
    CrosshairExtractor, HPBarExtractor, KillFeedExtractor,
    MoneyExtractor, RoundInfoExtractor,
)
from gamesight.perception.hud_parser import CS2HudParser
from gamesight.perception.hud_profiles import CS2_STANDARD_16X9
from gamesight.perception.first_person import (
    FirstPersonAnalyzer, build_first_person_summary_events,
)
from gamesight.perception.native_status import detect_native_deaths
from gamesight.perception.native_kill import detect_native_kills
from gamesight.perception.context import build_round_contexts, context_coverage
from gamesight.perception.cs2_detector import (
    CS2FactionDetector, PlayerDetectionSample, build_engagement_events,
    inference_stride,
)
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.corrections import (
    add_manual_events, apply_event_corrections, build_correction_export,
    diagnostic_rows,
)
from gamesight.web.demo import generate_demo_events, generate_demo_tracks

st.set_page_config(page_title="GameSight AI", page_icon="\U0001f3af", layout="wide", initial_sidebar_state="expanded")

if "analysis_run" not in st.session_state:
    st.session_state.update({
        "analysis_run": False, "result": None, "analysis_obj": None,
        "tracks": None, "progress": 0, "status": "",
        "coach_suggestions": None, "coach_summary": None,
        "coach_diagnostics": None,
        "screenshots": None, "clips": None, "live_result": None,
        "live_state": None, "locale": "en", "mode": "video",
        "content_locale": None, "debug_screenshots": [],
        "player_filter": False, "player_name": "", "use_yolo": False,
        "analysis_base": None, "event_corrections": {},
        "manual_events": [], "correction_revision": 0,
    })

MAX_TIME_REAL = 1800   # 30 min max for long real-video analysis
MAX_TIME_DEMO = 60     # 1 min max for demo
HUD_SAMPLE_FPS = 2.0   # Round/HUD state does not need high-rate analysis.
YOLO_SAMPLE_FPS = 2.0  # Enemy contacts do not justify >2 neural passes/sec.
SCORE_OCR_INTERVAL_SEC = 2.0  # Score persists throughout freeze time.
CONTEXT_OCR_INTERVAL_SEC = 8.0  # Money/location text changes more slowly.
def _loader(): return I18nLoader(st.session_state.get("locale", "en"))
def _t(key: str, **kwargs) -> str: return _loader().t(key, **kwargs)


def _knowledge_citation_text(citation) -> str:
    layer = _t(f"rag.layers.{citation.layer.value}")
    strength = _t(f"rag.strengths.{citation.rule_strength.value}")
    verified = (
        f" · {_t('rag.version_verified')}: {citation.last_verified}"
        if citation.version_sensitive and citation.last_verified else ""
    )
    return (
        f"{citation.title} · {citation.heading or '—'} · {layer} · {strength}"
        f"{verified} · {citation.source_uri} · {citation.score:.2f}"
    )


def _store_coach_run(run: CoachRun) -> None:
    """Store one atomic coach result so text and diagnostics cannot diverge."""
    st.session_state["coach_suggestions"] = run.suggestions
    st.session_state["coach_summary"] = run.summary
    st.session_state["coach_diagnostics"] = run.diagnostics


def _run_rule_coach(analysis: AnalysisResult, report) -> None:
    _store_coach_run(RuleBasedCoach(_loader()).run(analysis, report))


@st.cache_resource(show_spinner=False)
def _rag_store(model_id: str):
    """Cache the local embedding model and persistent Chroma connection."""
    repo_root = Path(__file__).resolve().parents[3]
    embedder = SentenceTransformerEmbeddingProvider(model_id)
    return LayeredChromaKnowledgeStore(
        repo_root / "data" / "knowledge_index", embedder,
    )


def _build_rag_retriever(uploads, model_id: str) -> KnowledgeRetriever:
    """Index built-in policy plus privacy-safe user uploads."""
    repo_root = Path(__file__).resolve().parents[3]
    documents = [load_knowledge_document(
        repo_root / "docs" / "COACHING_EVIDENCE_POLICY.md",
        source_uri="repo://docs/COACHING_EVIDENCE_POLICY.md",
        metadata={
            "knowledge_layer": "tactical_fundamentals",
            "rule_strength": "strategic_principle",
        },
    ), *durable_game_rule_documents(), *dynamic_game_documents()]
    local_manual = repo_root / "cs2_basic_rule.docx"
    if local_manual.is_file():
        documents.append(load_knowledge_document(
            local_manual, source_uri="local://cs2_basic_rule.docx",
            metadata={"knowledge_profile": "cs2_structured_manual"},
        ))
    for upload in uploads or []:
        documents.append(load_knowledge_bytes(
            upload.getvalue(), upload.name,
            source_uri=f"upload://{Path(upload.name).name}",
        ))
    store = _rag_store(model_id)
    index_documents(documents, store)
    return KnowledgeRetriever(store, top_k=4, min_score=0.15)


class _CachedHudParser:
    """Return a previously parsed HUD state when only language changes."""

    def __init__(self, state) -> None:
        self._state = state

    def parse(self, _image, _frame_index, _timestamp_sec):
        return self._state


def _refresh_localized_outputs() -> None:
    """Rebuild cached report/coach strings in the selected locale."""
    analysis = st.session_state.get("analysis_obj")
    if analysis is not None:
        caps = analysis.capabilities
        complete = caps.get("analysis_complete", True)
        personal = caps.get("personal_combat", True)
        kills = caps.get("personal_kills", personal)
        deaths = caps.get("personal_deaths", personal)
        if analysis.video.video_id == "demo_cs2_match":
            warning = None
        elif not complete and not caps.get("input_decode_complete", True):
            processed = max(
                [round_.end_sec or round_.start_sec for round_ in analysis.rounds]
                or [0.0]
            )
            warning = _t("run.video_decode_partial", seconds=processed)
        elif not complete:
            processed = max(
                [round_.end_sec or round_.start_sec for round_ in analysis.rounds]
                or [0.0]
            )
            warning = _t("run.analysis_timed_out", seconds=processed)
        elif personal:
            warning = _t("run.personal_combat_native")
        elif kills:
            warning = _t("run.personal_kills_native")
        elif deaths:
            warning = _t("run.personal_kills_unavailable")
        else:
            warning = _t("run.personal_combat_unavailable")
        analysis.warnings = [warning] if warning else []
        report = EvidenceReportBuilder(loader=_loader()).build(
            analysis, st.session_state.get("tracks"),
        )
        st.session_state["result"] = report.model_dump(mode="json")
        if complete:
            _run_rule_coach(analysis, report)
        st.session_state["status"] = _t("run.complete")

    for debug_result in st.session_state.get("debug_results", []):
        debug_result["advice"] = LiveAnalyzer(
            _CachedHudParser(debug_result["state"]), _loader(),
        ).analyze(debug_result["frame"])
    live_state = st.session_state.get("live_state")
    if live_state is not None:
        st.session_state["live_result"] = LiveAnalyzer(
            _CachedHudParser(live_state), _loader(),
        ).analyze(np.zeros((1, 1, 3), dtype=np.uint8))
    st.session_state["content_locale"] = st.session_state.get("locale", "en")


def _rebuild_analysis_outputs(analysis: AnalysisResult) -> None:
    """Make report and coach reflect the currently accepted event labels."""
    st.session_state["analysis_obj"] = analysis
    builder = EvidenceReportBuilder(loader=_loader())
    report = builder.build(analysis, st.session_state.get("tracks"))
    st.session_state["result"] = report.model_dump(mode="json")
    if analysis.capabilities.get("analysis_complete", True):
        _run_rule_coach(analysis, report)
    else:
        st.session_state["coach_suggestions"] = []
        st.session_state["coach_summary"] = None
        st.session_state["coach_diagnostics"] = None
    st.session_state["content_locale"] = st.session_state.get("locale", "en")


def _filter_post_death(analysis: AnalysisResult) -> tuple[AnalysisResult, int]:
    skipped = 0
    filtered_rounds = []
    for ra in analysis.rounds:
        death_ts = None
        for e in ra.events:
            if e.event_type == EventType.PLAYER_DEATH: death_ts = e.start_sec; break
        if death_ts is None: filtered_rounds.append(ra); continue
        kept = [e for e in ra.events if e.start_sec <= death_ts or e.event_type in (EventType.ROUND_START, EventType.ROUND_END)]
        skipped += len(ra.events) - len(kept)
        new_end = death_ts if ra.end_sec and death_ts < ra.end_sec else ra.end_sec
        filtered_rounds.append(RoundAnalysis(round_id=ra.round_id, start_sec=ra.start_sec, end_sec=new_end, events=kept))
    return AnalysisResult(video=analysis.video, metadata=analysis.metadata, rounds=filtered_rounds,
                          warnings=analysis.warnings,
                          capabilities=analysis.capabilities,
                          round_contexts=analysis.round_contexts,
                          analysis_metadata=analysis.analysis_metadata), skipped


def _real_pipeline(
    video_path: str, sample_fps: float, use_yolo: bool = False,
    source_name: str | None = None,
) -> dict:
    t0 = time.time()
    st.session_state["status"] = _t("run.reading_meta"); st.session_state["progress"] = 5
    video = VideoInput(
        video_id=Path(video_path).stem, path=Path(video_path),
        source_name=source_name,
    )
    reader = OpenCVVideoReader(); metadata = reader.inspect(video)
    duration = metadata.duration_sec or 60
    base_fps = min(float(sample_fps), HUD_SAMPLE_FPS)
    total_est = int(duration * base_fps)

    st.session_state["status"] = _t("run.processing", w=metadata.width or 0, h=metadata.height or 0, fps=metadata.fps or 0)
    st.session_state["progress"] = 10

    # Personal K/D is intentionally disabled, so the video pipeline only
    # needs native round information.  Crosshair/HP/kill-feed/money parsing
    # remains available in the single-screenshot Live Analyzer.
    parser = CS2HudParser(CS2_STANDARD_16X9, {
        "round_info": RoundInfoExtractor(),
    })

    ocr_detector = OCRRoundDetector(profile=CS2_STANDARD_16X9)
    use_score_ocr = ocr_detector.available
    if not use_score_ocr:
        st.warning(_t("run.ocr_fallback"))

    hud_states = []
    score_events = []
    clock_observations: list[dict] = []
    money_observations: list[dict] = []
    position_observations: list[dict] = []
    first_person_analyzer = FirstPersonAnalyzer()
    first_person_samples = []
    player_detection_samples: list[PlayerDetectionSample] = []
    faction_detector = None
    if use_yolo:
        model_path = Path(__file__).resolve().parents[3] / "models" / "yolov10n_cs2.pt"
        try:
            faction_detector = CS2FactionDetector(model_path=model_path)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            st.warning(f"{_t('run.cs2_model_unavailable')} ({exc})")
    # Stage 1 always scans the complete recording at no more than 2 FPS.  It
    # owns round/OCR detection, so a high requested effects rate can never
    # make round reconstruction slower or partial by itself.
    yolo_every = inference_stride(base_fps, YOLO_SAMPLE_FPS)
    next_hud_timestamp = 0.0
    last_score_ocr_timestamp: float | None = None
    last_context_ocr_timestamp: float | None = None
    timed_out = False
    processed_until_sec = 0.0
    for i, frame in enumerate(reader.frames(video, base_fps)):
        if time.time() - t0 > MAX_TIME_REAL:
            timed_out = True
            st.warning(_t(
                "run.analysis_timed_out",
                seconds=processed_until_sec,
            ))
            break
        processed_until_sec = frame.timestamp_sec
        first_person_sample = first_person_analyzer.update(
            frame.image, frame.frame_index, frame.timestamp_sec,
        )
        first_person_samples.append(first_person_sample)
        if faction_detector is not None and i % yolo_every == 0:
            detections = faction_detector.detect(
                frame.image, frame.frame_index, frame.timestamp_sec,
            )
            player_detection_samples.append(PlayerDetectionSample(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                player_team=first_person_sample.player_team,
                detections=detections,
            ))
        if frame.timestamp_sec + 1e-6 >= next_hud_timestamp:
            state = parser.parse(
                frame.image, frame.frame_index, frame.timestamp_sec,
            )
            hud_states.append(state)
            next_hud_timestamp = frame.timestamp_sec + 1.0 / HUD_SAMPLE_FPS
            if use_score_ocr:
                # Score digits persist across the round and freeze time.  A
                # fixed 2-second OCR cadence still supplies the two confirmed
                # reads required by OCRRoundDetector, but halves neural OCR
                # work versus the former once-per-second schedule.  Timer-only
                # HUD updates keep start-boundary precision at 0.5 seconds.
                should_read_score = last_score_ocr_timestamp is None or (
                    frame.timestamp_sec - last_score_ocr_timestamp
                    >= SCORE_OCR_INTERVAL_SEC
                )
                score_events.extend(ocr_detector.update(
                    state,
                    frame.image if should_read_score else None,
                    read_score=should_read_score,
                ))
                if should_read_score:
                    clock_observation = ocr_detector.latest_clock_observation
                    if clock_observation is not None:
                        clock_observations.append(clock_observation)
                    last_score_ocr_timestamp = frame.timestamp_sec
                should_read_context = last_context_ocr_timestamp is None or (
                    frame.timestamp_sec - last_context_ocr_timestamp
                    >= CONTEXT_OCR_INTERVAL_SEC
                )
                if should_read_context:
                    for observation in ocr_detector.read_native_context(
                        state, frame.image,
                        first_person_active=first_person_sample.health_hud_visible,
                    ):
                        kind = observation.get("kind")
                        clean = {
                            key: value for key, value in observation.items()
                            if key != "kind"
                        }
                        if kind == "money":
                            money_observations.append(clean)
                        elif kind == "position":
                            position_observations.append(clean)
                    last_context_ocr_timestamp = frame.timestamp_sec
        if i % 30 == 0:
            elapsed = time.time() - t0
            if i > 0:
                eta = elapsed / i * (total_est - i)
            else:
                eta = 0
            pct = min(10 + int(45 * i / max(total_est, 1)), 55)
            st.session_state["progress"] = pct
            st.session_state["status"] = (
                f"{_t('run.processing_frame', n=i)} | "
                f"{_t('run.eta', seconds=eta)}"
            )
    base_sampling_diagnostics = reader.last_sampling_diagnostics

    # Round boundaries are established from the complete low-rate scan before
    # any candidate refinement.  This is the guardrail that keeps round count
    # independent from the requested high-rate effects analysis.
    n_states = len(hud_states)
    st.session_state["status"] = _t("run.detecting_events", n=n_states)
    st.session_state["progress"] = 75

    rbd = RoundBoundaryDetector()
    # Brightness-only HUD heuristics cannot tell whether a kill-feed entry is
    # the POV player's kill, and the current colour-based HP estimate is not a
    # reliable death signal on custom HUDs.  Keep personal combat totals off
    # until identity-aware OCR is available rather than reporting false stats.
    ked = KillEventDetector(detect_deaths=False, detect_kills=False)
    events = list(score_events)
    if use_score_ocr:
        for state in hud_states:
            events.extend(ked.update(state))
        events.extend(ocr_detector.finalize())
    else:
        for state in hud_states:
            events.extend(rbd.update(state)); events.extend(ked.update(state))
        events.extend(rbd.finalize())
    events.extend(ked.finalize())

    rounds = aggregate_events(events)

    refinement_windows = build_refinement_windows(
        first_person_samples,
        player_detection_samples,
        duration_sec=metadata.duration_sec,
    )
    refined_samples = []
    refinement_sampling_diagnostics = None
    if sample_fps > base_fps and refinement_windows and not timed_out:
        st.session_state["status"] = _t(
            "run.refining_windows", n=len(refinement_windows), fps=sample_fps,
        )
        refined_analyzer = FirstPersonAnalyzer()
        refinement_est = max(1, int(sum(
            window.end_sec - window.start_sec for window in refinement_windows
        ) * sample_fps))
        for i, frame in enumerate(reader.frames_in_windows(
            video, sample_fps, refinement_windows,
        )):
            if time.time() - t0 > MAX_TIME_REAL:
                timed_out = True
                processed_until_sec = max(processed_until_sec, frame.timestamp_sec)
                st.warning(_t(
                    "run.analysis_timed_out", seconds=processed_until_sec,
                ))
                break
            refined_samples.append(refined_analyzer.update(
                frame.image, frame.frame_index, frame.timestamp_sec,
            ))
            if i % 30 == 0:
                st.session_state["progress"] = min(
                    55 + int(17 * i / refinement_est), 72,
                )
        refinement_sampling_diagnostics = reader.last_sampling_diagnostics
        first_person_samples = merge_samples(
            first_person_samples, refined_samples,
        )
    engagement_events = build_engagement_events(
        rounds, player_detection_samples, first_person_samples,
    )
    events.extend(engagement_events)
    native_deaths = detect_native_deaths(rounds, first_person_samples)
    events.extend(native_deaths.events)
    native_kills = detect_native_kills(
        rounds, first_person_samples, engagement_events,
    )
    events.extend(native_kills.events)
    events.extend(build_first_person_summary_events(rounds, first_person_samples))
    rounds = aggregate_events(events)
    round_contexts = build_round_contexts(
        rounds, first_person_samples, clock_observations,
        money_observations, position_observations,
    )
    context_counts = context_coverage(round_contexts)
    st.session_state["progress"] = 85
    native_combat_available = (
        native_kills.available and native_deaths.available
    )
    input_decode_complete = not base_sampling_diagnostics.truncated
    analysis_warnings = [_t(
        "run.personal_combat_native"
        if native_combat_available
        else (
            "run.personal_kills_native"
            if native_kills.available
            else (
                "run.personal_kills_unavailable"
                if native_deaths.available
                else "run.personal_combat_unavailable"
            )
        )
    )]
    if not input_decode_complete:
        analysis_warnings = [_t(
            "run.video_decode_partial", seconds=processed_until_sec,
        )]
    elif timed_out:
        analysis_warnings = [_t(
            "run.analysis_timed_out",
            seconds=processed_until_sec,
        )]
    analysis = AnalysisResult(
        video=video,
        metadata=metadata,
        rounds=rounds,
        warnings=analysis_warnings,
        capabilities={
            "analysis_complete": not timed_out and input_decode_complete,
            "input_decode_complete": input_decode_complete,
            "personal_combat": (
                native_combat_available and not timed_out and input_decode_complete
            ),
            "personal_kills": (
                native_kills.available and not timed_out and input_decode_complete
            ),
            "personal_deaths": (
                native_deaths.available and not timed_out and input_decode_complete
            ),
            "enemy_contact": faction_detector is not None,
            "two_stage_analysis": sample_fps > base_fps,
            "native_round_clock": context_counts["native_round_clock"] > 0,
            "held_weapon_category": context_counts["weapon"] > 0,
            "native_economy": context_counts["economy"] > 0,
            "held_utility": context_counts["utility"] > 0,
            "native_map_position": context_counts["map_position"] > 0,
        },
        round_contexts=round_contexts,
        analysis_metadata={
            "base_scan_fps": base_fps,
            "requested_fps": float(sample_fps),
            "refinement_window_count": len(refinement_windows),
            "refined_sample_count": len(refined_samples),
            "context_ocr_interval_sec": CONTEXT_OCR_INTERVAL_SEC,
            "creator_overlay_motion_filter": True,
            "localized_overlay_frame_count": sum(
                sample.localized_overlay_score >= .45
                for sample in first_person_samples
            ),
            "decode_failure_count": (
                base_sampling_diagnostics.decode_failures
                + (
                    refinement_sampling_diagnostics.decode_failures
                    if refinement_sampling_diagnostics is not None else 0
                )
            ),
            "decode_recovery_count": (
                base_sampling_diagnostics.recoveries
                + (
                    refinement_sampling_diagnostics.recoveries
                    if refinement_sampling_diagnostics is not None else 0
                )
            ),
            "decode_truncated": not input_decode_complete,
            "processed_until_sec": round(processed_until_sec, 3),
            "refinement_duration_sec": round(sum(
                window.end_sec - window.start_sec
                for window in refinement_windows
            ), 2),
        },
    )
    if st.session_state.get("player_filter"):
        analysis, skipped = _filter_post_death(analysis)
        if skipped: st.session_state["status"] += f" ({_t('player_filter.skipped_frames', n=skipped)})"
    st.session_state["analysis_base"] = analysis.model_copy(deep=True)
    st.session_state["event_corrections"] = {}
    st.session_state["manual_events"] = []
    st.session_state["analysis_obj"] = analysis
    builder = EvidenceReportBuilder(loader=_loader())
    cr = builder.build(analysis)
    if timed_out or not input_decode_complete:
        # Partial rounds must never feed match-level coaching conclusions.
        st.session_state["coach_suggestions"] = []
        st.session_state["coach_summary"] = None
        st.session_state["coach_diagnostics"] = None
    else:
        _run_rule_coach(analysis, cr)
    st.session_state["progress"] = 100
    elapsed_total = time.time() - t0
    st.session_state["status"] = (
        f"{_t('run.partial_complete', seconds=processed_until_sec)} "
        f"({elapsed_total:.0f}s)"
        if timed_out or not input_decode_complete
        else f"{_t('run.complete')} ({elapsed_total:.0f}s)"
    )
    st.session_state["content_locale"] = st.session_state.get("locale", "en")
    return cr.model_dump(mode="json")


def _demo_pipeline() -> dict:
    from gamesight.domain.models import VideoMetadata
    st.session_state["status"] = _t("run.generating_demo"); st.session_state["progress"] = 10
    events = generate_demo_events(rounds=5); tracks = generate_demo_tracks()
    st.session_state["progress"] = 40; rounds = aggregate_events(events)
    st.session_state["progress"] = 60
    analysis = AnalysisResult(
        video=VideoInput(video_id="demo_cs2_match", path=Path("demo.mp4")),
        metadata=VideoMetadata(duration_sec=640.0, fps=30.0, width=1920, height=1080), rounds=rounds,
        capabilities={"personal_combat": True})
    if st.session_state.get("player_filter"): analysis, _ = _filter_post_death(analysis)
    st.session_state["analysis_base"] = analysis.model_copy(deep=True)
    st.session_state["event_corrections"] = {}
    st.session_state["manual_events"] = []
    st.session_state["analysis_obj"] = analysis; st.session_state["tracks"] = tracks
    builder = EvidenceReportBuilder(loader=_loader())
    cr = builder.build(analysis, tracks)
    _run_rule_coach(analysis, cr)
    st.session_state["content_locale"] = st.session_state.get("locale", "en")
    st.session_state["progress"] = 100; st.session_state["status"] = _t("run.complete")
    return cr.model_dump(mode="json")



REGION_COLORS = {
    "minimap": (0, 255, 0), "round_info": (255, 255, 0),
    "kill_feed": (255, 0, 0), "crosshair": (0, 255, 255),
    "money": (255, 0, 255), "player_status": (0, 165, 255),
    "weapon_utility": (128, 0, 255),
}
def _draw_hud_debug(pil_img, profile) -> Image.Image:
    """Draw bounding boxes and labels for all HUD regions on the image."""
    import PIL.ImageDraw, PIL.ImageFont
    img = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = PIL.ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = PIL.ImageFont.load_default()

    for region in profile.regions:
        color = REGION_COLORS.get(region.name, (255, 255, 255))
        translated_name = _t(f"hud_regions.{region.name}")
        name = translated_name if not translated_name.startswith("hud_regions.") else region.name
        x, y, rw, rh = region.to_pixel(w, h)
        # Clamp
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x)); rh = max(1, min(rh, h - y))
        x2, y2 = x + rw, y + rh
        # Draw box
        draw.rectangle([x, y, x2, y2], outline=color, width=2)
        # Draw label background
        label = name
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = y - th - 4 if y > th + 4 else y2 + 2
        draw.rectangle([x, label_y, x + tw + 6, label_y + th + 4], fill=color)
        draw.text((x + 3, label_y + 2), label, fill=(0, 0, 0), font=font)
    return img

# Sidebar
rag_enabled = False
rag_provider = "deepseek"
rag_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
rag_uploads = []
with st.sidebar:
    st.title(f"🎯 {_t('app.title')}"); st.caption(_t("app.subtitle"))
    lang_labels = {"en": "English", "zh-CN": "简体中文"}
    lang = st.selectbox(_t("sidebar.language"), options=["en", "zh-CN"],
                        format_func=lambda x: lang_labels[x], index=0 if st.session_state["locale"] == "en" else 1)
    if lang != st.session_state["locale"]:
        st.session_state["locale"] = lang
        _refresh_localized_outputs()
        st.rerun()
    st.divider()
    new_mode = st.radio(
        f"📋 {_t('sidebar.mode')}",
        ["video", "screenshot"],
        format_func=lambda value: _t(f"sidebar.mode_{value}"),
        index=0 if st.session_state.get("mode", "video") == "video" else 1,
    )
    if new_mode != st.session_state.get("mode"): st.session_state["mode"] = new_mode; st.rerun()
    st.divider()
    if st.session_state.get("mode") == "video":
        st.subheader(f"📁 {_t('sidebar.input')}")
        use_demo = st.checkbox(_t("sidebar.demo_mode"), value=False, help=_t("sidebar.demo_help"))
        uploaded = None
        if not use_demo:
            uploaded = st.file_uploader(_t("sidebar.upload_label"), type=["mp4", "mov", "mkv"], help=_t("sidebar.upload_help"))
        pf = st.checkbox(_t("sidebar.player_filter"), value=st.session_state.get("player_filter", False), help=_t("sidebar.player_filter_help"))
        st.session_state["player_filter"] = pf
        if pf:
            pn = st.text_input(_t("sidebar.player_name"), value=st.session_state.get("player_name", ""), help=_t("sidebar.player_name_help"))
            st.session_state["player_name"] = pn; st.caption(_t("player_filter.active"))
        st.divider()
        st.subheader(f"🧠 {_t('sidebar.ai_features')}")
        st.caption(_t("sidebar.ocr_automatic"))
        use_yolo = st.checkbox(_t("sidebar.yolo_label"), value=st.session_state.get("use_yolo", False), help=_t("sidebar.yolo_help"))
        st.session_state["use_yolo"] = use_yolo
        rag_enabled = st.checkbox(
            _t("rag.enable"), value=st.session_state.get("rag_enabled", False),
            help=_t("rag.enable_help"),
        )
        st.session_state["rag_enabled"] = rag_enabled
        if rag_enabled:
            rag_provider = st.selectbox(
                _t("rag.provider"), ["deepseek", "ollama"],
                index=0 if st.session_state.get("rag_provider", "deepseek") == "deepseek" else 1,
            )
            st.session_state["rag_provider"] = rag_provider
            model_default = (
                os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
                if rag_provider == "deepseek" else
                os.getenv("OLLAMA_MODEL", "qwen3:8b")
            )
            rag_model = st.text_input(
                _t("rag.model"), value=st.session_state.get(
                    f"rag_model_{rag_provider}", model_default,
                ),
            )
            st.session_state[f"rag_model_{rag_provider}"] = rag_model
            rag_uploads = st.file_uploader(
                _t("rag.knowledge_upload"), type=["md", "txt", "docx"],
                accept_multiple_files=True, key="rag_knowledge_uploads",
                help=_t("rag.knowledge_help"),
            )
            if rag_provider == "deepseek":
                st.caption(_t(
                    "rag.key_ready" if os.getenv("DEEPSEEK_API_KEY") else
                    "rag.key_missing"
                ))
            else:
                st.caption(_t("rag.ollama_hint"))
        st.divider()
        st.subheader(f"⚙️ {_t('sidebar.settings')}")
        sample_fps = st.slider(_t("sidebar.sample_fps"), 1, 30, 10, help=_t("sidebar.sample_fps_help"))
        st.caption(_t("sidebar.sampling_strategy", fps=sample_fps))
    else:
        use_demo = False; uploaded = None; use_yolo = False; pf = False; sample_fps = 10
        st.subheader(f"📸 {_t('sidebar.debug_upload_title')}")
        debug_files = st.file_uploader(_t("sidebar.debug_upload_label"), type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True, key="debug_upload")
        if debug_files:
            st.session_state["debug_screenshots"] = debug_files
            st.caption(_t("sidebar.debug_loaded", n=len(debug_files)))
    st.divider(); st.caption(
        f"{_t('app.version')} | {_t('app.test_count', n=521)}"
    )

if st.session_state.get("mode") == "screenshot":
    # ---- Screenshot Debug Mode ----
    debug_files = st.session_state.get("debug_screenshots", [])
    if debug_files:
        if st.button(f"🔍 {_t('debug.analyze')}", type="primary"):
            with st.spinner(_t("loading")):
                st.session_state["debug_results"] = []
                parser = CS2HudParser(CS2_STANDARD_16X9, {
                    "crosshair": CrosshairExtractor(), "player_status": HPBarExtractor(enable_numeric_ocr=True),
                    "kill_feed": KillFeedExtractor(), "money": MoneyExtractor(),
                    "round_info": RoundInfoExtractor(),
                })
                for f in debug_files:
                    pil = Image.open(f).convert("RGB")
                    frame = np.array(pil)[:, :, ::-1].copy()
                    state = parser.parse(frame, 0, 0.0)
                    advice = LiveAnalyzer(
                        _CachedHudParser(state), _loader(),
                    ).analyze(frame)
                    st.session_state["debug_results"].append({
                        "name": f.name, "image": pil, "state": state,
                        "advice": advice, "frame": frame,
                    })
            st.rerun()
    debug_results = st.session_state.get("debug_results", [])
    if debug_results:
        for i, dr in enumerate(debug_results):
            st.subheader(f"📸 {dr['name']}")
            c1, c2 = st.columns([1, 1])
            with c1:
                debug_img = _draw_hud_debug(dr["image"], CS2_STANDARD_16X9)
                st.image(debug_img, caption=_t("debug.hud_regions"), use_container_width=True)
            with c2:
                st.subheader(_t("debug.hud_values"))
                vals = dr["state"].values
                st.json({k: v for k, v in sorted(vals.items()) if isinstance(v, (str, int, float, bool, type(None)))})
                st.divider()
                st.subheader(f"🎯 {_t('debug.advice')}")
                a = dr["advice"]
                st.metric(_t("debug.status"), a.status)
                st.markdown(f"**{_t('debug.action')}:** {a.next_action}")
                st.caption(f"{_t('debug.confidence')}: {a.confidence:.2f}")
                for tip in a.tips:
                    st.markdown(f"- {tip}")
            st.divider()
    elif not debug_files:
        st.info(_t("debug.empty_hint"))
    st.stop()

# ---- Video Analysis Mode ----
col1, col2 = st.columns([1, 3])
with col1:
    can_run = use_demo or (uploaded is not None)
    run_clicked = st.button(f"\u25b6\ufe0f {_t('run.button')}", type="primary", use_container_width=True, disabled=not can_run)
with col2:
    if st.session_state["progress"] > 0:
        st.progress(st.session_state["progress"] / 100, text=st.session_state["status"])

if run_clicked:
    st.session_state["analysis_run"] = True; st.session_state["progress"] = 0
    st.session_state["result"] = None; st.session_state["coach_suggestions"] = None
    st.session_state["coach_summary"] = None; st.session_state["coach_diagnostics"] = None
    st.session_state["screenshots"] = None
    st.session_state["clips"] = None; st.session_state["analysis_base"] = None
    st.session_state["event_corrections"] = {}
    st.session_state["manual_events"] = []
    st.session_state["correction_revision"] += 1
    with st.spinner(_t("loading")):
        if use_demo: result = _demo_pipeline()
        else:
            upload_suffix = Path(uploaded.name).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=upload_suffix) as f:
                uploaded.seek(0)
                shutil.copyfileobj(uploaded, f, length=8 * 1024 * 1024)
                video_path = f.name
            try:
                result = _real_pipeline(
                    video_path, sample_fps, use_yolo=use_yolo,
                    source_name=uploaded.name,
                )
                extractor = OpenCVScreenshotExtractor(max_screenshots=72)
                analysis = st.session_state.get("analysis_obj")
                if analysis is not None:
                    # Always provide representative gameplay frames even when
                    # personal kill/death detection is intentionally disabled.
                    phase_frames = build_round_keyframe_events(
                        analysis, samples_per_round=3, max_events=54,
                    )
                    # Include the exact frame that triggered each viewport
                    # summary so first-person advice has visible evidence.
                    verified_events = [
                        event for round_analysis in analysis.rounds
                        for event in round_analysis.events
                        if event.event_type in (
                            EventType.FIRST_PERSON_SUMMARY,
                            EventType.FIRST_PERSON_MOMENT,
                            EventType.ENGAGEMENT_CANDIDATE,
                            EventType.PLAYER_DEATH,
                        )
                    ]
                    # Evidence-triggered moments get capacity first.  This
                    # prevents late-round gunfights from being crowded out by
                    # generic phase frames when the 72-image cap is reached.
                    important = verified_events[:72]
                    important.extend(phase_frames[:max(0, 72 - len(important))])
                    important.sort(key=lambda event: event.start_sec)
                    # Preserve room for reliable personal events when an
                    # identity-aware detector is enabled in the future.
                    combat_events = [
                        e for r in analysis.rounds for e in r.events
                        if e.event_type in (EventType.PLAYER_KILL, EventType.PLAYER_DEATH)
                        and (e.start_sec - r.start_sec) > 12.0
                    ]
                    important.extend(combat_events[:max(0, 40 - len(important))])
                    st.session_state["screenshots"] = extractor.extract(video_path, important)
                    clip_events = [
                        event for round_analysis in analysis.rounds
                        for event in round_analysis.events
                        if event.event_type == EventType.PLAYER_DEATH
                    ]
                    clip_events.extend(
                        event for round_analysis in analysis.rounds
                        for event in round_analysis.events
                        if event.event_type == EventType.ENGAGEMENT_CANDIDATE
                    )
                    clip_events.extend(
                        event for round_analysis in analysis.rounds
                        for event in round_analysis.events
                        if event.event_type == EventType.FIRST_PERSON_MOMENT
                    )
                    try:
                        st.session_state["clips"] = EvidenceClipExtractor(
                            before_sec=2.0, after_sec=3.0, max_clips=12,
                        ).extract(video_path, clip_events)
                    except (ImportError, RuntimeError, OSError) as exc:
                        st.warning(f"{_t('run.clip_unavailable')} ({exc})")
            finally: Path(video_path).unlink(missing_ok=True)
        st.session_state["result"] = result; st.rerun()

if (
    st.session_state.get("analysis_obj") is not None
    and st.session_state.get("content_locale")
    != st.session_state.get("locale", "en")
):
    _refresh_localized_outputs()

result = st.session_state.get("result"); coach_suggestions = st.session_state.get("coach_suggestions")
coach_summary = st.session_state.get("coach_summary"); screenshots = st.session_state.get("screenshots")
coach_diagnostics = st.session_state.get("coach_diagnostics")
clips = st.session_state.get("clips")

if result is None:
    if not st.session_state.get("analysis_run"):
        st.info(_t("hint_upload"))
        st.markdown(f"### {_t('how_it_works.title')}")
        st.markdown(f"1. {_t('how_it_works.step1')}"); st.markdown(f"2. {_t('how_it_works.step2')}")
        st.markdown(f"3. {_t('how_it_works.step3')}"); st.markdown(f"4. {_t('how_it_works.step4')}")
    st.stop()

t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
    f"\U0001f4ca {_t('tabs.overview')}", f"\U0001f4c5 {_t('tabs.timeline')}",
    f"\U0001f4dd {_t('tabs.report')}", f"\U0001f517 {_t('tabs.evidence')}",
    f"\U0001f9e0 {_t('tabs.coach')}", f"\U0001f4f7 {_t('live.title')}",
    f"\U0001f9ea {_t('tabs.diagnostics')}", f"\U0001f4c4 {_t('tabs.json')}",
])

# Overview
with t1:
    ov = result["overview"]
    st.subheader(_t("overview.match_overview"))
    cols = st.columns(5)
    combat_available = ov.get("personal_combat_available", True)
    kills_available = ov.get("personal_kills_available", combat_available)
    deaths_available = ov.get("personal_deaths_available", combat_available)
    analysis_complete = ov.get("analysis_complete", True)
    kills_label = _t(
        "overview.kills_lower_bound"
        if kills_available and not combat_available else "overview.kills"
    )
    cols[0].metric(_t("overview.video"), ov["video_id"]); cols[1].metric(
        _t("overview.rounds"),
        ov["total_rounds"] if analysis_complete else f"{ov['total_rounds']}*",
    )
    cols[2].metric(
        _t("overview.duration"),
        f"{ov.get('duration_sec',0):.0f}s"
        if ov.get("duration_sec") else _t("common.unavailable"),
    )
    cols[3].metric(kills_label, ov["total_kills_detected"] if kills_available else "—"); cols[4].metric(_t("overview.deaths"), ov["total_deaths_detected"] if deaths_available else "—")
    if not analysis_complete:
        st.error(_t("overview.partial_result"))
    st.divider(); st.subheader(_t("overview.round_summary"))
    rows = []
    for r in result.get("rounds", []):
        s = r["stats"]
        row = {
            _t("overview.round"): r["round_id"],
            _t("overview.duration"): f"{r.get('duration_sec',0):.1f}s" if r.get("duration_sec") else "-",
            _t("overview.flash"): f"{s.get('flash_count', 0)} / {s.get('flash_exposure_sec', 0):.1f}s",
            _t("overview.scoped"): f"{s.get('scoped_sec', 0):.1f}s",
            _t("overview.view_motion"): f"{s.get('view_motion_avg', 0):.2f}",
            _t("overview.stationary"): f"{s.get('stationary_ratio', 0) * 100:.0f}%",
            _t("overview.engagements"): (
                f"{s.get('engagement_windows', 0)} / "
                f"{s.get('likely_firefights', 0)}"
            ),
        }
        if kills_available:
            row[kills_label] = s["kills_detected"]
        if deaths_available:
            row.update({
                _t("overview.player_died"): _t(
                    "common.yes" if s.get("player_died") else "common.no"
                ),
            })
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)

# Timeline
with t2:
    st.subheader(_t("timeline.title"))
    for r in result.get("rounds", []):
        label = f"**{r['round_id']}** | {r['duration_sec']:.1f}s" if r.get("duration_sec") else f"**{r['round_id']}** | {_t('timeline.truncated')}"
        with st.expander(label, expanded=len(result["rounds"]) <= 2):
            for f in r.get("findings", []):
                sev = f["severity"]; color = {"info":"#4fc3f7","warning":"#ffb74d","critical":"#ef5350"}.get(sev, "#888")
                severity = _t(f"severity.{sev}")
                st.markdown(f"""<div style="border-left:4px solid {color};padding:.5rem 1rem;margin:.4rem 0;border-radius:0 6px 6px 0;background:#161b22">
                    <strong>[{severity}]</strong> {f['text']}<br><small style="color:#8b949e">{_t('timeline.confidence')}: {f['confidence']:.2f} | {_t('common.id')}: {f['finding_id']}</small></div>""", unsafe_allow_html=True)
                if screenshots:
                    for img in [s for s in screenshots if s.event_id == f.get("finding_id")][:1]:
                        if img.exists(): st.image(
                            str(img.image_path),
                            caption=_t(
                                "common.frame_time",
                                frame=img.frame_index,
                                time=img.timestamp_sec,
                            ),
                            width=400,
                        )
                if f.get("evidence"):
                    with st.expander(_t("timeline.evidence_links"), expanded=False):
                        for lk in f["evidence"]: st.caption(_t(
                            "common.evidence_line",
                            frame=lk.get("frame_index", "?"),
                            time=lk["timestamp_sec"],
                            source=lk["source"],
                        ))

# Report
with t3:
    st.subheader(_t("report.title")); st.markdown(f"### {_t('report.match_summary')}")
    for f in result.get("match_findings", []):
        icon = {"info":"(i)","warning":"(!)","critical":"(!!)"}.get(f["severity"], "")
        severity = _t(f"severity.{f['severity']}")
        st.markdown(f"{icon} **[{severity}]** {f['text']}")
    st.divider()
    for r in result.get("rounds", []):
        st.markdown(f"### {r['round_id']}")
        if r.get("duration_sec"): st.caption(f"{_t('report.duration_label')}: {r['duration_sec']:.1f}s")
        s = r["stats"]; c1,c2,c3,c4 = st.columns(4)
        if kills_available or deaths_available:
            c1.metric(
                _t("report.kills_label"),
                s["kills_detected"] if kills_available else "—",
            )
            c2.metric(
                _t("report.deaths_label"),
                s["deaths_detected"] if deaths_available else "—",
            )
            surv = (
                f"{s['survival_sec']:.0f}s"
                if deaths_available and s.get("survival_sec") is not None
                else _t("common.unavailable")
            )
            c3.metric(_t("report.survival_label"), surv)
            c4.metric(_t("report.enemies_label"), s.get("enemies_encountered", s.get("enemy_tracks", 0)))
        else:
            c1.metric(_t("overview.flash"), f"{s.get('flash_count', 0)} / {s.get('flash_exposure_sec', 0):.1f}s")
            c2.metric(_t("overview.scoped"), f"{s.get('scoped_sec', 0):.1f}s")
            c3.metric(_t("overview.view_motion"), f"{s.get('view_motion_avg', 0):.2f}")
            c4.metric(_t("overview.stationary"), f"{s.get('stationary_ratio', 0) * 100:.0f}%")

        for f in r.get("findings", []):
            icon = {"info":"(i)","warning":"(!)","critical":"(!!)"}.get(f["severity"], ""); st.markdown(f"{icon} {f['text']}")
        st.divider()

# Evidence
with t4:
    st.subheader(_t("evidence.title")); links = []
    for r in result.get("rounds", []):
        for f in r.get("findings", []):
            for lk in f.get("evidence", []):
                links.append({_t("evidence.round_col"): r["round_id"], _t("evidence.finding_col"): f["finding_id"],
                              _t("evidence.category_col"): _t(f"finding_categories.{f['category']}"), _t("evidence.frame_col"): lk.get("frame_index","-"),
                              _t("evidence.time_col"): f"{lk['timestamp_sec']:.1f}s", _t("evidence.source_col"): lk["source"]})
    if links: st.caption(_t("evidence.count", n=len(links))); st.dataframe(links, use_container_width=True, hide_index=True)
    else: st.info(_t("evidence.no_links"))
    if clips:
        st.divider(); st.subheader(_t("evidence.clips_title"))
        st.caption(_t("evidence.clips_help"))
        for clip in clips:
            if clip.exists():
                st.markdown(
                    f"**{clip.event_id}** · {_t('common.time')}: "
                    f"{clip.trigger_sec:.1f}s"
                )
                st.video(str(clip.video_path), format="video/mp4")

# Coach
with t5:
    st.subheader(f"\U0001f9e0 {_t('coach.title')}"); st.caption(_t("coach.subtitle"))
    analysis_for_context = st.session_state.get("analysis_obj")
    if analysis_for_context is not None:
        coverage = context_coverage(analysis_for_context.round_contexts)
        with st.expander(_t("context.title"), expanded=False):
            st.caption(_t("context.subtitle"))
            context_rows = []
            for field in (
                "player_side", "native_round_clock", "weapon", "economy",
                "utility", "map_position",
            ):
                context_rows.append({
                    _t("context.field"): _t(f"context.fields.{field}"),
                    _t("context.coverage"): (
                        f"{coverage[field]} / {coverage['rounds']}"
                    ),
                    _t("context.status"): _t(
                        "context.available" if coverage[field] else
                        "context.unavailable"
                    ),
                })
            st.dataframe(context_rows, use_container_width=True, hide_index=True)
            evidence_rows = []
            for context in analysis_for_context.round_contexts:
                if (
                    context.native_round_clock_sec is None
                    and not context.weapon_categories
                    and context.money is None
                    and not context.utility
                    and context.map_position is None
                ):
                    continue
                minutes = None
                if context.native_round_clock_sec is not None:
                    total = int(context.native_round_clock_sec)
                    minutes = f"{total // 60}:{total % 60:02d}"
                evidence_rows.append({
                    _t("context.round"): context.round_id,
                    _t("context.native_clock_value"): minutes or "—",
                    _t("context.clock_reads"): len(
                        context.round_clock_observations
                    ),
                    _t("context.weapon_categories"): (
                        ", ".join(_t(f"context.weapons.{category}")
                                  for category in context.weapon_categories)
                        if context.weapon_categories else "—"
                    ),
                    _t("context.weapon_reads"): len(context.weapon_observations),
                    _t("context.money_value"): (
                        f"${context.money}" if context.money is not None else "—"
                    ),
                    _t("context.utility_value"): (
                        ", ".join(_t(f"context.utilities.{item}")
                                  for item in context.utility)
                        if context.utility else "—"
                    ),
                    _t("context.map_position_value"): (
                        f"{_t(f'context.maps.{context.map_name}')} / "
                        f"{_t(f'context.positions.{context.map_position}')}"
                        if context.map_name and context.map_position else "—"
                    ),
                })
            if evidence_rows:
                st.markdown(f"**{_t('context.details')}**")
                st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
            st.info(_t("context.abstention"))
    if rag_enabled and analysis_for_context is not None:
        llm_client = (
            DeepSeekClient(model=rag_model) if rag_provider == "deepseek"
            else OllamaClient(model=rag_model)
        )
        can_generate_rag = (
            analysis_for_context.capabilities.get("analysis_complete", True)
            and llm_client.available
        )
        if st.button(
            _t("rag.generate"), type="primary", disabled=not can_generate_rag,
            help=_t("rag.generate_help"),
        ):
            try:
                with st.spinner(_t("rag.generating")):
                    retriever = _build_rag_retriever(
                        rag_uploads, DEFAULT_MULTILINGUAL_MODEL,
                    )
                    report_for_rag = EvidenceReportBuilder(loader=_loader()).build(
                        analysis_for_context, st.session_state.get("tracks"),
                    )
                    engine = EvidenceBoundRagCoach(
                        retriever, llm_client,
                        base_coach=RuleBasedCoach(_loader()),
                        locale=st.session_state.get("locale", "en"),
                    )
                    _store_coach_run(engine.run(analysis_for_context, report_for_rag))
                st.rerun()
            except Exception as exc:
                st.error(_t(
                    "rag.generation_failed", error=type(exc).__name__,
                ))
        if not can_generate_rag:
            unavailable_key = (
                rag_provider == "deepseek" and not llm_client.available
            )
            st.info(_t(
                "rag.unavailable_key" if unavailable_key else
                "rag.unavailable_analysis"
            ))
    if coach_diagnostics:
        with st.expander(_t("rag.diagnostics"), expanded=False):
            mode_key = "rag.mode_rag" if coach_diagnostics.mode == "rag_llm" else "rag.mode_rules"
            st.caption(_t(mode_key))
            diagnostic_rows = [
                (_t("rag.provider"), coach_diagnostics.provider or "—"),
                (_t("rag.model"), coach_diagnostics.model or "—"),
                (_t("rag.knowledge_chunks"), coach_diagnostics.knowledge_chunks),
                (_t("rag.knowledge_layers"), " · ".join(
                    f"{_t(f'rag.layers.{layer}')} {count}"
                    for layer, count in coach_diagnostics.knowledge_layers.items()
                ) or "—"),
                (_t("rag.retrieved_chunks"), coach_diagnostics.retrieved_chunks),
                (_t("rag.accepted"), coach_diagnostics.accepted_enrichments),
                (_t("rag.rejected"), coach_diagnostics.rejected_enrichments),
                (_t("rag.latency"), f"{coach_diagnostics.latency_ms} ms"),
                (_t("rag.tokens"), coach_diagnostics.total_tokens),
            ]
            st.dataframe(
                [{_t("rag.metric"): name, _t("rag.value"): value}
                 for name, value in diagnostic_rows],
                use_container_width=True, hide_index=True,
            )
            if coach_diagnostics.fallback_reason:
                st.warning(_t(
                    "rag.fallback", reason=coach_diagnostics.fallback_reason,
                ))
    if coach_summary:
        st.subheader(f"\U0001f3c6 {_t('summary_title')}")
        st.markdown(f"**{_t('assessment')}:** {coach_summary.overall_assessment}")
        ca,cb = st.columns(2)
        with ca:
            st.markdown(f"### {_t('strengths')}")
            for item in coach_summary.strengths: st.markdown(f"- {item}")
            st.markdown(f"### {_t('focus_areas')}")
            for item in coach_summary.focus_areas: st.markdown(f"- {item}")
        with cb:
            st.markdown(f"### {_t('weaknesses')}")
            for item in coach_summary.weaknesses: st.markdown(f"- {item}")
            st.markdown(f"### {_t('practice_drills')}")
            for item in coach_summary.practice_drills: st.markdown(f"- {item}")
        if coach_summary.knowledge_citations:
            with st.expander(_t("rag.citations"), expanded=False):
                for citation in coach_summary.knowledge_citations:
                    st.caption(_knowledge_citation_text(citation))
        st.divider()
    if not combat_available:
        st.info(_t("coach.first_person_coverage"))
    if not coach_suggestions: st.info(_t("coach.no_suggestions"))
    else:
        for s in coach_suggestions:
            cat_name = _t(f"coach.categories.{s.category.value}")
            with st.expander(
                f"**{cat_name}** | {_t('coach.round_label')} {s.round_id} | "
                f"{_t('common.time')}: {s.timestamp_sec:.1f}s",
                expanded=len(coach_suggestions)<=4,
            ):
                st.markdown(f"**{_t('coach.reasoning')}:** {s.reasoning}")
                st.markdown(f"**{_t('coach.action')}:** {s.action}")
                st.caption(
                    f"{_t('coach.confidence')}: {s.confidence:.2f} | "
                    f"{_t('common.id')}: {s.suggestion_id}"
                )
                if screenshots:
                    # Prefer an event-id match, otherwise show one nearby frame.
                    # A single selection prevents the same image being rendered
                    # once by each matching strategy.
                    matching_image = next((
                        img for img in screenshots
                        if img.exists() and img.event_id and (
                            img.event_id in s.suggestion_id
                            or s.suggestion_id in img.event_id
                        )
                    ), None)
                    if matching_image is None:
                        matching_image = next((
                            img for img in screenshots
                            if img.exists()
                            and abs(img.timestamp_sec - s.timestamp_sec) < 2.0
                        ), None)
                    if matching_image is not None:
                        st.image(
                            str(matching_image.image_path),
                            caption=_t(
                                "common.frame_time",
                                frame=matching_image.frame_index,
                                time=matching_image.timestamp_sec,
                            ),
                            width=400,
                        )
                if clips:
                    matching_clip = next((
                        clip for clip in clips
                        if clip.exists() and (
                            clip.event_id in s.suggestion_id
                            or abs(clip.trigger_sec - s.timestamp_sec) < 2.1
                        )
                    ), None)
                    if matching_clip is not None:
                        st.caption(_t(
                            "coach.clip_range",
                            start=matching_clip.start_sec,
                            end=matching_clip.end_sec,
                        ))
                        st.video(str(matching_clip.video_path), format="video/mp4")
                if s.evidence:
                    with st.expander(_t("coach.evidence"), expanded=False):
                        for lk in s.evidence: st.caption(_t(
                            "common.evidence_line",
                            frame=lk.frame_index or "?",
                            time=lk.timestamp_sec,
                            source=lk.source,
                        ))
                if s.knowledge_citations:
                    with st.expander(_t("rag.citations"), expanded=False):
                        for citation in s.knowledge_citations:
                            st.caption(_knowledge_citation_text(citation))

# Live
with t6:
    st.subheader(f"\U0001f4f7 {_t('live.title')}"); st.caption(_t("live.subtitle"))
    live_img = st.file_uploader(_t("live.upload_label"), type=["jpg","jpeg","png"], key="live_upload", help=_t("live.upload_help"))
    if live_img is not None:
        st.image(live_img, caption=_t("live.uploaded_caption"), width=600)
        if st.button(f"\U0001f50d {_t('live.analyze_btn')}", type="primary"):
            with st.spinner(_t("loading")):
                pil_img = Image.open(live_img).convert("RGB"); frame = np.array(pil_img)[:,:,::-1].copy()
                parser = CS2HudParser(CS2_STANDARD_16X9, {"crosshair": CrosshairExtractor(), "player_status": HPBarExtractor(enable_numeric_ocr=True),
                    "kill_feed": KillFeedExtractor(), "money": MoneyExtractor(), "round_info": RoundInfoExtractor()})
                live_state = parser.parse(frame, 0, 0.0)
                st.session_state["live_state"] = live_state
                st.session_state["live_result"] = LiveAnalyzer(
                    _CachedHudParser(live_state), _loader(),
                ).analyze(frame)
    lr = st.session_state.get("live_result")
    if lr is not None:
        st.divider(); st.subheader(f"\U0001f3af {_t('live.results')}")
        cx,cy = st.columns(2)
        with cx: st.metric(_t("live.status"), lr.status); st.markdown(f"**{_t('live.next_action')}:** {lr.next_action}")
        with cy:
            st.markdown(f"### {_t('live.tips')}")
            for tip in lr.tips: st.markdown(f"- {tip}")

# Live -- multi-frame from analyzed video
    if st.session_state.get("analysis_run") and result is not None:
        st.divider()
        st.subheader(_t("live.select_frames"))
        st.caption(_t("live.select_hint"))
        # Get timeline events with screenshots
        key_events = st.session_state.get("screenshots")
        if key_events:
            cols = st.columns(5)
            for i, img in enumerate(key_events):
                col_idx = i % 5
                if img.exists():
                    with cols[col_idx]:
                        round_label = img.event_id.removeprefix("round_keyframe_").rsplit("_", 1)[0]
                        st.image(
                            str(img.image_path),
                            caption=_t(
                                "live.frame_caption",
                                round=round_label,
                                time=img.timestamp_sec,
                            ),
                            width=150,
                        )
                        if st.button(_t("live.analyze_frame"), key=f"live_frame_{i}"):
                            pil_img = Image.open(str(img.image_path)).convert("RGB")
                            frame = np.array(pil_img)[:,:,::-1].copy()
                            parser = CS2HudParser(CS2_STANDARD_16X9, {"crosshair": CrosshairExtractor(), "player_status": HPBarExtractor(enable_numeric_ocr=True),
                                "kill_feed": KillFeedExtractor(), "money": MoneyExtractor(), "round_info": RoundInfoExtractor()})
                            live_state = parser.parse(
                                frame, img.frame_index or 0, img.timestamp_sec,
                            )
                            st.session_state["live_state"] = live_state
                            st.session_state["live_result"] = LiveAnalyzer(
                                _CachedHudParser(live_state), _loader(),
                            ).analyze(frame, timestamp_sec=img.timestamp_sec)
                            st.rerun()

# Detection diagnostics and correction labels
with t7:
    st.subheader(_t("diagnostics.title"))
    st.caption(_t("diagnostics.subtitle"))
    base_analysis = st.session_state.get("analysis_base")
    corrections = st.session_state.get("event_corrections", {})
    if base_analysis is None:
        st.info(_t("diagnostics.no_events"))
    else:
        metadata = base_analysis.analysis_metadata
        pipeline_cols = st.columns(4)
        pipeline_cols[0].metric(
            _t("diagnostics.base_fps"), metadata.get("base_scan_fps", "-"),
        )
        pipeline_cols[1].metric(
            _t("diagnostics.refine_fps"), metadata.get("requested_fps", "-"),
        )
        pipeline_cols[2].metric(
            _t("diagnostics.windows"),
            metadata.get("refinement_window_count", 0),
        )
        pipeline_cols[3].metric(
            _t("diagnostics.refined_samples"),
            metadata.get("refined_sample_count", 0),
        )
        augmented_analysis = add_manual_events(
            base_analysis, st.session_state.get("manual_events", []),
        )
        rows = diagnostic_rows(augmented_analysis, corrections)
        st.markdown(f"### {_t('diagnostics.add_event')}")
        st.caption(_t("diagnostics.add_event_help"))
        with st.form("manual_event_form", clear_on_submit=True):
            add_col1, add_col2, add_col3 = st.columns(3)
            with add_col1:
                manual_round_id = st.selectbox(
                    _t("diagnostics.round"),
                    [round_.round_id for round_ in base_analysis.rounds],
                )
            with add_col2:
                manual_type = st.selectbox(
                    _t("diagnostics.event_type"),
                    [EventType.PLAYER_KILL.value, EventType.PLAYER_DEATH.value],
                    format_func=lambda value: _t(
                        f"diagnostics.event_types.{value}"
                    ),
                )
            selected_round = next(
                round_ for round_ in base_analysis.rounds
                if round_.round_id == manual_round_id
            )
            max_elapsed = max(
                0.0,
                (selected_round.end_sec or metadata.get(
                    "processed_until_sec", selected_round.start_sec + 180.0,
                )) - selected_round.start_sec,
            )
            with add_col3:
                manual_elapsed = st.number_input(
                    _t("diagnostics.elapsed"), min_value=0.0,
                    max_value=float(max(max_elapsed, 0.1)), step=0.1,
                )
            add_submitted = st.form_submit_button(
                _t("diagnostics.add"),
            )
        if add_submitted:
            timestamp = selected_round.start_sec + manual_elapsed
            event_id = (
                f"manual_{manual_type}_{manual_round_id}_"
                f"{int(round(timestamp * 1000)):09d}"
            )
            manual_events = st.session_state.get("manual_events", [])
            if not any(item["event_id"] == event_id for item in manual_events):
                manual_events.append({
                    "event_id": event_id,
                    "round_id": manual_round_id,
                    "event_type": manual_type,
                    "timestamp_sec": timestamp,
                })
            st.session_state["manual_events"] = manual_events
            corrected = apply_event_corrections(
                add_manual_events(base_analysis, manual_events), corrections,
            )
            _rebuild_analysis_outputs(corrected)
            st.rerun()
        if not rows:
            st.info(_t("diagnostics.no_events"))
        else:
            st.markdown(f"### {_t('diagnostics.events')}")
            st.caption(_t("diagnostics.events_help"))
            correction_changed = False
            revision = st.session_state.get("correction_revision", 0)
            for row in rows:
                event_label = _t(
                    f"diagnostics.event_types.{row['event_type']}"
                )
                with st.expander(
                    f"{row['round_id']} | {event_label} | "
                    f"t={row['timestamp_sec']:.1f}s | {row['event_id']}",
                    expanded=False,
                ):
                    accepted = st.checkbox(
                        _t("diagnostics.accepted"),
                        value=bool(row["accepted"]),
                        key=f"correction_{revision}_{row['event_id']}",
                        help=_t("diagnostics.accepted_help"),
                    )
                    st.caption(
                        f"{_t('diagnostics.confidence')}: "
                        f"{row['confidence']:.2f} | "
                        f"{_t('diagnostics.method')}: {row['method'] or '-'} | "
                        f"{_t('diagnostics.source')}: {row['source'] or '-'}"
                    )
                    if accepted != bool(row["accepted"]):
                        corrections[row["event_id"]] = accepted
                        correction_changed = True
            if correction_changed:
                st.session_state["event_corrections"] = corrections
                _rebuild_analysis_outputs(apply_event_corrections(
                    augmented_analysis, corrections,
                ))
                st.rerun()
            st.success(_t(
                "diagnostics.summary",
                accepted=sum(row["accepted"] for row in diagnostic_rows(
                    augmented_analysis, corrections,
                )),
                total=len(rows),
            ))
            if st.button(_t("diagnostics.reset")):
                st.session_state["event_corrections"] = {}
                st.session_state["manual_events"] = []
                st.session_state["correction_revision"] += 1
                _rebuild_analysis_outputs(base_analysis.model_copy(deep=True))
                st.rerun()
            correction_export = build_correction_export(
                augmented_analysis, corrections,
            )
            st.download_button(
                _t("diagnostics.download"),
                data=json.dumps(correction_export, indent=2, ensure_ascii=False),
                file_name=(
                    f"corrections_{base_analysis.video.video_id}.json"
                ),
                mime="application/json",
            )

# JSON
with t8:
    st.subheader(_t("json.title"))
    st.download_button(_t("json.download_report"), data=json.dumps(result, indent=2, ensure_ascii=False),
                       file_name=f"report_{result['overview']['video_id']}.json", mime="application/json")
    with st.expander(_t("json.preview"), expanded=False): st.json(result)

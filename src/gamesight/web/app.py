"""GameSight AI for CS2 - Streamlit Web Application.

Usage: streamlit run src/gamesight/web/app.py
"""

from __future__ import annotations

import io, json, tempfile, time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from gamesight.coach.engine import RuleBasedCoach
from gamesight.domain.models import AnalysisResult, EventType, GameEvent, RoundAnalysis, VideoInput
from gamesight.events.aggregator import aggregate_events
from gamesight.events.detectors import KillEventDetector, RoundBoundaryDetector
from gamesight.events.ocr_detector import OCRRoundDetector
from gamesight.evidence.extractor import OpenCVScreenshotExtractor
from gamesight.i18n.loader import I18nLoader
from gamesight.ingestion.video_reader import OpenCVVideoReader
from gamesight.live.analyzer import LiveAnalyzer
from gamesight.perception.extractors import (
    CrosshairExtractor, HPBarExtractor, KillFeedExtractor,
    MoneyExtractor, RoundInfoExtractor,
)
from gamesight.perception.hud_parser import CS2HudParser
from gamesight.perception.hud_profiles import CS2_STANDARD_16X9
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.web.demo import generate_demo_events, generate_demo_tracks

st.set_page_config(page_title="GameSight AI", page_icon="\U0001f3af", layout="wide", initial_sidebar_state="expanded")

if "analysis_run" not in st.session_state:
    st.session_state.update({
        "analysis_run": False, "result": None, "analysis_obj": None,
        "tracks": None, "progress": 0, "status": "",
        "coach_suggestions": None, "coach_summary": None,
        "screenshots": None, "live_result": None, "locale": "en",
        "player_filter": False, "player_name": "", "use_ocr": False, "use_yolo": False,
    })

MAX_TIME_REAL = 300    # 5 min max for real video
MAX_TIME_DEMO = 60     # 1 min max for demo
OCR_EVERY_N = 30       # Only pass image to OCR every N frames
YOLO_EVERY_N = 5       # Only run YOLO every N frames


def _loader(): return I18nLoader(st.session_state.get("locale", "en"))
def _t(key: str, **kwargs) -> str: return _loader().t(key, **kwargs)


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
                          warnings=analysis.warnings), skipped


def _real_pipeline(video_path: str, sample_fps: float) -> dict:
    t0 = time.time()
    st.session_state["status"] = _t("run.reading_meta"); st.session_state["progress"] = 5
    video = VideoInput(video_id=Path(video_path).stem, path=Path(video_path))
    reader = OpenCVVideoReader(); metadata = reader.inspect(video)
    duration = metadata.duration_sec or 60
    total_est = int(duration * sample_fps)

    st.session_state["status"] = _t("run.processing", w=metadata.width or 0, h=metadata.height or 0, fps=metadata.fps or 0)
    st.session_state["progress"] = 10

    parser = CS2HudParser(CS2_STANDARD_16X9, {
        "crosshair": CrosshairExtractor(), "player_status": HPBarExtractor(),
        "kill_feed": KillFeedExtractor(), "money": MoneyExtractor(),
        "round_info": RoundInfoExtractor(),
    })

    use_ocr = st.session_state.get("use_ocr", False)
    ocr_detector = OCRRoundDetector(profile=CS2_STANDARD_16X9)
    if use_ocr and not ocr_detector.available:
        st.warning("EasyOCR not installed. Run: pip install easyocr. Falling back to heuristic detection.")
        use_ocr = False

    hud_states = []
    for i, frame in enumerate(reader.frames(video, sample_fps)):
        if time.time() - t0 > MAX_TIME_REAL:
            st.warning(f"Analysis timed out after {MAX_TIME_REAL}s. Processing partial results.")
            break
        state = parser.parse(frame.image, frame.frame_index, frame.timestamp_sec)
        # Only pass full image to OCR every OCR_EVERY_N frames (sparse sampling)
        img_for_ocr = frame.image if (use_ocr and i % OCR_EVERY_N == 0) else None
        hud_states.append((state, img_for_ocr))
        if i % 30 == 0:
            elapsed = time.time() - t0
            if i > 0:
                eta = elapsed / i * (total_est - i)
            else:
                eta = 0
            pct = min(10 + int(60 * i / max(total_est, 1)), 70)
            st.session_state["progress"] = pct
            st.session_state["status"] = f"{_t('run.processing_frame', n=i)} | ETA {eta:.0f}s"

    n_states = len(hud_states)
    st.session_state["status"] = _t("run.detecting_events", n=n_states)
    st.session_state["progress"] = 75

    rbd = RoundBoundaryDetector(); ked = KillEventDetector()
    events = []
    if use_ocr:
        for state, image in hud_states:
            if image is not None:
                events.extend(ocr_detector.update(state, image))
            events.extend(ked.update(state))
        events.extend(ocr_detector.finalize())
    else:
        for state, _ in hud_states:
            events.extend(rbd.update(state)); events.extend(ked.update(state))
        events.extend(rbd.finalize())
    events.extend(ked.finalize())

    rounds = aggregate_events(events)
    st.session_state["progress"] = 85
    analysis = AnalysisResult(video=video, metadata=metadata, rounds=rounds)
    if st.session_state.get("player_filter"):
        analysis, skipped = _filter_post_death(analysis)
        if skipped: st.session_state["status"] += f" ({_t('player_filter.skipped_frames', n=skipped)})"
    st.session_state["analysis_obj"] = analysis
    coach = RuleBasedCoach(_loader()); builder = EvidenceReportBuilder(loader=_loader())
    cr = builder.build(analysis)
    st.session_state["coach_suggestions"] = coach.generate(analysis, cr)
    st.session_state["coach_summary"] = coach.summarize(st.session_state["coach_suggestions"], analysis, cr)
    st.session_state["progress"] = 100
    elapsed_total = time.time() - t0
    st.session_state["status"] = f"{_t('run.complete')} ({elapsed_total:.0f}s)"
    return cr.model_dump(mode="json")


def _demo_pipeline() -> dict:
    from gamesight.domain.models import VideoMetadata
    st.session_state["status"] = _t("run.generating_demo"); st.session_state["progress"] = 10
    events = generate_demo_events(rounds=5); tracks = generate_demo_tracks()
    st.session_state["progress"] = 40; rounds = aggregate_events(events)
    st.session_state["progress"] = 60
    analysis = AnalysisResult(
        video=VideoInput(video_id="demo_cs2_match", path=Path("demo.mp4")),
        metadata=VideoMetadata(duration_sec=640.0, fps=30.0, width=1920, height=1080), rounds=rounds)
    if st.session_state.get("player_filter"): analysis, _ = _filter_post_death(analysis)
    st.session_state["analysis_obj"] = analysis; st.session_state["tracks"] = tracks
    coach = RuleBasedCoach(_loader()); builder = EvidenceReportBuilder(loader=_loader())
    cr = builder.build(analysis, tracks)
    st.session_state["coach_suggestions"] = coach.generate(analysis, cr)
    st.session_state["coach_summary"] = coach.summarize(st.session_state["coach_suggestions"], analysis, cr)
    st.session_state["progress"] = 100; st.session_state["status"] = _t("run.complete")
    return cr.model_dump(mode="json")


# Sidebar
with st.sidebar:
    st.title(f"\U0001f3af {_t('app.title')}"); st.caption(_t("app.subtitle")); st.divider()
    lang_labels = {"en": "English", "zh-CN": "\u7b80\u4f53\u4e2d\u6587"}
    lang = st.selectbox(_t("sidebar.language"), options=["en", "zh-CN"],
                        format_func=lambda x: lang_labels[x], index=0 if st.session_state["locale"] == "en" else 1)
    if lang != st.session_state["locale"]: st.session_state["locale"] = lang; st.rerun()
    st.divider()
    st.subheader(f"\U0001f4c1 {_t('sidebar.input')}")
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
    st.subheader(f"\U0001f9e0 {_t("sidebar.ai_features")}")
    use_ocr = st.checkbox(_t("sidebar.ocr_label"), value=st.session_state.get("use_ocr", False),
                          help=_t("sidebar.ocr_help"))
    st.session_state["use_ocr"] = use_ocr
    use_yolo = st.checkbox(_t("sidebar.yolo_label"), value=st.session_state.get("use_yolo", False),
                           help=_t("sidebar.yolo_help"))
    st.session_state["use_yolo"] = use_yolo
    st.divider()
    st.subheader(f"\u2699\ufe0f {_t('sidebar.settings')}")
    sample_fps = st.slider(_t("sidebar.sample_fps"), 1, 30, 10, help=_t("sidebar.sample_fps_help"))
    st.divider(); st.caption(f"{_t('app.version')} | 395 tests")

# Run
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
    st.session_state["coach_summary"] = None; st.session_state["screenshots"] = None
    with st.spinner(_t("loading")):
        if use_demo: result = _demo_pipeline()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(uploaded.read()); video_path = f.name
            try:
                result = _real_pipeline(video_path, sample_fps)
                extractor = OpenCVScreenshotExtractor(max_screenshots=30)
                analysis = st.session_state.get("analysis_obj")
                if analysis is not None:
                    # Skip freeze-time events (< 12s after round start).
                    important = [e for r in analysis.rounds for e in r.events
                                 if e.event_type.value in ("player_kill", "player_death")
                                 and (e.start_sec - r.start_sec) > 12.0]
                    st.session_state["screenshots"] = extractor.extract(video_path, important)
            finally: Path(video_path).unlink(missing_ok=True)
        st.session_state["result"] = result; st.rerun()

result = st.session_state.get("result"); coach_suggestions = st.session_state.get("coach_suggestions")
coach_summary = st.session_state.get("coach_summary"); screenshots = st.session_state.get("screenshots")

if result is None:
    if not st.session_state.get("analysis_run"):
        st.info(_t("hint_upload"))
        st.markdown(f"### {_t('how_it_works.title')}")
        st.markdown(f"1. {_t('how_it_works.step1')}"); st.markdown(f"2. {_t('how_it_works.step2')}")
        st.markdown(f"3. {_t('how_it_works.step3')}"); st.markdown(f"4. {_t('how_it_works.step4')}")
    st.stop()

t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    f"\U0001f4ca {_t('tabs.overview')}", f"\U0001f4c5 {_t('tabs.timeline')}",
    f"\U0001f4dd {_t('tabs.report')}", f"\U0001f517 {_t('tabs.evidence')}",
    f"\U0001f9e0 {_t('tabs.coach')}", f"\U0001f4f7 {_t('live.title')}", f"\U0001f4c4 {_t('tabs.json')}",
])

# Overview
with t1:
    ov = result["overview"]
    st.subheader(_t("overview.match_overview"))
    cols = st.columns(5)
    cols[0].metric(_t("overview.video"), ov["video_id"]); cols[1].metric(_t("overview.rounds"), ov["total_rounds"])
    cols[2].metric(_t("overview.duration"), f"{ov.get('duration_sec',0):.0f}s" if ov.get("duration_sec") else "N/A")
    cols[3].metric(_t("overview.kills"), ov["total_kills_detected"]); cols[4].metric(_t("overview.deaths"), ov["total_deaths_detected"])
    st.divider(); st.subheader(_t("overview.round_summary"))
    rows = []
    for r in result.get("rounds", []):
        s = r["stats"]
        rows.append({_t("overview.round"): r["round_id"], _t("overview.duration"): f"{r.get('duration_sec',0):.1f}s" if r.get("duration_sec") else "-",
                     _t("overview.kills"): s["kills_detected"], _t("overview.player_died"): "Yes" if s.get("player_died") else "No",
                     _t("overview.killfeed"): s["killfeed_events"], _t("overview.enemy_tracks"): s["enemy_tracks"],
                     _t("overview.first_enemy"): f"{s.get('enemy_first_visible_sec',0):.1f}s" if s.get("enemy_first_visible_sec") else "-"})
    st.dataframe(rows, use_container_width=True, hide_index=True)

# Timeline
with t2:
    st.subheader(_t("timeline.title"))
    for r in result.get("rounds", []):
        label = f"**{r['round_id']}** | {r['duration_sec']:.1f}s" if r.get("duration_sec") else f"**{r['round_id']}** | {_t('timeline.truncated')}"
        with st.expander(label, expanded=len(result["rounds"]) <= 2):
            for f in r.get("findings", []):
                sev = f["severity"]; color = {"info":"#4fc3f7","warning":"#ffb74d","critical":"#ef5350"}.get(sev, "#888")
                st.markdown(f"""<div style="border-left:4px solid {color};padding:.5rem 1rem;margin:.4rem 0;border-radius:0 6px 6px 0;background:#161b22">
                    <strong>[{sev.upper()}]</strong> {f['text']}<br><small style="color:#8b949e">{_t('timeline.confidence')}: {f['confidence']:.2f} | {f['finding_id']}</small></div>""", unsafe_allow_html=True)
                if screenshots:
                    for img in [s for s in screenshots if s.event_id == f.get("finding_id")][:1]:
                        if img.exists(): st.image(str(img.image_path), caption=f"frame {img.frame_index} | t={img.timestamp_sec:.1f}s", width=400)
                if f.get("evidence"):
                    with st.expander(_t("timeline.evidence_links"), expanded=False):
                        for lk in f["evidence"]: st.caption(f"frame={lk.get('frame_index','?')} | t={lk['timestamp_sec']:.1f}s | {lk['source']}")

# Report
with t3:
    st.subheader(_t("report.title")); st.markdown(f"### {_t('report.match_summary')}")
    for f in result.get("match_findings", []):
        icon = {"info":"(i)","warning":"(!)","critical":"(!!)"}.get(f["severity"], "")
        st.markdown(f"{icon} **[{f['severity'].upper()}]** {f['text']}")
    st.divider()
    for r in result.get("rounds", []):
        st.markdown(f"### {r['round_id']}")
        if r.get("duration_sec"): st.caption(f"{_t('report.duration_label')}: {r['duration_sec']:.1f}s")
        s = r["stats"]; c1,c2,c3,c4 = st.columns(4)
        c1.metric(_t("report.kills_label"), s["kills_detected"]); c2.metric(_t("report.deaths_label"), s["deaths_detected"])
        surv = f"{s["survival_sec"]:.0f}s" if s.get("survival_sec") else "N/A"; c3.metric(_t("report.survival_label"), surv); c4.metric(_t("report.enemies_label"), s.get("enemies_encountered", s.get("enemy_tracks", 0)))

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
                              _t("evidence.category_col"): f["category"], _t("evidence.frame_col"): lk.get("frame_index","-"),
                              _t("evidence.time_col"): f"{lk['timestamp_sec']:.1f}s", _t("evidence.source_col"): lk["source"]})
    if links: st.caption(_t("evidence.count", n=len(links))); st.dataframe(links, use_container_width=True, hide_index=True)
    else: st.info(_t("evidence.no_links"))

# Coach
with t5:
    st.subheader(f"\U0001f9e0 {_t('coach.title')}"); st.caption(_t("coach.subtitle"))
    if not coach_suggestions: st.info(_t("coach.no_suggestions"))
    else:
        for s in coach_suggestions:
            cat_name = _t(f"coach.categories.{s.category.value}")
            with st.expander(f"**{cat_name}** | {_t('coach.round_label')} {s.round_id} | t={s.timestamp_sec:.1f}s", expanded=len(coach_suggestions)<=4):
                st.markdown(f"**{_t('coach.reasoning')}:** {s.reasoning}")
                st.markdown(f"**{_t('coach.action')}:** {s.action}")
                st.caption(f"{_t('coach.confidence')}: {s.confidence:.2f} | id: {s.suggestion_id}")
                if screenshots:
                    for img in screenshots:
                        # Prefer exact event_id match; fall back to timestamp proximity.
                        if img.event_id and (img.event_id in s.suggestion_id or s.suggestion_id in img.event_id) and img.exists():
                            st.image(str(img.image_path), caption=f"frame {img.frame_index} | t={img.timestamp_sec:.1f}s", width=400); break
                    # Second pass: timestamp proximity fallback.
                    for img in screenshots:
                        if abs(img.timestamp_sec - s.timestamp_sec) < 2.0 and img.exists():
                            st.image(str(img.image_path), caption=f"frame {img.frame_index} | t={img.timestamp_sec:.1f}s", width=400); break
                if s.evidence:
                    with st.expander(_t("coach.evidence"), expanded=False):
                        for lk in s.evidence: st.caption(f"frame={lk.frame_index or '?'} | t={lk.timestamp_sec:.1f}s | {lk.source}")
    if coach_summary:
        st.divider(); st.subheader(f"\U0001f3c6 {_t("summary_title")}")
        st.markdown(f"**{_t("assessment")}:** {coach_summary.overall_assessment}")
        ca,cb = st.columns(2)
        with ca:
            st.markdown(f"### {_t("strengths")}")
            for item in coach_summary.strengths: st.markdown(f"- {item}")
            st.markdown(f"### {_t("focus_areas")}")
            for item in coach_summary.focus_areas: st.markdown(f"- {item}")
        with cb:
            st.markdown(f"### {_t("weaknesses")}")
            for item in coach_summary.weaknesses: st.markdown(f"- {item}")
            st.markdown(f"### {_t("practice_drills")}")
            for item in coach_summary.practice_drills: st.markdown(f"- {item}")

# Live
with t6:
    st.subheader(f"\U0001f4f7 {_t('live.title')}"); st.caption(_t("live.subtitle"))
    live_img = st.file_uploader(_t("live.upload_label"), type=["jpg","jpeg","png"], key="live_upload", help=_t("live.upload_help"))
    if live_img is not None:
        st.image(live_img, caption="Uploaded screenshot", width=600)
        if st.button(f"\U0001f50d {_t('live.analyze_btn')}", type="primary"):
            with st.spinner(_t("loading")):
                pil_img = Image.open(live_img).convert("RGB"); frame = np.array(pil_img)[:,:,::-1].copy()
                parser = CS2HudParser(CS2_STANDARD_16X9, {"crosshair": CrosshairExtractor(), "player_status": HPBarExtractor(),
                    "kill_feed": KillFeedExtractor(), "money": MoneyExtractor(), "round_info": RoundInfoExtractor()})
                st.session_state["live_result"] = LiveAnalyzer(parser, _loader()).analyze(frame)
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
            for i, img in enumerate(key_events[:10]):
                col_idx = i % 5
                if img.exists():
                    with cols[col_idx]:
                        st.image(str(img.image_path), caption=f"t={img.timestamp_sec:.1f}s", width=150)
                        if st.button(f"Analyze", key=f"live_frame_{i}"):
                            pil_img = Image.open(str(img.image_path)).convert("RGB")
                            frame = np.array(pil_img)[:,:,::-1].copy()
                            parser = CS2HudParser(CS2_STANDARD_16X9, {"crosshair": CrosshairExtractor(), "player_status": HPBarExtractor(),
                                "kill_feed": KillFeedExtractor(), "money": MoneyExtractor(), "round_info": RoundInfoExtractor()})
                            st.session_state["live_result"] = LiveAnalyzer(parser, _loader()).analyze(frame, timestamp_sec=img.timestamp_sec)
                            st.rerun()

# JSON
with t7:
    st.subheader(_t("json.title"))
    st.download_button(_t("json.download_report"), data=json.dumps(result, indent=2, ensure_ascii=False),
                       file_name=f"report_{result['overview']['video_id']}.json", mime="application/json")
    with st.expander(_t("json.preview"), expanded=False): st.json(result)

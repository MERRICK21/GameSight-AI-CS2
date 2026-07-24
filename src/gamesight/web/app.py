"""GameSight AI for CS2 鈥?Streamlit Web Application.

Usage
-----
.. code-block:: bash

    streamlit run src/gamesight/web/app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from gamesight.coach.engine import RuleBasedCoach
from gamesight.domain.models import AnalysisResult, VideoInput
from gamesight.events.aggregator import aggregate_events
from gamesight.events.detectors import KillEventDetector, RoundBoundaryDetector
from gamesight.evidence.extractor import OpenCVScreenshotExtractor
from gamesight.i18n.loader import I18nLoader
from gamesight.ingestion.video_reader import OpenCVVideoReader
from gamesight.perception.extractors import (
    CrosshairExtractor,
    HPBarExtractor,
    KillFeedExtractor,
    MoneyExtractor,
    RoundInfoExtractor,
)
from gamesight.perception.hud_parser import CS2HudParser
from gamesight.perception.hud_profiles import CS2_STANDARD_16X9
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.web.demo import generate_demo_events, generate_demo_tracks

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GameSight AI",
    page_icon="馃幆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

DEFAULTS = {
    "analysis_run": False,
    "result": None,
    "analysis_obj": None,
    "tracks": None,
    "progress": 0,
    "status": "",
    "coach_suggestions": None,
    "screenshots": None,
    "locale": "en",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# i18n helper
# ---------------------------------------------------------------------------

def _t(key: str, **kwargs) -> str:
    locale = st.session_state.get("locale", "en")
    loader = I18nLoader(locale)
    return loader.t(key, **kwargs)

# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_real_pipeline(video_path: str, sample_fps: float) -> dict:
    st.session_state["status"] = _t("run.reading_meta")
    st.session_state["progress"] = 5

    video = VideoInput(video_id=Path(video_path).stem, path=Path(video_path))
    reader = OpenCVVideoReader()
    metadata = reader.inspect(video)

    st.session_state["status"] = _t("run.processing", w=metadata.width or 0, h=metadata.height or 0, fps=metadata.fps or 0)
    st.session_state["progress"] = 10

    parser = CS2HudParser(extractors=[
        CrosshairExtractor(), HPBarExtractor(), KillFeedExtractor(),
        MoneyExtractor(), RoundInfoExtractor(),
    ])

    hud_states = []
    total_frames = int((metadata.duration_sec or 60) * sample_fps)
    count = 0
    for frame in reader.frames(video, sample_fps):
        state = parser.parse(frame.image, frame.frame_index, frame.timestamp_sec)
        hud_states.append(state)
        count += 1
        if count % 30 == 0:
            pct = min(10 + int(60 * count / max(total_frames, 1)), 70)
            st.session_state["progress"] = pct
            st.session_state["status"] = _t("run.processing_frame", n=count)

    st.session_state["status"] = _t("run.detecting_events", n=len(hud_states))
    st.session_state["progress"] = 75

    rbd = RoundBoundaryDetector()
    ked = KillEventDetector()
    events = []
    for state in hud_states:
        events.extend(rbd.update(state))
        events.extend(ked.update(state))
    events.extend(rbd.finalize())
    events.extend(ked.finalize())

    rounds = aggregate_events(events)
    st.session_state["progress"] = 85
    analysis = AnalysisResult(video=video, metadata=metadata, rounds=rounds)
    st.session_state["analysis_obj"] = analysis

    st.session_state["status"] = _t("run.building")
    coach = RuleBasedCoach()
    report_builder = EvidenceReportBuilder()
    st.session_state["coach_suggestions"] = coach.generate(analysis, report_builder.build(analysis))

    report = report_builder.build(analysis)
    st.session_state["progress"] = 100
    st.session_state["status"] = _t("run.complete")
    return report.model_dump(mode="json")


def _run_demo_pipeline() -> dict:
    from gamesight.domain.models import VideoMetadata

    st.session_state["status"] = _t("run.generating_demo")
    st.session_state["progress"] = 10

    events = generate_demo_events(rounds=5)
    tracks = generate_demo_tracks()

    st.session_state["progress"] = 40
    rounds = aggregate_events(events)

    st.session_state["progress"] = 60
    analysis = AnalysisResult(
        video=VideoInput(video_id="demo_cs2_match", path=Path("demo.mp4")),
        metadata=VideoMetadata(duration_sec=640.0, fps=30.0, width=1920, height=1080),
        rounds=rounds,
    )
    st.session_state["analysis_obj"] = analysis
    st.session_state["tracks"] = tracks

    report_builder = EvidenceReportBuilder()
    coach = RuleBasedCoach()
    st.session_state["coach_suggestions"] = coach.generate(analysis, report_builder.build(analysis, tracks))

    report = report_builder.build(analysis, tracks)
    st.session_state["progress"] = 100
    st.session_state["status"] = _t("run.complete")
    return report.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title(f"馃幆 {_t('app.title')}")
    st.caption(_t("app.subtitle"))

    st.divider()

    lang = st.selectbox(
        _t("sidebar.language"),
        options=["en", "zh-CN"],
        format_func=lambda x: {"en": "English", "zh-CN": "简体中文"locale"] == "en" else 1,
    )
    if lang != st.session_state["locale"]:
        st.session_state["locale"] = lang
        st.rerun()

    st.divider()
    st.subheader(f"馃搧 {_t('sidebar.input')}")

    use_demo = st.checkbox(_t("sidebar.demo_mode"), value=False, help=_t("sidebar.demo_help"))

    uploaded = None
    if not use_demo:
        uploaded = st.file_uploader(
            _t("sidebar.upload_label"),
            type=["mp4", "mov", "mkv"],
            help=_t("sidebar.upload_help"),
        )

    st.divider()
    st.subheader(f"鈿欙笍 {_t('sidebar.settings')}")

    sample_fps = st.slider(
        _t("sidebar.sample_fps"),
        min_value=1, max_value=30, value=10, step=1,
        help=_t("sidebar.sample_fps_help"),
    )

    st.divider()
    st.caption(f"{_t('app.version')} 路 356 tests")

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 3])

with col1:
    can_run = use_demo or (uploaded is not None)
    run_clicked = st.button(
        f"鈻讹笍 {_t('run.button')}",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

with col2:
    if st.session_state["progress"] > 0:
        st.progress(st.session_state["progress"] / 100, text=st.session_state["status"])

if run_clicked:
    st.session_state["analysis_run"] = True
    st.session_state["progress"] = 0
    st.session_state["result"] = None
    st.session_state["coach_suggestions"] = None
    st.session_state["screenshots"] = None

    with st.spinner(_t("loading")):
        if use_demo:
            result = _run_demo_pipeline()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(uploaded.read())
                video_path = f.name
            try:
                result = _run_real_pipeline(video_path, sample_fps)
                # Extract screenshots for important events
                extractor = OpenCVScreenshotExtractor(max_screenshots=30)
                analysis = st.session_state.get("analysis_obj")
                if analysis is not None:
                    important = [e for r in analysis.rounds for e in r.events
                                 if e.event_type.value in ("player_kill", "player_death", "round_start", "enemy_first_visible")]
                    st.session_state["screenshots"] = extractor.extract(video_path, important)
            finally:
                Path(video_path).unlink(missing_ok=True)

        st.session_state["result"] = result
        st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.get("result")
coach_suggestions = st.session_state.get("coach_suggestions")
screenshots = st.session_state.get("screenshots")

if result is None:
    if not st.session_state.get("analysis_run"):
        st.info(_t("hint_upload"))
        st.markdown(f"### {_t('how_it_works.title')}")
        st.markdown(f"1. {_t('how_it_works.step1')}")
        st.markdown(f"2. {_t('how_it_works.step2')}")
        st.markdown(f"3. {_t('how_it_works.step3')}")
        st.markdown(f"4. {_t('how_it_works.step4')}")
    st.stop()

# ---- Tabs ----
tabs = [
    f"馃搳 {_t('tabs.overview')}", f"馃搮 {_t('tabs.timeline')}", f"馃摑 {_t('tabs.report')}",
    f"馃敆 {_t('tabs.evidence')}", f"馃 {_t('tabs.coach')}", f"馃搫 {_t('tabs.json')}",
]
t1, t2, t3, t4, t5, t6 = st.tabs(tabs)

# ===== Overview =====
with t1:
    ov = result["overview"]
    st.subheader(_t("overview.match_overview"))
    cols = st.columns(5)
    cols[0].metric(_t("overview.video"), ov["video_id"])
    cols[1].metric(_t("overview.rounds"), ov["total_rounds"])
    cols[2].metric(_t("overview.duration"), f"{ov.get('duration_sec',0):.0f}s" if ov.get("duration_sec") else "N/A")
    cols[3].metric(_t("overview.kills"), ov["total_kills_detected"])
    cols[4].metric(_t("overview.deaths"), ov["total_deaths_detected"])

    st.divider()
    st.subheader(_t("overview.round_summary"))
    rows = []
    for r in result.get("rounds", []):
        s = r["stats"]
        rows.append({
            _t("overview.round"): r["round_id"],
            _t("overview.duration"): f"{r.get('duration_sec',0):.1f}s" if r.get("duration_sec") else "鈥?,
            _t("overview.kills"): s["kills_detected"],
            _t("overview.deaths"): s["deaths_detected"],
            _t("overview.killfeed"): s["killfeed_events"],
            _t("overview.enemy_tracks"): s["enemy_tracks"],
            _t("overview.first_enemy"): f"{s.get('enemy_first_visible_sec',0):.1f}s" if s.get("enemy_first_visible_sec") else "鈥?,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

# ===== Timeline =====
with t2:
    st.subheader(_t("timeline.title"))
    for r in result.get("rounds", []):
        label = f"**{r['round_id']}** 路 {r['duration_sec']:.1f}s" if r.get("duration_sec") else f"**{r['round_id']}** 路 {_t('timeline.truncated')}"
        with st.expander(label, expanded=len(result["rounds"]) <= 2):
            for f in r.get("findings", []):
                sev = f["severity"]
                color = {"info": "#4fc3f7", "warning": "#ffb74d", "critical": "#ef5350"}.get(sev, "#888")
                st.markdown(
                    f"""<div style="border-left:4px solid {color};padding:.5rem 1rem;margin:.4rem 0;border-radius:0 6px 6px 0;background:#161b22">
                    <strong>[{sev.upper()}]</strong> {f['text']}<br>
                    <small style="color:#8b949e">{_t('timeline.confidence')}: {f['confidence']:.2f} 路 {f['finding_id']}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )
                # Show screenshot if available
                if screenshots:
                    matching = [s for s in screenshots if s.event_id == f.get("finding_id") or s.event_id in [lk.get("source","") for lk in f.get("evidence",[])]]
                    for img in matching[:1]:
                        if img.exists():
                            st.image(str(img.image_path), caption=f"{_t('timeline.frame')} {img.frame_index} 路 t={img.timestamp_sec:.1f}s", width=400)
                if f.get("evidence"):
                    with st.expander(_t("timeline.evidence_links"), expanded=False):
                        for lk in f["evidence"]:
                            st.caption(f"{_t('timeline.frame')}={lk.get('frame_index','?')} 路 t={lk['timestamp_sec']:.1f}s 路 {lk['source']}")

# ===== Report =====
with t3:
    st.subheader(_t("report.title"))
    st.markdown(f"### {_t('report.match_summary')}")
    for f in result.get("match_findings", []):
        sev = f["severity"]
        icon = {"info": "鈩癸笍", "warning": "鈿狅笍", "critical": "馃毃"}.get(sev, "")
        st.markdown(f"{icon} **[{sev.upper()}]** {f['text']}")
    st.divider()

    for r in result.get("rounds", []):
        st.markdown(f"### {r['round_id']}")
        if r.get("duration_sec"):
            st.caption(f"{_t('report.duration_label')}: {r['duration_sec']:.1f}s")
        s = r["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(_t("report.kills_label"), s["kills_detected"])
        c2.metric(_t("report.deaths_label"), s["deaths_detected"])
        c3.metric(_t("report.enemy_tracks_label"), s["enemy_tracks"])
        c4.metric(_t("report.first_enemy_label"), f"{s.get('enemy_first_visible_sec',0):.1f}s" if s.get("enemy_first_visible_sec") else "N/A")
        for f in r.get("findings", []):
            icon = {"info": "鈩癸笍", "warning": "鈿狅笍", "critical": "馃毃"}.get(f["severity"], "")
            st.markdown(f"{icon} {f['text']}")
        st.divider()

# ===== Evidence =====
with t4:
    st.subheader(_t("evidence.title"))
    links = []
    for r in result.get("rounds", []):
        for f in r.get("findings", []):
            for lk in f.get("evidence", []):
                links.append({
                    _t("evidence.round_col"): r["round_id"],
                    _t("evidence.finding_col"): f["finding_id"],
                    _t("evidence.category_col"): f["category"],
                    _t("evidence.frame_col"): lk.get("frame_index", "鈥?),
                    _t("evidence.time_col"): f"{lk['timestamp_sec']:.1f}s",
                    _t("evidence.source_col"): lk["source"],
                })
    if links:
        st.caption(_t("evidence.count", n=len(links)))
        st.dataframe(links, use_container_width=True, hide_index=True)
    else:
        st.info(_t("evidence.no_links"))

# ===== AI Coach =====
with t5:
    st.subheader(f"馃 {_t('coach.title')}")
    st.caption(_t("coach.subtitle"))

    if not coach_suggestions:
        st.info(_t("coach.no_suggestions"))
    else:
        for s in coach_suggestions:
            cat_name = _t(f"coach.categories.{s.category.value}")
            with st.expander(f"**{cat_name}** 鈥?{_t('coach.round_label')} {s.round_id} 路 t={s.timestamp_sec:.1f}s", expanded=len(coach_suggestions) <= 4):
                st.markdown(f"**{_t('coach.reasoning')}:** {s.reasoning}")
                st.markdown(f"**{_t('coach.action')}:** {s.action}")
                st.caption(f"{_t('coach.confidence')}: {s.confidence:.2f} 路 id: {s.suggestion_id}")

                # Show screenshot if available
                if screenshots:
                    for img in screenshots:
                        if abs(img.timestamp_sec - s.timestamp_sec) < 1.0 and img.exists():
                            st.image(str(img.image_path), caption=f"{_t('timeline.frame')} {img.frame_index} 路 t={img.timestamp_sec:.1f}s", width=400)
                            break

                if s.evidence:
                    with st.expander(_t("coach.evidence"), expanded=False):
                        for lk in s.evidence:
                            st.caption(f"{_t('timeline.frame')}={lk.frame_index or '?'} 路 t={lk.timestamp_sec:.1f}s 路 {lk.source}")

# ===== JSON =====
with t6:
    st.subheader(_t("json.title"))
    st.download_button(
        _t("json.download_report"),
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"report_{result['overview']['video_id']}.json",
        mime="application/json",
    )
    with st.expander(_t("json.preview"), expanded=False):
        st.json(result)

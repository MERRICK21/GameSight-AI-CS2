"""GameSight AI for CS2 — Streamlit Demo Application.

Usage
-----
.. code-block:: bash

    streamlit run src/gamesight/web/app.py

The app provides a full GUI for the analysis pipeline:
1. Upload a CS2 POV recording (or use demo mode)
2. Configure pipeline parameters
3. Run analysis with live progress
4. Browse results: timeline, evidence report, raw JSON
5. Export results to file
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from gamesight.domain.models import AnalysisResult, Track
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.serialization.timeline import TimelineBuilder
from gamesight.web.demo import generate_demo_events, generate_demo_tracks, run_demo_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GameSight AI — CS2",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .finding-card {
        border-left: 4px solid #555;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        background: #1a1a2e;
    }
    .finding-card.info { border-left-color: #4fc3f7; }
    .finding-card.warning { border-left-color: #ffb74d; }
    .finding-card.critical { border-left-color: #ef5350; }
    .evidence-link {
        font-family: monospace;
        font-size: 0.8rem;
        color: #888;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

DEFAULTS = {
    "analysis_run": False,
    "analysis_result": None,
    "match_timeline": None,
    "match_report": None,
    "tracks": None,
    "progress": 0,
    "status": "",
    "demo_video_path": None,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Pipeline runner adapter (with streamlit progress)
# ---------------------------------------------------------------------------

def _run_with_progress(video_path: str, sample_fps: float):
    """Thin wrapper that adds streamlit progress updates."""
    st.session_state["status"] = "Generating events..."
    st.session_state["progress"] = 10

    analysis, tracks = run_demo_pipeline(video_path, sample_fps)
    st.session_state["progress"] = 60

    st.session_state["status"] = "Building timeline & report..."
    tl = TimelineBuilder().build(analysis, tracks)
    st.session_state["match_timeline"] = tl

    report = EvidenceReportBuilder().build(analysis, tracks)
    st.session_state["match_report"] = report

    st.session_state["analysis_result"] = analysis
    st.session_state["tracks"] = tracks
    st.session_state["progress"] = 100
    st.session_state["status"] = "Complete!"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict:
    """Render the sidebar and return configuration dict."""
    with st.sidebar:
        st.title("🎯 GameSight AI")
        st.caption("CS2 POV Analysis Pipeline")

        st.divider()

        st.subheader("📁 Input")
        use_demo = st.checkbox("Demo Mode (synthetic data)", value=True)

        uploaded = None
        if not use_demo:
            uploaded = st.file_uploader(
                "Upload CS2 recording",
                type=["mp4", "mov", "mkv"],
                help="Select a CS2 POV gameplay recording.",
            )

        st.divider()

        st.subheader("⚙️ Pipeline")
        sample_fps = st.slider(
            "Analysis FPS",
            min_value=1, max_value=30, value=10, step=1,
            help="Frames sampled per second for HUD parsing and detection.",
        )

        st.divider()

        st.subheader("🎯 Detection & Tracking")
        enable_detection = st.checkbox("Object Detection (YOLO)", value=False, disabled=True,
                                       help="Requires YOLO model file.")
        enable_tracking = st.checkbox("Tracking (IOU)", value=False, disabled=True,
                                      help="Requires detection to be enabled.")

        st.divider()

        st.caption("GameSight v0.1.0 · Sprint 8")

    return {
        "demo": use_demo,
        "uploaded": uploaded,
        "sample_fps": float(sample_fps),
        "detection": enable_detection,
        "tracking": enable_tracking,
    }


# ---------------------------------------------------------------------------
# Run button + progress
# ---------------------------------------------------------------------------

def _render_run_section(config: dict):
    """Render the Run button and progress bar."""
    col1, col2 = st.columns([1, 3])

    with col1:
        run_clicked = st.button(
            "▶️ Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not config["demo"] and config["uploaded"] is None,
        )

    with col2:
        if 0 < st.session_state["progress"] < 100:
            st.progress(st.session_state["progress"] / 100, text=st.session_state["status"])
        elif st.session_state["progress"] == 100:
            st.success("✅ Analysis complete!")

    if run_clicked:
        st.session_state["progress"] = 0
        st.session_state["analysis_run"] = True

        with st.spinner("Running analysis pipeline..."):
            if config["demo"]:
                video_path = "demo_cs2_match.mp4"
            elif config["uploaded"] is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(config["uploaded"].read())
                    video_path = f.name
            else:
                st.error("No video provided.")
                return

            _run_with_progress(video_path, config["sample_fps"])
            st.rerun()


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

def _render_overview():
    """Render the match overview tab."""
    report = st.session_state.get("match_report")
    if report is None:
        st.info("Run the analysis to see results.")
        return

    overview = report.overview

    st.subheader("📊 Match Overview")

    cols = st.columns(5)
    cols[0].metric("🎬 Video ID", overview.video_id)
    cols[1].metric("🔄 Rounds", overview.total_rounds)
    cols[2].metric("⏱️ Duration", f"{overview.duration_sec:.0f}s" if overview.duration_sec else "N/A")
    cols[3].metric("🎯 Kills", overview.total_kills_detected)
    cols[4].metric("💀 Deaths", overview.total_deaths_detected)

    st.divider()

    cols2 = st.columns(3)
    cols2[0].metric("👁️ Enemy Tracks", overview.total_enemy_tracks)
    cols2[1].metric("📐 Resolution", f"{overview.resolution.get('width', '?')}×{overview.resolution.get('height', '?')}")
    cols2[2].metric("🎞️ FPS", f"{overview.fps:.0f}" if overview.fps else "N/A")

    st.divider()
    st.subheader("📋 Round Summary")

    round_data = []
    for rr in report.rounds:
        s = rr.stats
        round_data.append({
            "Round": rr.round_id,
            "Duration": f"{rr.duration_sec:.1f}s" if rr.duration_sec else "—",
            "Kills": s.kills_detected,
            "Deaths": s.deaths_detected,
            "Killfeed": s.killfeed_events,
            "Enemy Tracks": s.enemy_tracks,
            "1st Enemy": f"{s.enemy_first_visible_sec:.1f}s" if s.enemy_first_visible_sec else "—",
        })

    st.dataframe(round_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Timeline tab
# ---------------------------------------------------------------------------

def _render_timeline():
    """Render the timeline tab showing all events chronologically."""
    report = st.session_state.get("match_report")
    if report is None:
        st.info("Run the analysis to see results.")
        return

    st.subheader("📅 Event Timeline")

    for rr in report.rounds:
        label = f"**{rr.round_id}**  ·  {rr.duration_sec:.1f}s" if rr.duration_sec else f"**{rr.round_id}**  ·  truncated"
        with st.expander(label, expanded=len(report.rounds) <= 2):
            if not rr.findings:
                st.caption("No findings for this round.")
                continue

            for f in rr.findings:
                severity_class = f.severity.value
                st.markdown(
                    f"""<div class="finding-card {severity_class}">
                        <strong>[{f.severity.value.upper()}]</strong> {f.text}
                        <br><span class="evidence-link">confidence: {f.confidence:.2f} · id: {f.finding_id}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if f.evidence:
                    with st.expander("🔗 Evidence", expanded=False):
                        for link in f.evidence:
                            st.caption(
                                f"frame={link.frame_index or '?'}  ·  "
                                f"t={link.timestamp_sec:.2f}s  ·  "
                                f"source={link.source}"
                            )


# ---------------------------------------------------------------------------
# Report tab
# ---------------------------------------------------------------------------

def _render_report():
    """Render the evidence report as a structured document."""
    report = st.session_state.get("match_report")
    if report is None:
        st.info("Run the analysis to see results.")
        return

    st.subheader("📝 Evidence Report")

    # Match-level findings
    st.markdown("### Match Summary")
    for f in report.match_findings:
        severity_class = f.severity.value
        st.markdown(
            f"""<div class="finding-card {severity_class}">
                <strong>[{f.severity.value.upper()}]</strong> {f.text}
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    for rr in report.rounds:
        st.markdown(f"### Round {rr.round_id}")
        if rr.duration_sec:
            st.caption(f"Duration: {rr.duration_sec:.1f}s")
        else:
            st.caption("Truncated round — video ended mid-round")

        s = rr.stats
        cols = st.columns(4)
        cols[0].metric("Kills", s.kills_detected)
        cols[1].metric("Deaths", s.deaths_detected)
        cols[2].metric("Enemy Tracks", s.enemy_tracks)
        cols[3].metric("1st Enemy At", f"{s.enemy_first_visible_sec:.1f}s" if s.enemy_first_visible_sec else "N/A")

        for f in rr.findings:
            sev = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(f.severity.value, "")
            st.markdown(f"{sev} {f.text}")
            if f.evidence:
                with st.expander("Evidence links"):
                    for link in f.evidence:
                        st.code(
                            f"frame={link.frame_index} t={link.timestamp_sec:.1f}s src={link.source}",
                            language=None,
                        )

        st.divider()


# ---------------------------------------------------------------------------
# Evidence tab
# ---------------------------------------------------------------------------

def _render_evidence():
    """Render all evidence links across the report in one view."""
    report = st.session_state.get("match_report")
    if report is None:
        st.info("Run the analysis to see results.")
        return

    st.subheader("🔗 Evidence Explorer")

    all_links: list[dict] = []
    for rr in report.rounds:
        for f in rr.findings:
            for link in f.evidence:
                all_links.append({
                    "Round": rr.round_id,
                    "Finding": f.finding_id,
                    "Category": f.category.value,
                    "Frame": link.frame_index or "—",
                    "Timestamp": f"{link.timestamp_sec:.2f}s",
                    "Source": link.source,
                    "Asset": link.asset_path or "—",
                })

    if not all_links:
        st.info("No evidence links in the report.")
        return

    st.caption(f"{len(all_links)} evidence links across all rounds")
    st.dataframe(all_links, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Raw JSON tab
# ---------------------------------------------------------------------------

def _render_raw_json():
    """Render raw JSON exports of timeline and report."""
    timeline = st.session_state.get("match_timeline")
    report = st.session_state.get("match_report")

    if timeline is None and report is None:
        st.info("Run the analysis to see results.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Timeline JSON")
        if timeline is not None:
            tl_json = timeline.model_dump(mode="json")
            st.download_button(
                "⬇️ Download Timeline",
                data=json.dumps(tl_json, indent=2, ensure_ascii=False),
                file_name=f"timeline_{timeline.video_id}.json",
                mime="application/json",
            )
            with st.expander("Preview", expanded=False):
                st.json(tl_json)
        else:
            st.caption("No timeline available.")

    with col2:
        st.subheader("📝 Report JSON")
        if report is not None:
            rpt_json = report.model_dump(mode="json")
            st.download_button(
                "⬇️ Download Report",
                data=json.dumps(rpt_json, indent=2, ensure_ascii=False, default=str),
                file_name=f"report_{report.overview.video_id}.json",
                mime="application/json",
            )
            with st.expander("Preview", expanded=False):
                st.json(rpt_json)
        else:
            st.caption("No report available.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit entry point."""
    st.title("GameSight AI — CS2 Analysis")
    st.caption("Upload a CS2 POV recording and get an evidence-grounded match report.")

    config = _render_sidebar()
    _render_run_section(config)

    if st.session_state.get("analysis_run"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📅 Timeline", "📝 Report", "🔗 Evidence", "📄 Raw JSON"
        ])

        with tab1:
            _render_overview()

        with tab2:
            _render_timeline()

        with tab3:
            _render_report()

        with tab4:
            _render_evidence()

        with tab5:
            _render_raw_json()

    else:
        st.info("👈 Configure the pipeline in the sidebar and click **Run Analysis** to begin.")
        st.markdown("""
        ### What GameSight does

        1. **Ingests** your CS2 POV recording
        2. **Parses HUD** — crosshair, HP, armour, kill feed, money, round info
        3. **Detects players** with YOLO and classifies enemy/teammate
        4. **Tracks** players across frames with IOU matching
        5. **Detects events** — round boundaries, kills, deaths, enemy encounters
        6. **Aggregates** events into a structured timeline
        7. **Generates** an evidence-grounded report with traceable findings
        """)


if __name__ == "__main__":
    main()

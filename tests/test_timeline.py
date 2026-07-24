"""Unit tests for timeline serialisation — models, builder, and exporter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from gamesight.domain.models import (
    AnalysisResult,
    Detection,
    EventType,
    Evidence,
    GameEvent,
    RoundAnalysis,
    Track,
    VideoInput,
    VideoMetadata,
)
from gamesight.serialization.exporter import (
    TimelineExporter,
    export_timeline,
    timeline_to_json,
)
from gamesight.serialization.timeline import (
    EvidenceRef,
    MatchTimeline,
    RoundTimeline,
    TimelineBuilder,
    TimelineEvent,
    TrackSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    event_type: EventType,
    ts: float,
    round_id: str | None = None,
    event_id: str | None = None,
    end_sec: float | None = None,
) -> GameEvent:
    attrs: dict[str, str | int | float | bool | None] = {}
    if round_id is not None:
        attrs["round_id"] = round_id
    return GameEvent(
        event_id=event_id or f"{event_type.value}_test",
        event_type=event_type,
        start_sec=ts,
        end_sec=end_sec,
        confidence=0.9,
        evidence=[Evidence(timestamp_sec=ts, frame_index=10, source="test")],
        attributes=attrs,
    )


def _detection(ts: float, label: str = "player", conf: float = 0.85) -> Detection:
    return Detection(
        label=label,
        confidence=conf,
        bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
        frame_index=int(ts * 10),
        timestamp_sec=ts,
    )


def _track(track_id: str, label: str, detections: list[Detection]) -> Track:
    return Track(track_id=track_id, label=label, detections=detections)


def _analysis(
    video_id: str = "vid_001",
    rounds: list[RoundAnalysis] | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        video=VideoInput(video_id=video_id, path=Path("/fake/video.mp4")),
        metadata=VideoMetadata(duration_sec=120.0, fps=30.0, width=1920, height=1080),
        rounds=rounds or [],
    )


# ---------------------------------------------------------------------------
# Data-model construction tests
# ---------------------------------------------------------------------------

class EvidenceRefTests(TestCase):
    def test_minimal_construction(self) -> None:
        ref = EvidenceRef(timestamp_sec=5.0, source="test")
        self.assertEqual(ref.timestamp_sec, 5.0)
        self.assertIsNone(ref.frame_index)

    def test_full_construction(self) -> None:
        ref = EvidenceRef(frame_index=42, timestamp_sec=5.0, source="RoundBoundaryDetector")
        self.assertEqual(ref.frame_index, 42)
        self.assertEqual(ref.source, "RoundBoundaryDetector")


class TimelineEventTests(TestCase):
    def test_construction(self) -> None:
        te = TimelineEvent(
            event_id="round_start_round_001",
            event_type="round_start",
            start_sec=0.0,
            confidence=0.9,
            round_id="round_001",
            evidence=[EvidenceRef(timestamp_sec=0.0, source="test")],
        )
        self.assertEqual(te.event_id, "round_start_round_001")
        self.assertEqual(te.event_type, "round_start")
        self.assertEqual(te.confidence, 0.9)
        self.assertEqual(len(te.evidence), 1)

    def test_attributes_preserved(self) -> None:
        te = TimelineEvent(
            event_id="kill_001",
            event_type="player_kill",
            start_sec=5.0,
            confidence=0.55,
            attributes={"kill_index": 1, "kf_key": "kill_feed.active"},
        )
        self.assertEqual(te.attributes["kill_index"], 1)
        self.assertEqual(te.attributes["kf_key"], "kill_feed.active")


class TrackSummaryTests(TestCase):
    def test_construction(self) -> None:
        ts = TrackSummary(
            track_id="track_0001",
            label="enemy",
            first_seen_sec=3.0,
            last_seen_sec=45.0,
            detection_count=120,
            avg_confidence=0.87,
        )
        self.assertEqual(ts.track_id, "track_0001")
        self.assertEqual(ts.label, "enemy")
        self.assertEqual(ts.detection_count, 120)


class RoundTimelineTests(TestCase):
    def test_complete_round(self) -> None:
        rt = RoundTimeline(
            round_id="round_001",
            start_sec=0.0,
            end_sec=95.0,
            duration_sec=95.0,
        )
        self.assertEqual(rt.round_id, "round_001")
        self.assertEqual(rt.duration_sec, 95.0)
        self.assertEqual(rt.events, [])
        self.assertEqual(rt.tracks, [])

    def test_truncated_round(self) -> None:
        rt = RoundTimeline(round_id="round_002", start_sec=10.0)
        self.assertIsNone(rt.end_sec)
        self.assertIsNone(rt.duration_sec)


class MatchTimelineTests(TestCase):
    def test_minimal_construction(self) -> None:
        mt = MatchTimeline(video_id="vid_001")
        self.assertEqual(mt.schema_version, "2.0")
        self.assertEqual(mt.video_id, "vid_001")
        self.assertEqual(mt.total_rounds, 0)

    def test_model_dump(self) -> None:
        mt = MatchTimeline(
            video_id="vid_001",
            duration_sec=120.0,
            fps=30.0,
            resolution={"width": 1920, "height": 1080},
            total_rounds=1,
        )
        d = mt.model_dump(mode="json")
        self.assertEqual(d["schema_version"], "2.0")
        self.assertEqual(d["video_id"], "vid_001")
        self.assertEqual(d["resolution"]["width"], 1920)


# ---------------------------------------------------------------------------
# TimelineBuilder tests
# ---------------------------------------------------------------------------

class TimelineBuilderBasicTests(TestCase):
    def setUp(self) -> None:
        self.builder = TimelineBuilder()

    def test_empty_analysis(self) -> None:
        analysis = _analysis()
        mt = self.builder.build(analysis)
        self.assertEqual(mt.total_rounds, 0)
        self.assertEqual(mt.rounds, [])
        self.assertEqual(mt.video_id, "vid_001")

    def test_single_complete_round_no_tracks(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=95.0,
            events=[
                _event(EventType.ROUND_START, 0.0, round_id="round_001"),
                _event(EventType.PLAYER_KILL, 30.0),
                _event(EventType.ROUND_END, 95.0, round_id="round_001"),
            ],
        )
        analysis = _analysis(rounds=[ra])
        mt = self.builder.build(analysis)

        self.assertEqual(mt.total_rounds, 1)
        rt = mt.rounds[0]
        self.assertEqual(rt.round_id, "round_001")
        self.assertEqual(rt.start_sec, 0.0)
        self.assertEqual(rt.end_sec, 95.0)
        self.assertEqual(rt.duration_sec, 95.0)
        self.assertEqual(len(rt.events), 3)
        self.assertEqual(rt.tracks, [])

    def test_multiple_rounds(self) -> None:
        rounds = [
            RoundAnalysis(
                round_id="round_001",
                start_sec=0.0,
                end_sec=95.0,
                events=[
                    _event(EventType.ROUND_START, 0.0, round_id="round_001"),
                    _event(EventType.ROUND_END, 95.0, round_id="round_001"),
                ],
            ),
            RoundAnalysis(
                round_id="round_002",
                start_sec=110.0,
                end_sec=200.0,
                events=[
                    _event(EventType.ROUND_START, 110.0, round_id="round_002"),
                    _event(EventType.PLAYER_DEATH, 150.0),
                    _event(EventType.ROUND_END, 200.0, round_id="round_002"),
                ],
            ),
        ]
        analysis = _analysis(rounds=rounds)
        mt = self.builder.build(analysis)

        self.assertEqual(mt.total_rounds, 2)
        self.assertEqual(mt.rounds[0].round_id, "round_001")
        self.assertEqual(len(mt.rounds[0].events), 2)
        self.assertEqual(mt.rounds[1].round_id, "round_002")
        self.assertEqual(len(mt.rounds[1].events), 3)

    def test_truncated_round_no_end(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            events=[
                _event(EventType.ROUND_START, 0.0, round_id="round_001"),
                _event(EventType.PLAYER_KILL, 30.0),
            ],
        )
        analysis = _analysis(rounds=[ra])
        mt = self.builder.build(analysis)

        rt = mt.rounds[0]
        self.assertIsNone(rt.end_sec)
        self.assertIsNone(rt.duration_sec)
        self.assertEqual(len(rt.events), 2)


class TimelineBuilderEventConversionTests(TestCase):
    def setUp(self) -> None:
        self.builder = TimelineBuilder()

    def test_event_type_flattened_to_string(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0, round_id="round_001"),
                _event(EventType.PLAYER_KILL, 25.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 12.0),
            ],
        )
        mt = self.builder.build(_analysis(rounds=[ra]))

        event_types = [e.event_type for e in mt.rounds[0].events]
        self.assertEqual(event_types, ["round_start", "player_kill", "enemy_first_visible"])

    def test_event_evidence_converted_to_ref(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[
                _event(EventType.PLAYER_DEATH, 15.0, event_id="death_001"),
            ],
        )
        mt = self.builder.build(_analysis(rounds=[ra]))

        te = mt.rounds[0].events[0]
        self.assertEqual(len(te.evidence), 1)
        ref = te.evidence[0]
        self.assertIsInstance(ref, EvidenceRef)
        self.assertEqual(ref.source, "test")
        self.assertEqual(ref.timestamp_sec, 15.0)

    def test_event_attributes_transferred(self) -> None:
        ev = GameEvent(
            event_id="kill_001",
            event_type=EventType.PLAYER_KILL,
            start_sec=30.0,
            confidence=0.55,
            evidence=[],
            attributes={"kill_index": 1, "kf_key": "kill_feed.active"},
        )
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[ev],
        )
        mt = self.builder.build(_analysis(rounds=[ra]))
        te = mt.rounds[0].events[0]
        self.assertEqual(te.attributes["kill_index"], 1)

    def test_event_end_sec_preserved(self) -> None:
        ev = _event(EventType.COMBAT_START, 20.0, end_sec=30.0, event_id="combat_001")
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[ev],
        )
        mt = self.builder.build(_analysis(rounds=[ra]))
        te = mt.rounds[0].events[0]
        self.assertEqual(te.end_sec, 30.0)


class TimelineBuilderTrackTests(TestCase):
    def setUp(self) -> None:
        self.builder = TimelineBuilder()

    def test_build_with_tracks(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [
            _track("track_0000", "enemy", [
                _detection(5.0, "enemy", 0.9),
                _detection(10.0, "enemy", 0.85),
                _detection(15.0, "enemy", 0.88),
            ]),
        ]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 1)
        ts = mt.rounds[0].tracks[0]
        self.assertEqual(ts.track_id, "track_0000")
        self.assertEqual(ts.label, "enemy")
        self.assertEqual(ts.first_seen_sec, 5.0)
        self.assertEqual(ts.last_seen_sec, 15.0)
        self.assertEqual(ts.detection_count, 3)

    def test_duplicate_tracks_deduplicated(self) -> None:
        """Simulate tracker returning same track IDs across frames."""
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        # Same track appears many times (as in pipeline output).
        tracks = [
            _track("track_0000", "enemy", [_detection(1.0)]),
            _track("track_0000", "enemy", [_detection(1.0), _detection(2.0)]),
            _track("track_0000", "enemy", [_detection(1.0), _detection(2.0), _detection(3.0)]),
            _track("track_0001", "teammate", [_detection(1.5)]),
        ]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        # Should have 2 unique tracks, with the best version of track_0000 kept.
        self.assertEqual(len(mt.rounds[0].tracks), 2)
        track0 = [t for t in mt.rounds[0].tracks if t.track_id == "track_0000"][0]
        self.assertEqual(track0.detection_count, 3)  # kept the one with most detections

    def test_empty_track_detections(self) -> None:
        """Track with zero detections should get default summary values."""
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [_track("track_0000", "enemy", [])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        ts = mt.rounds[0].tracks[0]
        self.assertEqual(ts.detection_count, 0)
        self.assertEqual(ts.first_seen_sec, 0.0)
        self.assertEqual(ts.avg_confidence, 0.0)

    def test_track_avg_confidence_rounded(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [
            _track("t0", "enemy", [
                _detection(1.0, conf=0.9123),
                _detection(2.0, conf=0.8456),
                _detection(3.0, conf=0.9001),
            ]),
        ]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(mt.rounds[0].tracks[0].avg_confidence, 0.886)


class TimelineBuilderTrackOverlapTests(TestCase):
    """Track-to-round assignment via time-window overlap."""

    def setUp(self) -> None:
        self.builder = TimelineBuilder()

    def test_track_fully_inside_round(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(10.0), _detection(50.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 1)

    def test_track_partially_overlaps_round_start(self) -> None:
        """Track starts before round but ends inside — should be included."""
        ra = RoundAnalysis(
            round_id="round_002",
            start_sec=110.0,
            end_sec=200.0,
            events=[_event(EventType.ROUND_START, 110.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(90.0), _detection(150.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 1)

    def test_track_partially_overlaps_round_end(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(40.0), _detection(80.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 1)

    def test_track_before_round_excluded(self) -> None:
        ra = RoundAnalysis(
            round_id="round_002",
            start_sec=110.0,
            end_sec=200.0,
            events=[_event(EventType.ROUND_START, 110.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(5.0), _detection(50.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 0)

    def test_track_after_round_excluded(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(70.0), _detection(90.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 0)

    def test_track_truncated_round_no_end(self) -> None:
        """Track seen after round start when round has no end — included."""
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(30.0), _detection(50.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 1)

    def test_track_before_truncated_round_excluded(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=10.0,
            events=[_event(EventType.ROUND_START, 10.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(2.0), _detection(5.0)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(len(mt.rounds[0].tracks), 0)


class TimelineBuilderMetadataTests(TestCase):
    def setUp(self) -> None:
        self.builder = TimelineBuilder()

    def test_video_metadata_transferred(self) -> None:
        analysis = _analysis()
        mt = self.builder.build(analysis)
        self.assertEqual(mt.video_id, "vid_001")
        self.assertEqual(mt.duration_sec, 120.0)
        self.assertEqual(mt.fps, 30.0)
        self.assertEqual(mt.resolution, {"width": 1920, "height": 1080})

    def test_source_name_transferred(self) -> None:
        analysis = AnalysisResult(
            video=VideoInput(video_id="vid_002", path=Path("/f.mp4"), source_name="obs_recording"),
            metadata=VideoMetadata(),
        )
        mt = self.builder.build(analysis)
        self.assertEqual(mt.source_name, "obs_recording")

    def test_warnings_transferred(self) -> None:
        analysis = _analysis()
        analysis.warnings.append("Frame processing error: test")
        mt = self.builder.build(analysis)
        self.assertIn("Frame processing error: test", mt.warnings)


# ---------------------------------------------------------------------------
# Exporter tests
# ---------------------------------------------------------------------------

class TimelineExporterTests(TestCase):
    def setUp(self) -> None:
        self.builder = TimelineBuilder()
        self.exporter = TimelineExporter()

    def _make_timeline(self) -> MatchTimeline:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0, round_id="round_001"),
                _event(EventType.PLAYER_KILL, 25.0),
            ],
        )
        return self.builder.build(_analysis(rounds=[ra]))

    def test_to_json_returns_valid_json(self) -> None:
        mt = self._make_timeline()
        json_str = self.exporter.to_json(mt)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["video_id"], "vid_001")
        self.assertEqual(parsed["total_rounds"], 1)

    def test_to_json_event_in_round(self) -> None:
        mt = self._make_timeline()
        parsed = json.loads(self.exporter.to_json(mt))
        events = parsed["rounds"][0]["events"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "round_start")

    def test_to_json_evidence_inlined(self) -> None:
        mt = self._make_timeline()
        parsed = json.loads(self.exporter.to_json(mt))
        evidence = parsed["rounds"][0]["events"][0]["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["source"], "test")

    def test_export_writes_file(self) -> None:
        mt = self._make_timeline()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "timeline.json"
            result = self.exporter.export(mt, out)
            self.assertTrue(result.exists())
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn('"video_id": "vid_001"', content)

    def test_export_creates_parent_dirs(self) -> None:
        mt = self._make_timeline()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "deep" / "timeline.json"
            result = self.exporter.export(mt, out)
            self.assertTrue(result.exists())

    def test_convenience_timeline_to_json(self) -> None:
        mt = self._make_timeline()
        json_str = timeline_to_json(mt)
        self.assertIn("vid_001", json_str)

    def test_convenience_export_timeline(self) -> None:
        mt = self._make_timeline()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tl.json"
            result = export_timeline(mt, out)
            self.assertTrue(result.exists())

    def test_exporter_indent_configurable(self) -> None:
        mt = self._make_timeline()
        # Default indent=2
        s2 = TimelineExporter(indent=2).to_json(mt)
        s4 = TimelineExporter(indent=4).to_json(mt)
        self.assertNotEqual(s2, s4)

    def test_empty_timeline_json(self) -> None:
        mt = MatchTimeline(video_id="empty")
        parsed = json.loads(timeline_to_json(mt))
        self.assertEqual(parsed["total_rounds"], 0)
        self.assertEqual(parsed["rounds"], [])

    def test_full_timeline_with_tracks_json(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(10.0, conf=0.9), _detection(20.0, conf=0.8)])]
        mt = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        parsed = json.loads(timeline_to_json(mt))
        self.assertEqual(len(parsed["rounds"][0]["tracks"]), 1)
        self.assertEqual(parsed["rounds"][0]["tracks"][0]["label"], "enemy")
        self.assertEqual(parsed["rounds"][0]["tracks"][0]["detection_count"], 2)

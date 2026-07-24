"""Unit tests for evidence-grounded report generation."""

from __future__ import annotations

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
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.generator import EvidenceReportGenerator, ReportGenerator
from gamesight.reporting.models import (
    EvidenceLink,
    FindingCategory,
    FindingSeverity,
    MatchOverview,
    MatchReport,
    ReportFinding,
    RoundReport,
    RoundStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    event_type: EventType,
    ts: float,
    round_id: str | None = None,
    event_id: str | None = None,
    source: str = "test",
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
        evidence=[Evidence(timestamp_sec=ts, frame_index=int(ts * 10), source=source)],
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
# Data model tests
# ---------------------------------------------------------------------------

class EvidenceLinkTests(TestCase):
    def test_minimal(self) -> None:
        link = EvidenceLink(timestamp_sec=5.0, source="test")
        self.assertEqual(link.timestamp_sec, 5.0)
        self.assertIsNone(link.frame_index)

    def test_full(self) -> None:
        link = EvidenceLink(frame_index=42, timestamp_sec=5.0, source="RoundBoundaryDetector", asset_path="/tmp/a.png")
        self.assertEqual(link.frame_index, 42)
        self.assertEqual(link.asset_path, "/tmp/a.png")


class ReportFindingTests(TestCase):
    def test_construction(self) -> None:
        rf = ReportFinding(
            finding_id="combat_kill_001",
            category=FindingCategory.COMBAT,
            severity=FindingSeverity.WARNING,
            text="Kill detected.",
            confidence=0.55,
            evidence=[EvidenceLink(timestamp_sec=5.0, source="KillEventDetector")],
        )
        self.assertEqual(rf.category, FindingCategory.COMBAT)
        self.assertEqual(rf.severity, FindingSeverity.WARNING)
        self.assertEqual(len(rf.evidence), 1)

    def test_defaults(self) -> None:
        rf = ReportFinding(
            finding_id="test_001",
            category=FindingCategory.ROUND_FLOW,
            text="Default test.",
            confidence=0.5,
        )
        self.assertEqual(rf.severity, FindingSeverity.INFO)
        self.assertEqual(rf.evidence, [])

    def test_metadata_field(self) -> None:
        rf = ReportFinding(
            finding_id="meta_001",
            category=FindingCategory.MOVEMENT,
            text="With metadata.",
            confidence=0.8,
            metadata={"distance": 42.0, "zone": "A"},
        )
        self.assertEqual(rf.metadata["zone"], "A")


class RoundStatsTests(TestCase):
    def test_defaults(self) -> None:
        rs = RoundStats(round_id="round_001")
        self.assertEqual(rs.kills_detected, 0)
        self.assertEqual(rs.deaths_detected, 0)

    def test_full(self) -> None:
        rs = RoundStats(
            round_id="round_001",
            duration_sec=85.0,
            kills_detected=3,
            deaths_detected=1,
            enemy_tracks=2,
            enemy_first_visible_sec=12.0,
        )
        self.assertEqual(rs.kills_detected, 3)
        self.assertEqual(rs.enemy_first_visible_sec, 12.0)


class RoundReportTests(TestCase):
    def test_construction(self) -> None:
        rr = RoundReport(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            duration_sec=60.0,
            stats=RoundStats(round_id="round_001"),
        )
        self.assertEqual(rr.round_id, "round_001")
        self.assertEqual(rr.findings, [])


class MatchOverviewTests(TestCase):
    def test_construction(self) -> None:
        mo = MatchOverview(
            video_id="vid_001",
            total_rounds=3,
            total_kills_detected=5,
            total_deaths_detected=2,
        )
        self.assertEqual(mo.total_rounds, 3)
        self.assertEqual(mo.warnings, [])


class MatchReportTests(TestCase):
    def test_generated_at_is_set(self) -> None:
        mr = MatchReport(overview=MatchOverview(video_id="v1"))
        self.assertIsNotNone(mr.generated_at)

    def test_model_dump_for_json(self) -> None:
        mr = MatchReport(overview=MatchOverview(video_id="v1"))
        d = mr.model_dump_for_json()
        self.assertEqual(d["report_version"], "1.0")
        self.assertIn("generated_at", d)
        self.assertEqual(d["rounds"], [])


# ---------------------------------------------------------------------------
# EvidenceReportBuilder tests
# ---------------------------------------------------------------------------

class EvidenceReportBuilderBasicTests(TestCase):
    def setUp(self) -> None:
        self.builder = EvidenceReportBuilder()

    def test_empty_analysis(self) -> None:
        report = self.builder.build(_analysis())
        self.assertEqual(report.overview.total_rounds, 0)
        self.assertEqual(report.rounds, [])
        self.assertEqual(report.overview.video_id, "vid_001")

    def test_single_round_no_combat(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0, round_id="round_001"),
                _event(EventType.ROUND_END, 60.0, round_id="round_001"),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.overview.total_rounds, 1)
        self.assertEqual(len(report.rounds), 1)
        self.assertEqual(report.rounds[0].round_id, "round_001")
        self.assertEqual(report.rounds[0].duration_sec, 60.0)
        self.assertEqual(report.rounds[0].stats.kills_detected, 0)
        self.assertEqual(report.rounds[0].stats.deaths_detected, 0)

    def test_multiple_rounds(self) -> None:
        rounds = [
            RoundAnalysis(
                round_id="round_001",
                start_sec=0.0, end_sec=60.0,
                events=[
                    _event(EventType.ROUND_START, 0.0),
                    _event(EventType.PLAYER_KILL, 25.0),
                    _event(EventType.ROUND_END, 60.0),
                ],
            ),
            RoundAnalysis(
                round_id="round_002",
                start_sec=70.0, end_sec=140.0,
                events=[
                    _event(EventType.ROUND_START, 70.0),
                    _event(EventType.PLAYER_KILL, 90.0),
                    _event(EventType.PLAYER_DEATH, 100.0),
                    _event(EventType.ROUND_END, 140.0),
                ],
            ),
        ]
        report = self.builder.build(_analysis(rounds=rounds))
        self.assertEqual(report.overview.total_rounds, 2)
        self.assertEqual(report.overview.total_kills_detected, 2)
        self.assertEqual(report.overview.total_deaths_detected, 1)
        self.assertEqual(report.rounds[0].stats.kills_detected, 1)
        self.assertEqual(report.rounds[1].stats.kills_detected, 1)
        self.assertEqual(report.rounds[1].stats.deaths_detected, 1)

    def test_round_with_killfeed_events(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 25.0, source="KillEventDetector.kill_feed.kill_feed_active"),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.rounds[0].stats.killfeed_events, 1)


class EvidenceReportBuilderCombatTests(TestCase):
    def setUp(self) -> None:
        self.builder = EvidenceReportBuilder()

    def test_kill_events_counted(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 10.0),
                _event(EventType.PLAYER_KILL, 20.0),
                _event(EventType.PLAYER_KILL, 30.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.rounds[0].stats.kills_detected, 3)

    def test_death_events_counted(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_DEATH, 15.0),
                _event(EventType.PLAYER_DEATH, 45.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.rounds[0].stats.deaths_detected, 2)

    def test_combat_start_counted(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.COMBAT_START, 10.0),
                _event(EventType.COMBAT_START, 30.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.rounds[0].stats.combat_segments, 2)


class EvidenceReportBuilderEnemyVisibleTests(TestCase):
    def setUp(self) -> None:
        self.builder = EvidenceReportBuilder()

    def test_enemy_first_visible_recorded(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 12.5),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.rounds[0].stats.enemy_first_visible_sec, 12.5)

    def test_earliest_enemy_visible_kept(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 20.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 8.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertEqual(report.rounds[0].stats.enemy_first_visible_sec, 8.0)


class EvidenceReportBuilderTrackTests(TestCase):
    def setUp(self) -> None:
        self.builder = EvidenceReportBuilder()

    def test_enemy_tracks_counted(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        tracks = [
            _track("t0", "enemy", [_detection(10.0, "enemy"), _detection(20.0, "enemy")]),
            _track("t1", "enemy", [_detection(15.0, "enemy")]),
        ]
        report = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(report.rounds[0].stats.enemy_tracks, 2)

    def test_teammate_tracks_counted(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        tracks = [
            _track("t0", "teammate", [_detection(5.0, "teammate")]),
            _track("t1", "teammate", [_detection(30.0, "teammate")]),
            _track("t2", "teammate", [_detection(45.0, "teammate")]),
        ]
        report = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(report.rounds[0].stats.teammate_tracks, 3)

    def test_mixed_tracks(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        tracks = [
            _track("t0", "enemy", [_detection(10.0, "enemy")]),
            _track("t1", "teammate", [_detection(20.0, "teammate")]),
            _track("t2", "enemy", [_detection(30.0, "enemy")]),
            _track("t3", "player", [_detection(40.0, "player")]),  # unclassified
        ]
        report = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        self.assertEqual(report.rounds[0].stats.enemy_tracks, 2)
        self.assertEqual(report.rounds[0].stats.teammate_tracks, 1)


class EvidenceReportBuilderFindingsTests(TestCase):
    def setUp(self) -> None:
        self.builder = EvidenceReportBuilder()

    def test_round_flow_findings_present(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        findings = report.rounds[0].findings
        self.assertTrue(any("started" in f.text for f in findings))
        self.assertTrue(any("ended" in f.text for f in findings))

    def test_truncated_round_generates_warning(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0,
            events=[_event(EventType.ROUND_START, 0.0)],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        truncated = [f for f in report.rounds[0].findings
                     if f.finding_id.startswith("round_flow_truncated")]
        self.assertEqual(len(truncated), 1)
        self.assertEqual(truncated[0].severity, FindingSeverity.WARNING)

    def test_kill_finding_has_evidence(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 25.0, source="KillEventDetector"),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        kill_findings = [f for f in report.rounds[0].findings
                         if f.finding_id.startswith("combat_kills")]
        self.assertEqual(len(kill_findings), 1)
        self.assertTrue(len(kill_findings[0].evidence) > 0)
        self.assertIn("KillEventDetector", kill_findings[0].evidence[0].source)

    def test_death_finding_has_evidence(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_DEATH, 35.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        death_findings = [f for f in report.rounds[0].findings
                          if f.finding_id.startswith("combat_deaths")]
        self.assertEqual(len(death_findings), 1)

    def test_no_combat_round_generates_info(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        none_findings = [f for f in report.rounds[0].findings
                         if f.finding_id.startswith("combat_none")]
        self.assertEqual(len(none_findings), 1)

    def test_enemy_first_visible_finding(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 15.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        efv = [f for f in report.rounds[0].findings
               if f.finding_id.startswith("movement_enemy_first_visible")]
        self.assertEqual(len(efv), 1)
        self.assertIn("15.0s", efv[0].text)

    def test_enemy_tracks_finding(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(10.0, "enemy")])]
        report = self.builder.build(_analysis(rounds=[ra]), tracks=tracks)
        et = [f for f in report.rounds[0].findings
              if f.finding_id.startswith("movement_enemy_tracks")]
        self.assertEqual(len(et), 1)

    def test_match_level_findings(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 25.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertTrue(any("round(s)" in f.text for f in report.match_findings))
        self.assertTrue(any("Total kills" in f.text for f in report.match_findings))

    def test_no_combat_match_level(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        report = self.builder.build(_analysis(rounds=[ra]))
        self.assertTrue(any("No combat" in f.text for f in report.match_findings))

    def test_warnings_in_match_findings(self) -> None:
        analysis = _analysis()
        analysis.warnings.append("Test warning: corrupted frame")
        report = self.builder.build(analysis)
        warning_findings = [f for f in report.match_findings
                            if "corrupted frame" in f.text]
        self.assertEqual(len(warning_findings), 1)


class EvidenceReportBuilderMetadataTests(TestCase):
    def setUp(self) -> None:
        self.builder = EvidenceReportBuilder()

    def test_video_metadata_in_overview(self) -> None:
        report = self.builder.build(_analysis())
        self.assertEqual(report.overview.video_id, "vid_001")
        self.assertEqual(report.overview.duration_sec, 120.0)
        self.assertEqual(report.overview.fps, 30.0)
        self.assertEqual(report.overview.resolution, {"width": 1920, "height": 1080})

    def test_source_name_in_overview(self) -> None:
        analysis = AnalysisResult(
            video=VideoInput(video_id="v2", path=Path("/x.mp4"), source_name="obs_record"),
            metadata=VideoMetadata(),
        )
        report = self.builder.build(analysis)
        self.assertEqual(report.overview.source_name, "obs_record")


# ---------------------------------------------------------------------------
# EvidenceReportGenerator tests
# ---------------------------------------------------------------------------

class EvidenceReportGeneratorInterfaceTests(TestCase):
    def test_implements_report_generator(self) -> None:
        gen = EvidenceReportGenerator()
        self.assertIsInstance(gen, ReportGenerator)

    def test_generate_returns_dict(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        gen = EvidenceReportGenerator()
        result = gen.generate(_analysis(rounds=[ra]))
        self.assertIsInstance(result, dict)
        self.assertIn("overview", result)
        self.assertIn("rounds", result)

    def test_generate_report_returns_matchreport(self) -> None:
        gen = EvidenceReportGenerator()
        report = gen.generate_report(_analysis())
        self.assertIsInstance(report, MatchReport)

    def test_with_tracks(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 60.0)],
        )
        tracks = [_track("t0", "enemy", [_detection(10.0, "enemy")])]
        gen = EvidenceReportGenerator(tracks=tracks)
        d = gen.generate(_analysis(rounds=[ra]))
        self.assertEqual(d["rounds"][0]["stats"]["enemy_tracks"], 1)

    def test_empty_analysis(self) -> None:
        gen = EvidenceReportGenerator()
        result = gen.generate(_analysis())
        self.assertEqual(result["overview"]["total_rounds"], 0)

    def test_dict_contains_serialisable_types(self) -> None:
        """All values in the generated dict should be JSON-serialisable."""
        import json

        ra = RoundAnalysis(
            round_id="round_001",
            start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 25.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        gen = EvidenceReportGenerator()
        result = gen.generate(_analysis(rounds=[ra]))
        # Should not raise.
        json.dumps(result)

"""Evidence report builder -- converts pipeline outputs into a MatchReport.

Now supports i18n via an optional I18nLoader so findings text
matches the user's language.
"""

from __future__ import annotations

from gamesight.domain.models import (
    AnalysisResult, EventType, Evidence, GameEvent, RoundAnalysis, Track,
)
from gamesight.i18n.loader import I18nLoader
from gamesight.reporting.models import (
    EvidenceLink, FindingCategory, FindingSeverity, MatchOverview,
    MatchReport, ReportFinding, RoundReport, RoundStats,
)
from gamesight.serialization.timeline import MatchTimeline, TimelineBuilder


class EvidenceReportBuilder:
    """Build an evidence-grounded MatchReport with i18n support."""

    _PREFIX_ROUND_FLOW = "round_flow"
    _PREFIX_COMBAT = "combat"
    _PREFIX_MOVEMENT = "movement"
    _PREFIX_MATCH = "match"

    def __init__(self, timeline_builder: TimelineBuilder | None = None, loader: I18nLoader | None = None) -> None:
        self._tl_builder = timeline_builder or TimelineBuilder()
        self._t = loader or I18nLoader("en")
        self._finding_counter = 0

    def build(self, analysis: AnalysisResult, tracks: list[Track] | None = None) -> MatchReport:
        self._finding_counter = 0
        timeline = self._tl_builder.build(analysis, tracks)
        round_reports = [
            self._build_round_report(ra, tl_round)
            for ra, tl_round in zip(analysis.rounds, timeline.rounds)
        ]
        overview = self._build_overview(analysis, round_reports, timeline)
        return MatchReport(
            overview=overview, rounds=round_reports,
            match_findings=self._build_match_findings(analysis, overview),
        )

    def _build_round_report(self, ra: RoundAnalysis, tl_round) -> RoundReport:
        stats = self._compute_round_stats(ra, tl_round)
        findings = self._build_round_findings(ra, stats)
        return RoundReport(
            round_id=ra.round_id, start_sec=ra.start_sec, end_sec=ra.end_sec,
            duration_sec=stats.duration_sec, stats=stats, findings=findings,
        )

    def _compute_round_stats(self, ra: RoundAnalysis, tl_round) -> RoundStats:
        kills = 0; deaths = 0; killfeed = 0
        enemy_first_vis: float | None = None; combat_segments = 0
        for event in ra.events:
            et = event.event_type
            if et == EventType.PLAYER_KILL: kills += 1
            elif et == EventType.PLAYER_DEATH: deaths += 1
            elif et == EventType.ENEMY_FIRST_VISIBLE:
                if enemy_first_vis is None or event.start_sec < enemy_first_vis:
                    enemy_first_vis = event.start_sec
            elif et == EventType.COMBAT_START: combat_segments += 1
            for ev in event.evidence:
                if "kill_feed" in ev.source.lower(): killfeed += 1; break
        duration = round(ra.end_sec - ra.start_sec, 2) if ra.end_sec is not None else None
        enemy_tracks = sum(1 for t in tl_round.tracks if t.label == "enemy")
        teammate_tracks = sum(1 for t in tl_round.tracks if t.label == "teammate")
        return RoundStats(
            round_id=ra.round_id, duration_sec=duration, kills_detected=kills,
            deaths_detected=deaths, killfeed_events=killfeed, enemy_tracks=enemy_tracks,
            teammate_tracks=teammate_tracks, enemy_first_visible_sec=enemy_first_vis,
            combat_segments=combat_segments,
        )

    def _build_round_findings(self, ra: RoundAnalysis, stats: RoundStats) -> list[ReportFinding]:
        findings: list[ReportFinding] = []
        t = self._t
        start_ev = self._first_event(ra, EventType.ROUND_START)
        end_ev = self._first_event(ra, EventType.ROUND_END)

        if start_ev is not None:
            findings.append(self._make_finding(f"{self._PREFIX_ROUND_FLOW}_start",
                FindingCategory.ROUND_FLOW, FindingSeverity.INFO,
                t.t("report_finding.round_start", time=start_ev.start_sec), 0.95, start_ev.evidence))
        if end_ev is not None and stats.duration_sec is not None:
            findings.append(self._make_finding(f"{self._PREFIX_ROUND_FLOW}_end",
                FindingCategory.ROUND_FLOW, FindingSeverity.INFO,
                t.t("report_finding.round_end", time=end_ev.start_sec, dur=stats.duration_sec), 0.90, end_ev.evidence))
        elif end_ev is None:
            findings.append(self._make_finding(f"{self._PREFIX_ROUND_FLOW}_truncated",
                FindingCategory.ROUND_FLOW, FindingSeverity.WARNING,
                t.t("report_finding.round_truncated"), 0.80, []))

        if stats.kills_detected > 0:
            kill_events = [e for e in ra.events if e.event_type == EventType.PLAYER_KILL]
            evidence = [lk for ev in kill_events for lk in self._to_links(ev.evidence)]
            findings.append(self._make_finding(f"{self._PREFIX_COMBAT}_kills",
                FindingCategory.COMBAT, FindingSeverity.INFO,
                t.t("report_finding.kills_detected", n=stats.kills_detected), 0.55, evidence))
        if stats.deaths_detected > 0:
            death_events = [e for e in ra.events if e.event_type == EventType.PLAYER_DEATH]
            evidence = [lk for ev in death_events for lk in self._to_links(ev.evidence)]
            findings.append(self._make_finding(f"{self._PREFIX_COMBAT}_deaths",
                FindingCategory.COMBAT, FindingSeverity.WARNING,
                t.t("report_finding.deaths_detected", n=stats.deaths_detected), 0.85, evidence))
        if stats.kills_detected == 0 and stats.deaths_detected == 0:
            findings.append(self._make_finding(f"{self._PREFIX_COMBAT}_none",
                FindingCategory.COMBAT, FindingSeverity.INFO,
                t.t("report_finding.no_combat"), 0.70, []))
        if stats.enemy_first_visible_sec is not None:
            efv_events = [e for e in ra.events if e.event_type == EventType.ENEMY_FIRST_VISIBLE]
            evidence = [lk for ev in efv_events for lk in self._to_links(ev.evidence)]
            findings.append(self._make_finding(f"{self._PREFIX_MOVEMENT}_enemy_first_visible",
                FindingCategory.MOVEMENT, FindingSeverity.INFO,
                t.t("report_finding.enemy_first_visible", time=stats.enemy_first_visible_sec), 0.80, evidence))
        if stats.enemy_tracks > 0:
            findings.append(self._make_finding(f"{self._PREFIX_MOVEMENT}_enemy_tracks",
                FindingCategory.MOVEMENT, FindingSeverity.INFO,
                t.t("report_finding.enemy_tracks", n=stats.enemy_tracks), 0.75, []))
        return findings

    def _build_overview(self, analysis, round_reports, timeline) -> MatchOverview:
        total_kills = sum(r.stats.kills_detected for r in round_reports)
        total_deaths = sum(r.stats.deaths_detected for r in round_reports)
        total_enemy = sum(r.stats.enemy_tracks for r in round_reports)
        return MatchOverview(
            video_id=analysis.video.video_id, source_name=analysis.video.source_name,
            duration_sec=analysis.metadata.duration_sec, fps=analysis.metadata.fps,
            resolution=timeline.resolution, total_rounds=len(round_reports),
            total_kills_detected=total_kills, total_deaths_detected=total_deaths,
            total_enemy_tracks=total_enemy, warnings=list(analysis.warnings),
        )

    def _build_match_findings(self, analysis, overview) -> list[ReportFinding]:
        findings: list[ReportFinding] = []
        t = self._t
        findings.append(self._make_finding(f"{self._PREFIX_MATCH}_rounds",
            FindingCategory.ROUND_FLOW, FindingSeverity.INFO,
            t.t("report_finding.match_rounds", n=overview.total_rounds,
                vid=overview.video_id, dur=overview.duration_sec or "unknown"), 1.0, []))
        if overview.total_kills_detected > 0:
            findings.append(self._make_finding(f"{self._PREFIX_MATCH}_total_kills",
                FindingCategory.COMBAT, FindingSeverity.INFO,
                t.t("report_finding.match_total_kills", n=overview.total_kills_detected), 0.55, []))
        if overview.total_deaths_detected > 0:
            findings.append(self._make_finding(f"{self._PREFIX_MATCH}_total_deaths",
                FindingCategory.COMBAT, FindingSeverity.WARNING,
                t.t("report_finding.match_total_deaths", n=overview.total_deaths_detected), 0.85, []))
        if overview.warnings:
            for i, w in enumerate(overview.warnings):
                findings.append(self._make_finding(f"{self._PREFIX_MATCH}_warning_{i}",
                    FindingCategory.ROUND_FLOW, FindingSeverity.WARNING, w, 0.90, []))
        if overview.total_kills_detected == 0 and overview.total_deaths_detected == 0:
            findings.append(self._make_finding(f"{self._PREFIX_MATCH}_no_combat",
                FindingCategory.COMBAT, FindingSeverity.INFO,
                t.t("report_finding.match_no_combat"), 0.70, []))
        return findings

    def _make_finding(self, suffix, category, severity, text, confidence, evidence) -> ReportFinding:
        self._finding_counter += 1
        return ReportFinding(
            finding_id=f"{suffix}_{self._finding_counter:03d}",
            category=category, severity=severity, text=text,
            confidence=confidence, evidence=self._to_links(evidence),
        )

    @staticmethod
    def _to_links(evidence: list[Evidence]) -> list[EvidenceLink]:
        return [EvidenceLink(frame_index=e.frame_index, timestamp_sec=e.timestamp_sec, source=e.source, asset_path=e.asset_path) for e in evidence]

    @staticmethod
    def _first_event(ra: RoundAnalysis, event_type: EventType) -> GameEvent | None:
        for e in ra.events:
            if e.event_type == event_type:
                return e
        return None

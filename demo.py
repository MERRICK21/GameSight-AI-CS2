"""Quick demo: run the full GameSight pipeline end-to-end."""
from gamesight.web.demo import run_demo_pipeline
from gamesight.serialization.timeline import TimelineBuilder
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.generator import EvidenceReportGenerator

print("=== Pipeline ===")
analysis, tracks = run_demo_pipeline("demo_cs2_match.mp4", sample_fps=10)
print(f"Rounds: {len(analysis.rounds)}")

print()
print("=== Timeline ===")
tl = TimelineBuilder().build(analysis, tracks)
print(f"Total rounds: {tl.total_rounds}")

print()
print("=== Evidence Report ===")
report = EvidenceReportBuilder().build(analysis, tracks)
for f in report.match_findings:
    print(f"  [{f.severity.value}] {f.text}")
print()
for rr in report.rounds:
    s = rr.stats
    print(f"  Round {rr.round_id} | {rr.duration_sec:.0f}s | {s.kills_detected} kills, {s.deaths_detected} deaths")

print()
gen = EvidenceReportGenerator(tracks=tracks)
rpt = gen.generate(analysis)
total = sum(len(rr["findings"]) for rr in rpt["rounds"]) + len(rpt["match_findings"])
print(f"Total findings: {total}")
print("Done!")

"""Run GameSight on a real CS2 POV video file.

Usage
-----
.. code-block:: bash

    python run_analysis.py path/to/your_cs2_clip.mp4
    python run_analysis.py path/to/your_cs2_clip.mp4 --sample-fps 5 --export

Requirements
------------
- opencv-python (pip install opencv-python)
- Optional: ultralytics + torch (for YOLO detection/tracking)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gamesight.domain.models import VideoInput
from gamesight.events.aggregator import aggregate_events
from gamesight.events.detectors import KillEventDetector, RoundBoundaryDetector
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
from gamesight.reporting.generator import EvidenceReportGenerator
from gamesight.serialization.exporter import (
    TimelineExporter,
)
from gamesight.serialization.timeline import TimelineBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="GameSight AI 鈥?CS2 Video Analysis")
    parser.add_argument("video", type=Path, help="Path to CS2 POV recording (.mp4, .mov, .mkv)")
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Frames per second to analyze (default: 10)")
    parser.add_argument("--export", action="store_true", help="Export timeline and report JSON to outputs/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory (default: outputs/)")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    suffix = video_path.suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv"}:
        print(f"Warning: '{suffix}' is not a standard format. OpenCV may still handle it.")

    video = VideoInput(video_id=video_path.stem, path=video_path)

    # ---- Step 1: Ingestion ----
    print(f"Reading: {video_path.name}")
    reader = OpenCVVideoReader()
    metadata = reader.inspect(video)
    print(f"  Resolution: {metadata.width}x{metadata.height}")
    print(f"  FPS: {metadata.fps}")
    print(f"  Duration: {metadata.duration_sec:.1f}s" if metadata.duration_sec else "  Duration: unknown")
    print(f"  Codec: {metadata.codec}")

    # ---- Step 2: Frame processing ----
    print(f"Processing frames at {args.sample_fps} fps...")
    parser_hud = CS2HudParser(
        extractors=[
            CrosshairExtractor(),
            HPBarExtractor(),
            KillFeedExtractor(),
            MoneyExtractor(),
            RoundInfoExtractor(),
        ],
    )

    hud_states = []
    frame_count = 0
    for frame in reader.frames(video, args.sample_fps):
        state = parser_hud.parse(frame.image, frame.frame_index, frame.timestamp_sec)
        hud_states.append(state)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")

    print(f"  Total frames processed: {len(hud_states)}")

    # ---- Step 3: Event detection ----
    print("Detecting events...")
    rbd = RoundBoundaryDetector()
    ked = KillEventDetector()

    events = []
    for state in hud_states:
        events.extend(rbd.update(state))
        events.extend(ked.update(state))
    events.extend(rbd.finalize())
    events.extend(ked.finalize())

    # ---- Step 4: Aggregation ----
    rounds = aggregate_events(events)
    print(f"  Rounds detected: {len(rounds)}")

    # ---- Step 5: Timeline + Report ----
    from gamesight.domain.models import AnalysisResult

    analysis = AnalysisResult(
        video=video,
        metadata=metadata,
        rounds=rounds,
    )
    timeline = TimelineBuilder().build(analysis, tracks=None)
    report_gen = EvidenceReportGenerator()
    report = report_gen.generate(analysis)

    # ---- Step 6: Print summary ----
    print()
    print("=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Video:    {video_path.name}")
    print(f"Rounds:   {len(rounds)}")
    print(f"Frames:   {len(hud_states)}")
    print()

    for f in timeline.rounds:
        if f.duration_sec:
            print(f"  {f.round_id}: {f.duration_sec:.1f}s, {len(f.events)} events")
        else:
            print(f"  {f.round_id}: truncated, {len(f.events)} events")

    print()
    print("Match findings:")
    for f in report["match_findings"]:
        print(f"  [{f['severity']}] {f['text']}")

    # ---- Step 7: Export (optional) ----
    if args.export:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tl_path = args.output_dir / f"timeline_{video.video_id}.json"
        rpt_path = args.output_dir / f"report_{video.video_id}.json"

        TimelineExporter().export(timeline, tl_path)
        rpt_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print()
        print(f"Timeline exported: {tl_path}")
        print(f"Report exported:    {rpt_path}")


if __name__ == "__main__":
    main()

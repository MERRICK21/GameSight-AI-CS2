"""GameSight Web Server 鈥?single-file HTTP server with upload + analysis UI.

Usage
-----
.. code-block:: bash

    python server.py
    # Open http://localhost:8765 in your browser

No dependencies beyond Python stdlib + GameSight + OpenCV.
"""

from __future__ import annotations

import html
import json
import os
import tempfile
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# GameSight imports
from gamesight.domain.models import AnalysisResult, VideoInput
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
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.models import MatchReport
from gamesight.serialization.timeline import TimelineBuilder

PORT = 8765
UPLOAD_DIR = Path(tempfile.gettempdir()) / "gamesight_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HTML page (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GameSight AI 鈥?CS2 Analysis</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d2991d;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }

  header { text-align: center; margin-bottom: 2rem; }
  header h1 { font-size: 1.8rem; margin-bottom:.25rem; }
  header p { color: var(--muted); font-size:.9rem; }

  .upload-zone {
    border: 2px dashed var(--border); border-radius: 12px; padding: 3rem 2rem;
    text-align: center; cursor: pointer; transition: border-color .2s;
    background: var(--surface); margin-bottom: 1.5rem;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent); }
  .upload-zone input { display: none; }
  .upload-zone .icon { font-size: 3rem; margin-bottom:.75rem; }
  .upload-zone .hint { color: var(--muted); font-size:.85rem; margin-top:.5rem; }

  .config-row { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .config-row label { color: var(--muted); font-size:.85rem; }
  .config-row input, .config-row select {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); padding:.4rem .75rem; font-size:.9rem;
  }

  .btn {
    display: inline-flex; align-items: center; gap:.4rem;
    padding:.6rem 1.5rem; border: none; border-radius: 8px; font-size:.95rem;
    font-weight: 600; cursor: pointer; transition: opacity .2s;
  }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }

  .progress-bar {
    width: 100%; height: 6px; background: var(--surface); border-radius: 3px;
    margin: 1rem 0; overflow: hidden; display: none;
  }
  .progress-bar.active { display: block; }
  .progress-fill {
    height: 100%; background: var(--accent); border-radius: 3px;
    width: 0%; transition: width .3s;
  }
  .status-text { text-align: center; color: var(--muted); font-size:.85rem; margin-bottom: 1rem; }

  .results { display: none; }
  .results.active { display: block; }

  .tab-bar { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
  .tab-btn {
    padding:.5rem 1.2rem; background: none; border: none; color: var(--muted);
    cursor: pointer; font-size:.9rem; border-bottom: 2px solid transparent;
    transition: color .2s, border-color .2s;
  }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .kpi { background: var(--surface); border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid var(--border); }
  .kpi .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .kpi .label { font-size:.75rem; color: var(--muted); text-transform: uppercase; margin-top:.25rem; }

  .finding { border-left: 4px solid var(--border); padding:.75rem 1rem; margin:.5rem 0; border-radius: 0 6px 6px 0; background: var(--surface); }
  .finding.info { border-left-color: var(--accent); }
  .finding.warning { border-left-color: var(--orange); }
  .finding.critical { border-left-color: var(--red); }
  .finding .meta { font-size:.8rem; color: var(--muted); margin-top:.3rem; }

  table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size:.9rem; }
  th, td { padding:.5rem .75rem; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size:.8rem; text-transform: uppercase; }

  .download-btn { margin-top: 1rem; }

  .file-name { display: block; margin-top:.5rem; color: var(--green); font-weight: 600; }

  @media (max-width: 600px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .tab-bar { overflow-x: auto; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>&#x1F3AF; GameSight AI</h1>
    <p>CS2 POV Analysis Pipeline 鈥?upload a recording and get an evidence-grounded report</p>
  </header>

  <!-- Upload -->
  <div class="upload-zone" id="dropZone">
    <div class="icon">&#x1F4C1;</div>
    <strong id="dropText">Click to select or drop a CS2 recording</strong>
    <span class="file-name" id="fileName"></span>
    <p class="hint">.mp4 / .mov / .mkv &middot; 16:9 (1920x1080) recommended &middot; First-person POV</p>
    <input type="file" id="fileInput" accept=".mp4,.mov,.mkv">
  </div>

  <!-- Config -->
  <div class="config-row">
    <label>Analysis FPS:</label>
    <input type="number" id="sampleFps" value="10" min="1" max="30" step="1" style="width:70px">
    <button class="btn btn-primary" id="runBtn" disabled>Run Analysis</button>
    <button class="btn btn-secondary" id="resetBtn" style="display:none">New Analysis</button>
  </div>

  <!-- Progress -->
  <div class="progress-bar" id="progressBar"><div class="progress-fill" id="progressFill"></div></div>
  <div class="status-text" id="statusText"></div>

  <!-- Results -->
  <div class="results" id="results">
    <div class="tab-bar" id="tabBar">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="timeline">Timeline</button>
      <button class="tab-btn" data-tab="report">Report</button>
      <button class="tab-btn" data-tab="evidence">Evidence</button>
      <button class="tab-btn" data-tab="json">Raw JSON</button>
    </div>
    <div class="tab-content active" id="tab-overview"></div>
    <div class="tab-content" id="tab-timeline"></div>
    <div class="tab-content" id="tab-report"></div>
    <div class="tab-content" id="tab-evidence"></div>
    <div class="tab-content" id="tab-json"></div>
  </div>

</div>

<script>
// ---- State ----
let currentResult = null;

// ---- Upload ----
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const dropText = document.getElementById('dropText');
const runBtn = document.getElementById('runBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; updateFile(); }
});
fileInput.addEventListener('change', updateFile);

function updateFile() {
  const f = fileInput.files[0];
  if (f) {
    fileName.textContent = f.name + ' (' + formatSize(f.size) + ')';
    dropText.style.display = 'none';
    runBtn.disabled = false;
  } else {
    fileName.textContent = '';
    dropText.style.display = '';
    runBtn.disabled = true;
  }
}

function formatSize(bytes) {
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ---- Run ----
runBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  const file = fileInput.files[0];
  if (!file) return;

  const progressBar = document.getElementById('progressBar');
  const progressFill = document.getElementById('progressFill');
  const statusText = document.getElementById('statusText');
  const results = document.getElementById('results');

  runBtn.disabled = true;
  progressBar.classList.add('active');
  results.classList.remove('active');
  statusText.textContent = 'Uploading...';
  progressFill.style.width = '10%';

  const formData = new FormData();
  formData.append('video', file);
  formData.append('sample_fps', document.getElementById('sampleFps').value);

  try {
    statusText.textContent = 'Analyzing frames...';
    progressFill.style.width = '30%';

    const resp = await fetch('/analyze', { method: 'POST', body: formData });
    progressFill.style.width = '80%';

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Analysis failed');
    }

    const data = await resp.json();
    progressFill.style.width = '100%';
    statusText.textContent = 'Analysis complete!';
    currentResult = data;
    renderAll(data);
    results.classList.add('active');
    document.getElementById('resetBtn').style.display = 'inline-flex';
  } catch (err) {
    statusText.textContent = 'Error: ' + err.message;
    runBtn.disabled = false;
  }
}

// ---- Reset ----
document.getElementById('resetBtn').addEventListener('click', () => {
  currentResult = null;
  document.getElementById('results').classList.remove('active');
  document.getElementById('progressBar').classList.remove('active');
  document.getElementById('statusText').textContent = '';
  runBtn.disabled = false;
  document.getElementById('resetBtn').style.display = 'none';
  fileInput.value = '';
  fileName.textContent = '';
  dropText.style.display = '';
});

// ---- Tabs ----
document.getElementById('tabBar').addEventListener('click', e => {
  if (!e.target.classList.contains('tab-btn')) return;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById('tab-' + e.target.dataset.tab).classList.add('active');
});

// ---- Render ----
function renderAll(data) {
  renderOverview(data);
  renderTimeline(data);
  renderReport(data);
  renderEvidence(data);
  renderJson(data);
}

function renderOverview(data) {
  const ov = data.overview;
  let html = '<div class="kpi-grid">' +
    kpi('Video', ov.video_id) +
    kpi('Rounds', ov.total_rounds) +
    kpi('Duration', ov.duration_sec ? ov.duration_sec.toFixed(0)+'s' : 'N/A') +
    kpi('Kills', ov.total_kills_detected) +
    kpi('Deaths', ov.total_deaths_detected) +
    kpi('Enemy Tracks', ov.total_enemy_tracks) +
    '</div>';

  html += '<h3>Round Summary</h3><table><tr><th>Round</th><th>Duration</th><th>Kills</th><th>Deaths</th><th>Killfeed</th><th>Enemy Tracks</th></tr>';
  (data.rounds||[]).forEach(r => {
    const s = r.stats;
    html += `<tr><td>${r.round_id}</td><td>${r.duration_sec?.toFixed(1)||'-'}s</td><td>${s.kills_detected}</td><td>${s.deaths_detected}</td><td>${s.killfeed_events}</td><td>${s.enemy_tracks}</td></tr>`;
  });
  html += '</table>';
  document.getElementById('tab-overview').innerHTML = html;
}

function renderTimeline(data) {
  let html = '';
  (data.rounds||[]).forEach(r => {
    html += `<details ${r === data.rounds[0] ? 'open' : ''}><summary><strong>${r.round_id}</strong> &middot; ${r.duration_sec?.toFixed(1)||'truncated'}s</summary>`;
    if (!r.findings || !r.findings.length) {
      html += '<p style="color:var(--muted)">No findings.</p>';
    } else {
      r.findings.forEach(f => {
        html += `<div class="finding ${f.severity}">
          <strong>[${f.severity.toUpperCase()}]</strong> ${h(f.text)}
          <div class="meta">confidence: ${f.confidence.toFixed(2)} &middot; id: ${h(f.finding_id)}</div>`;
        if (f.evidence && f.evidence.length) {
          html += '<details style="margin-top:.3rem"><summary>Evidence links</summary>';
          f.evidence.forEach(lk => {
            html += `<div class="meta">frame=${lk.frame_index||'?'} t=${lk.timestamp_sec.toFixed(1)}s src=${h(lk.source)}</div>`;
          });
          html += '</details>';
        }
        html += '</div>';
      });
    }
    html += '</details>';
  });
  document.getElementById('tab-timeline').innerHTML = html;
}

function renderReport(data) {
  let html = '<h3>Match Summary</h3>';
  (data.match_findings||[]).forEach(f => {
    html += `<div class="finding ${f.severity}"><strong>[${f.severity.toUpperCase()}]</strong> ${h(f.text)}</div>`;
  });
  html += '<hr style="border-color:var(--border);margin:1.5rem 0">';

  (data.rounds||[]).forEach(r => {
    html += `<h3>Round ${r.round_id}</h3>`;
    if (r.duration_sec) html += `<p style="color:var(--muted)">Duration: ${r.duration_sec.toFixed(1)}s</p>`;
    const s = r.stats;
    html += `<div class="kpi-grid" style="grid-template-columns:repeat(4,1fr)">` +
      kpi('Kills', s.kills_detected) + kpi('Deaths', s.deaths_detected) +
      kpi('Enemy Tracks', s.enemy_tracks) +
      kpi('1st Enemy', s.enemy_first_visible_sec ? s.enemy_first_visible_sec.toFixed(1)+'s' : 'N/A') +
      '</div>';
    (r.findings||[]).forEach(f => {
      const icon = {info:'&#x2139;',warning:'&#x26A0;',critical:'&#x1F6A8;'}[f.severity]||'';
      html += `<p>${icon} ${h(f.text)}</p>`;
    });
    html += '<hr style="border-color:var(--border);margin:1rem 0">';
  });
  document.getElementById('tab-report').innerHTML = html;
}

function renderEvidence(data) {
  let links = [];
  (data.rounds||[]).forEach(r => {
    (r.findings||[]).forEach(f => {
      (f.evidence||[]).forEach(lk => {
        links.push({round:r.round_id, finding:f.finding_id, category:f.category, frame:lk.frame_index||'-', timestamp:lk.timestamp_sec.toFixed(1)+'s', source:lk.source});
      });
    });
  });
  if (!links.length) { document.getElementById('tab-evidence').innerHTML = '<p style="color:var(--muted)">No evidence links.</p>'; return; }

  let html = `<p style="color:var(--muted)">${links.length} evidence links across all rounds</p><table><tr><th>Round</th><th>Finding</th><th>Category</th><th>Frame</th><th>Time</th><th>Source</th></tr>`;
  links.forEach(l => {
    html += `<tr><td>${l.round}</td><td>${h(l.finding)}</td><td>${l.category}</td><td>${l.frame}</td><td>${l.timestamp}</td><td>${h(l.source)}</td></tr>`;
  });
  html += '</table>';
  document.getElementById('tab-evidence').innerHTML = html;
}

function renderJson(data) {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  document.getElementById('tab-json').innerHTML =
    `<button class="btn btn-primary download-btn" onclick="window.open('${url}')">Download Report JSON</button>
     <pre style="background:var(--surface);padding:1rem;border-radius:8px;overflow:auto;max-height:70vh;font-size:.8rem;margin-top:1rem">${h(json)}</pre>`;
}

function kpi(label, value) {
  return `<div class="kpi"><div class="value">${value}</div><div class="label">${label}</div></div>`;
}
function h(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class GameSightHandler(SimpleHTTPRequestHandler):
    """Custom handler: serves the UI page and the /analyze API endpoint."""

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/analyze":
            self._handle_analyze()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        content = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_analyze(self) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json({"error": "Expected multipart/form-data"}, 400)
                return

            # Parse multipart form data
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Extract boundary
            boundary = content_type.split("boundary=")[1].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]

            # Parse parts
            parts = self._parse_multipart(body, boundary.encode())

            video_data = parts.get("video")
            if not video_data:
                self._send_json({"error": "No video file uploaded"}, 400)
                return

            sample_fps = float(parts.get("sample_fps", b"10").decode())

            # Save uploaded file
            filename = "uploaded_clip.mp4"
            filepath = UPLOAD_DIR / filename
            filepath.write_bytes(video_data)

            # Run pipeline
            result = self._run_pipeline(filepath, sample_fps)

            # Cleanup
            filepath.unlink(missing_ok=True)

            self._send_json(result)

        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, 500)

    def _run_pipeline(self, filepath: Path, sample_fps: float) -> dict:
        """Run the full GameSight pipeline on the uploaded video."""
        video = VideoInput(video_id=filepath.stem, path=filepath)

        # Ingestion
        reader = OpenCVVideoReader()
        metadata = reader.inspect(video)

        # HUD parsing
        parser = CS2HudParser(CS2_STANDARD_16X9, {`n            "crosshair": CrosshairExtractor(),`n            "player_status": HPBarExtractor(),`n            "kill_feed": KillFeedExtractor(),`n            "money": MoneyExtractor(),`n            "round_info": RoundInfoExtractor(),`n        })

        hud_states = []
        for frame in reader.frames(video, sample_fps):
            state = parser.parse(frame.image, frame.frame_index, frame.timestamp_sec)
            hud_states.append(state)

        # Event detection
        rbd = RoundBoundaryDetector()
        ked = KillEventDetector()
        events = []
        for state in hud_states:
            events.extend(rbd.update(state))
            events.extend(ked.update(state))
        events.extend(rbd.finalize())
        events.extend(ked.finalize())

        # Aggregation
        rounds = aggregate_events(events)

        analysis = AnalysisResult(video=video, metadata=metadata, rounds=rounds)

        # Timeline + Report
        timeline = TimelineBuilder().build(analysis)
        report = EvidenceReportBuilder().build(analysis)

        return report.model_dump(mode="json")

    @staticmethod
    def _parse_multipart(body: bytes, boundary: bytes) -> dict[str, bytes]:
        """Minimal multipart/form-data parser."""
        parts: dict[str, bytes] = {}
        sep = b"--" + boundary

        # Split by boundary
        sections = body.split(sep)
        for section in sections:
            if not section or section == b"--\r\n" or section == b"--":
                continue

            # Split headers from body
            if b"\r\n\r\n" not in section:
                continue
            header_part, body_part = section.split(b"\r\n\r\n", 1)

            # Remove trailing \r\n before next boundary
            body_part = body_part.rstrip(b"\r\n")
            # Also handle trailing --
            if body_part.endswith(b"--"):
                body_part = body_part[:-2]

            # Extract field name
            headers = header_part.decode("utf-8", errors="replace")
            name = None
            for line in headers.split("\r\n"):
                if 'name="' in line:
                    start = line.index('name="') + 6
                    end = line.index('"', start)
                    name = line[start:end]
                    break

            if name:
                parts[name] = body_part

        return parts

    def _send_json(self, data: dict, status: int = 200) -> None:
        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args) -> None:
        """Suppress default access log noise."""
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    server = HTTPServer(("0.0.0.0", PORT), GameSightHandler)
    print(f"  GameSight AI 鈥?CS2 Analysis")
    print(f"  Open http://localhost:{PORT} in your browser")
    print(f"  Press Ctrl+C to stop")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

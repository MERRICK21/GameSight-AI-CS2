# Optional CS2 enemy detector

GameSight can optionally use `models/yolov10n_cs2.pt` to distinguish T and
CT bodies and create evidence-backed enemy-contact review windows. The weight
is intentionally not committed to this repository.

- Source: https://huggingface.co/Vombit/yolov10n_cs2
- Upstream labels: `c`, `ch`, `t`, `th`
- Upstream license: CC-BY-NC-ND-4.0 (non-commercial)
- Expected local path: `models/yolov10n_cs2.pt`

Without this file, round reconstruction, flash detection, scope detection and
representative keyframes continue to work. The app shows a warning and does
not invent enemy contacts.

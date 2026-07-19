# Architecture Notes

The project uses ports-and-adapters boundaries:

- `domain` owns typed data contracts.
- `ingestion` and `preprocessing` adapt raw recordings.
- `perception` contains replaceable detector and HUD-parser ports.
- `tracking` and `events` provide temporal reasoning boundaries.
- `reporting` creates evidence-grounded narrative output.
- `orchestration` composes the interfaces without coupling to a model provider.

Concrete YOLO, ByteTrack, OCR, VLM, LLM, and Streamlit implementations belong in later sprints and must depend on these interfaces, not the reverse.

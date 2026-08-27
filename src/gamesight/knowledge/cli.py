"""Command-line utilities for reproducible local knowledge indexing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gamesight.knowledge.embeddings import (
    DEFAULT_MULTILINGUAL_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from gamesight.knowledge.indexing import index_documents
from gamesight.knowledge.loaders import load_knowledge_document
from gamesight.knowledge.store import ChromaKnowledgeStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gamesight-knowledge",
        description="Build or query the local CS2 coaching knowledge index.",
    )
    parser.add_argument(
        "--index", default="data/knowledge_index",
        help="Local Chroma index directory.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MULTILINGUAL_MODEL,
        help="Sentence Transformers embedding model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Index knowledge sources.")
    build.add_argument("sources", nargs="+", type=Path)
    build.add_argument("--reset", action="store_true")
    query = subparsers.add_parser("query", help="Retrieve matching passages.")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    embedder = SentenceTransformerEmbeddingProvider(args.model)
    store = ChromaKnowledgeStore(
        args.index, embedder, reset=getattr(args, "reset", False),
    )
    if args.command == "build":
        documents = [load_knowledge_document(path) for path in args.sources]
        result = index_documents(documents, store)
        print(json.dumps({
            "documents": result.document_count,
            "chunks": result.chunk_count,
            "index": str(Path(args.index)),
            "model": args.model,
        }, ensure_ascii=False))
        return 0
    matches = store.query(args.text, top_k=args.top_k)
    print(json.dumps([
        {
            "chunk_id": match.chunk.chunk_id,
            "score": round(match.score, 4),
            "title": match.chunk.title,
            "heading": match.chunk.heading,
            "source_uri": match.chunk.source_uri,
            "content": match.chunk.content,
        }
        for match in matches
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Eidolon-style offline stream RAG CLI.

Standalone project, run from this repository root:
  uv run python scripts/stream_rag.py --help
Dependencies are managed by this repository's pyproject.toml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# src-layout: run directly without depending on any AI-Wife or global PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streamrag.corpus import (
    ChatClient, prepare, refine, check_refined_complete, build_catalog, quality_report,
)
from streamrag.retrieval import (
    EmbeddingClient, CohereReranker, build_index, search, evaluate,
)
from streamrag.manual import export_batches, import_responses, validate_import
from streamrag.embedding_setup import (
    DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL, configure_key,
    embedding_status, resolve_api_key,
)


def embedding_client(args):
    mode = args.embed_mode
    key, _ = resolve_api_key(args.embed_key_env)
    if mode == "hash-test" and not args.allow_test_embeddings:
        raise ValueError("hash-test requires --allow-test-embeddings; never use it in production")
    return EmbeddingClient(mode=mode, model=args.embed_model,
                           base_url=args.embed_url, api_key=key)


def llm_client(args):
    return ChatClient(base_url=args.llm_url, model=args.llm_model,
                      api_key=os.getenv(args.llm_key_env, ""))


def reranker_client(args):
    if not args.rerank_url:
        return None
    return CohereReranker(args.rerank_url, args.rerank_model,
                          os.getenv(args.rerank_key_env, ""))


def add_embed_flags(parser):
    parser.add_argument("--embed-mode", choices=("api", "hash-test"), default="api")
    parser.add_argument("--allow-test-embeddings", action="store_true")
    parser.add_argument("--embed-model", default=os.getenv(
        "STREAM_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--embed-url", default=os.getenv(
        "STREAM_RAG_EMBED_URL", DEFAULT_EMBED_URL))
    parser.add_argument("--embed-key-env", default="STREAM_RAG_EMBED_API_KEY")


def add_llm_flags(parser):
    parser.add_argument("--llm-url", default=os.getenv(
        "STREAM_RAG_LLM_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--llm-model", default=os.getenv("STREAM_RAG_LLM_MODEL", ""))
    parser.add_argument("--llm-key-env", default="STREAM_RAG_LLM_API_KEY")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum NEW windows to refine (cost control)")
    parser.add_argument("--retries", type=int, default=2)


def add_retrieval_flags(parser):
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=15)
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--rerank-url", default="",
                        help="Optional full Cohere-compatible HTTPS rerank URL")
    parser.add_argument("--rerank-model", default="rerank-v3.5")
    parser.add_argument("--rerank-key-env", default="STREAM_RAG_RERANK_API_KEY")


def add_source_flags(parser):
    parser.add_argument("--input", required=True, type=Path,
                        help="Transcript .txt / .srt / .jsonl, or directory")
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--source-id", help="Stable source ID, single input file only")
    parser.add_argument("--metadata", type=Path, help="JSON map: relative path -> source metadata")
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--max-window-chars", type=int, default=3600)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LLM refine -> local LanceDB hybrid retrieval (Eidolon-style, not Eidolon source code)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("embedding-configure", help="Save an Embedding API key privately (hidden prompt)")

    p = sub.add_parser("embedding-status", help="Inspect embedding config without exposing secrets")
    add_embed_flags(p)
    p.add_argument("--live", action="store_true",
                   help="Explicitly send one test embedding request to the provider")

    p = sub.add_parser("prepare", help="Normalize transcripts into five-minute windows")
    add_source_flags(p)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("refine", help="LLM enrich each window; supports safe resume")
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    add_llm_flags(p)

    p = sub.add_parser("export-batches", help="Make offline prompt files for Gemini/Astra")
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--max-windows", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=8000)

    p = sub.add_parser("import-refined", help="Import signed model JSON without an LLM API")
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, help="e.g. gemini4-web or astra-local")

    p = sub.add_parser("validate", help="Validate refined data, signatures and coverage")
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--strict", action="store_true", help="Fail if any window is missing")

    p = sub.add_parser("ingest", help="Embed and build a local LanceDB index")
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--windows", type=Path,
                   help="Verify complete, matching prepared windows before indexing")
    p.add_argument("--batch-size", type=int, default=16)
    add_embed_flags(p)

    p = sub.add_parser("catalog", help="Aggregate refined chunks into source catalog (no LLM)")
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("report", help="Show coverage and speaker uncertainty counts")
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("search", help="Offline index search (optional embedding/rerank API)")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--query", required=True)
    add_embed_flags(p)
    add_retrieval_flags(p)

    p = sub.add_parser("eval", help="Recall@k and MRR on labeled golden queries")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path)
    add_embed_flags(p)
    add_retrieval_flags(p)

    p = sub.add_parser("pipeline", help="Prepare -> refine -> ingest -> catalog -> report")
    add_source_flags(p)
    p.add_argument("--workdir", type=Path, required=True)
    add_llm_flags(p)
    add_embed_flags(p)
    p.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args(argv)
    if args.command in ("prepare", "pipeline") and not args.input.exists():
        parser.error(f"Input path does not exist: {args.input}")
    if args.command in ("search", "eval") and not 1 <= args.top_k <= 50:
        parser.error("top-k must be 1..50")
    if args.command in ("refine", "pipeline") and (
        args.retries < 0 or args.retries > 5 or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Invalid LLM retries or limit")
    if args.command == "embedding-configure":
        result = configure_key()
    elif args.command == "embedding-status":
        if args.embed_mode != "api":
            raise ValueError("embedding-status only supports real API mode")
        result = embedding_status(
            args.embed_model, args.embed_url, args.embed_key_env, live=args.live)
    elif args.command == "prepare":
        rows = prepare(args.input, args.output, args.speaker_id, args.source_id,
                       args.metadata, args.max_window_chars, args.duration_seconds)
        result = {"prepared_windows": len(rows), "output": str(args.output)}
    elif args.command == "refine":
        result = refine(args.windows, args.output, llm_client(args),
                        limit=args.limit, retries=args.retries)
    elif args.command == "export-batches":
        result = export_batches(args.windows, args.outdir,
                                max_windows=args.max_windows, max_chars=args.max_chars)
    elif args.command == "import-refined":
        result = import_responses(args.windows, args.manifest, args.responses,
                                  args.output, args.model)
    elif args.command == "validate":
        result = validate_import(args.windows, args.refined, strict=args.strict)
    elif args.command == "ingest":
        if args.windows:
            check_refined_complete(args.windows, args.refined)
        result = build_index(args.refined, args.db, embedding_client(args),
                             batch_size=args.batch_size)
    elif args.command == "catalog":
        result = build_catalog(args.refined, args.output)
    elif args.command == "report":
        result = quality_report(args.refined, args.output)
    elif args.command == "search":
        result = search(args.db, args.query,
                        None if args.lexical_only else embedding_client(args),
                        top_k=args.top_k, candidates=args.candidates,
                        reranker=reranker_client(args), lexical_only=args.lexical_only,
                        verbose_errors=True)
    elif args.command == "eval":
        result = evaluate(args.db, args.gold,
                          None if args.lexical_only else embedding_client(args),
                          top_k=args.top_k, candidates=args.candidates,
                          lexical_only=args.lexical_only,
                          reranker=reranker_client(args))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    else:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)
        windows = workdir / "windows.jsonl"
        refined = workdir / "refined_streams.jsonl"
        db = workdir / "lancedb"
        prepare(args.input, windows, args.speaker_id, args.source_id,
                args.metadata, args.max_window_chars, args.duration_seconds)
        llm_summary = refine(windows, refined, llm_client(args),
                             limit=args.limit, retries=args.retries)
        if llm_summary["errors"]:
            raise RuntimeError("Refinement failed for some windows; rerun for checkpoint recovery")
        check_refined_complete(windows, refined)
        index = build_index(refined, db, embedding_client(args), batch_size=args.batch_size)
        catalog = build_catalog(refined, workdir / "stream_catalog.json")
        report = quality_report(refined, workdir / "quality_report.json")
        result = {"llm": llm_summary, "index": index,
                  "catalog_sources": len(catalog["broadcasts"]), "quality": report,
                  "paths": {"refined": str(refined), "db": str(db),
                            "catalog": str(workdir / "stream_catalog.json")}}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Stream RAG failed: {exc}", file=sys.stderr)
        sys.exit(1)

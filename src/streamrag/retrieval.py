"""Local LanceDB stream archive with Chinese BM25 + vector RRF + optional rerank.

The DB holds third-party stream evidence, never the app's user memories.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
import tempfile
import urllib.request
import warnings
from uuid import uuid4
from pathlib import Path

from .corpus import read_jsonl

STOPWORDS = {"了", "的", "是", "呀", "啊", "呢", "吧", "吗", "哦", "和",
             "就", "有", "在", "而且", "这个", "那个", "然后", "就是", "一个"}


def _jieba():
    try:
        import jieba
        return jieba
    except ImportError as exc:
        raise RuntimeError("Install dependency: uv pip install jieba") from exc


def terms(text):
    tokenizer = _jieba()
    words = tokenizer.lcut(str(text).lower())
    return [word for word in words
            if word not in STOPWORDS and re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9_]+", word)
            and (len(word) > 1 or word.isascii())]


def _lancedb():
    try:
        import lancedb
        return lancedb
    except ImportError as exc:
        raise RuntimeError("Install dependency: uv pip install lancedb") from exc


class EmbeddingClient:
    """OpenAI-compatible embeddings; hash-test is explicit and is NOT a real embedder."""
    def __init__(self, mode, model, base_url="", api_key="", timeout=40):
        if mode not in ("api", "hash-test"):
            raise ValueError("Embedding mode must be api or hash-test")
        if mode == "api":
            if not model or not api_key:
                raise ValueError("Embedding API key/model required")
            if not base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
                raise ValueError("Embedding endpoint must be HTTPS or localhost")
        self.mode, self.model = mode, model if mode == "api" else "HASH-TEST-NOT-PRODUCTION"
        self.base_url = base_url.rstrip("/")
        self.api_key, self.timeout = api_key, timeout

    def embed(self, texts):
        is_batch = isinstance(texts, list)
        sources = texts if is_batch else [texts]
        if not sources or any(not isinstance(s, str) or not s.strip() for s in sources):
            raise ValueError("Empty embedding input")
        if self.mode == "hash-test":
            output = []
            for s in sources:
                vec = [0.0] * 128
                for token in terms(s):
                    code = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                    vec[code % 128] += 1.0 if code & 1 else -1.0
                norm = math.sqrt(sum(x*x for x in vec)) or 1.0
                output.append([x/norm for x in vec])
            return output if is_batch else output[0]
        path = self.base_url if self.base_url.endswith("/embeddings") else self.base_url + "/embeddings"
        req = urllib.request.Request(
            path,
            data=json.dumps({"model": self.model, "input": sources,
                             "encoding_format": "float"}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.load(response)
        returned = sorted(data["data"], key=lambda x: int(x.get("index", 0)))
        if len(returned) != len(sources):
            raise ValueError("Embedding response count mismatch")
        vectors = [row["embedding"] for row in returned]
        if any(not isinstance(v, list) or not v or
               any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in v)
               for v in vectors):
            raise ValueError("Embedding response contains invalid numbers")
        if len({len(v) for v in vectors}) != 1:
            raise ValueError("Embedding dimension mismatch")
        return vectors if is_batch else vectors[0]


def embedding_document(item):
    """Embed the distilled content plus bounded source evidence.

    The source excerpt helps when LLM summaries omit a useful term, without
    allowing multi-minute transcripts to swamp the distillation or exceed API limits.
    """
    parts = [
        item.get("details", "")[:2400],
        "\n".join(item.get("dialogue_flow", []))[:950],
        item.get("interaction_notes", "")[:650],
        "\n".join(item.get("verbatim_excerpts", []))[:850],
        item.get("summary", "")[:750],
        item.get("raw_text", "")[:1350],
        item.get("topic", "")[:120],
        " ".join(item.get("topics", []))[:120],
    ]
    return "\n".join(str(part) for part in parts if part).strip()[:6500]


def _publish_index(staging, destination):
    """Keep an old index recoverable until a fully built replacement exists."""
    staging, destination = Path(staging), Path(destination)
    backup = destination.parent / ("." + destination.name + ".previous-" + uuid4().hex)
    had_existing = destination.exists()
    try:
        if had_existing:
            if not destination.is_dir():
                raise ValueError(f"Index destination is not a directory: {destination}")
            destination.rename(backup)
        try:
            staging.rename(destination)
        except Exception:
            if had_existing:
                backup.rename(destination)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and destination.exists():
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                warnings.warn(
                    f"Index published successfully, but old backup could not be "
                    f"removed: {backup} ({type(exc).__name__}). Clean it up later.",
                    RuntimeWarning, stacklevel=2)


def build_index(refined_path, index_dir, embeddings, batch_size=16):
    rows = [row for row in read_jsonl(refined_path) if row.get("keep", False)]
    if not rows:
        raise ValueError("No retained stream chunks; refine input first")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs: clean or rebuild the refined JSONL")
    speakers = {row["speaker_id"] for row in rows}
    if len(speakers) != 1:
        raise ValueError("Keep one broadcaster per index; never mix speaker identities")
    if batch_size < 1 or batch_size > 64:
        raise ValueError("Embedding batch must have 1-64 items")
    vectors = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        vectors.extend(embeddings.embed([embedding_document(r) for r in batch]))
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Inconsistent vector dimensions")
    lancedb = _lancedb()
    if len(vectors) != len(rows):
        raise ValueError("Embedding result count mismatch")
    if not vectors[0] or any(not isinstance(vec, (tuple, list))
                              or len(vec) != len(vectors[0])
                              or any(not isinstance(x, (int, float))
                                     or not math.isfinite(float(x)) for x in vec)
                              for vec in vectors):
        raise ValueError("Invalid embedding vectors")

    dest = Path(index_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        dir=dest.parent, prefix="." + dest.name + ".building-"))
    try:
        records = []
        for item, vector in zip(rows, vectors):
            full_content = "\n".join([
                item.get("topic", ""), item.get("summary", ""),
                item.get("details", ""),
                "\n".join(item.get("dialogue_flow", [])),
                item.get("interaction_notes", ""),
                "\n".join(item.get("verbatim_excerpts", [])),
                " ".join(item.get("topics", [])), item["raw_text"]
            ])
            records.append({
                "id": item["id"], "source_id": item["source_id"],
                "speaker_id": item["speaker_id"], "source_file": item["source_file"],
                "source_sha256": item["source_sha256"],
                "start_ms": -1 if item.get("start_ms") is None else int(item["start_ms"]),
                "end_ms": -1 if item.get("end_ms") is None else int(item["end_ms"]),
                "topic": item.get("topic", ""), "summary": item.get("summary", ""),
                "details": item.get("details", ""), "raw_text": item["raw_text"],
                "trigger_context": item.get("trigger_context") or "",
                "interaction_notes": item.get("interaction_notes", ""),
                "dialogue_flow_json": json.dumps(item.get("dialogue_flow", []), ensure_ascii=False),
                "verbatim_excerpts_json": json.dumps(item.get("verbatim_excerpts", []), ensure_ascii=False),
                "scene_summaries_json": json.dumps(item.get("scene_summaries", []), ensure_ascii=False),
                "topics_json": json.dumps(item.get("topics", []), ensure_ascii=False),
                "source_meta_json": json.dumps(
                    item.get("source_meta", {}), ensure_ascii=False),
                "search_terms": " ".join(terms(full_content)),
                "vector": list(map(float, vector)),
            })
        db = lancedb.connect(str(staging))
        table = db.create_table("stream_chunks", data=records, mode="overwrite")
        from lancedb.index import FTS
        table.create_index("search_terms", config=FTS(
            base_tokenizer="whitespace", stem=False,
            remove_stop_words=False), replace=True)
        manifest = {
            "schema_version": 1, "table": "stream_chunks",
            "row_count": len(records), "speaker_id": next(iter(speakers)),
            "embedding_mode": embeddings.mode, "embedding_model": embeddings.model,
            "vector_dimensions": len(vectors[0]),
            "source_ids": sorted({r["source_id"] for r in rows}),
            "embedding_text_version": 3,
            "narrative_evidence_fields": True,
            "refined_sha256": hashlib.sha256(Path(refined_path).read_bytes()).hexdigest(),
            "note": ("Test embedding mode is not suitable for real RAG."
                     if embeddings.mode == "hash-test" else ""),
        }
        (staging / "stream_rag_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        del table, db
        gc.collect()
        _publish_index(staging, dest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest

def rrf_merge(vector_results, lexical_results, top_count=15, k=60):
    """Fuse independent search rankings. Do not mix incomparable raw scores."""
    candidates = {}
    for group in (vector_results, lexical_results):
        for rank, row in enumerate(group, 1):
            key = row["id"]
            current = candidates.setdefault(key, {"item": row, "score": 0.0})
            current["score"] += 1.0 / (k + rank)
    return sorted(candidates.values(), key=lambda v: (-v["score"], v["item"]["id"]))[:top_count]


class CohereReranker:
    """Optional Cohere-compatible /rerank endpoint, never a required dependency."""
    def __init__(self, url, model, key, timeout=30):
        if not key or not url:
            raise ValueError("Rerank URL/key must be set together")
        if not url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Rerank endpoint must use HTTPS or localhost")
        self.url, self.model, self.key, self.timeout = url, model, key, timeout

    def rerank(self, query, merged, top_k):
        if not merged:
            return []
        documents = [embedding_document({
            "summary": x["item"].get("summary", ""),
            "details": x["item"].get("details", ""),
            "dialogue_flow": json.loads(x["item"].get("dialogue_flow_json", "[]")),
            "interaction_notes": x["item"].get("interaction_notes", ""),
            "verbatim_excerpts": json.loads(x["item"].get("verbatim_excerpts_json", "[]")),
            "raw_text": x["item"].get("raw_text", ""),
            "topic": x["item"].get("topic", ""),
        }) for x in merged]
        req = urllib.request.Request(
            self.url, method="POST",
            data=json.dumps({"model": self.model, "query": query,
                             "documents": documents, "top_n": top_k}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.load(response)
        result = []
        for row in data["results"]:
            i = int(row["index"])
            if i < 0 or i >= len(merged):
                raise ValueError("Out-of-range rerank index")
            result.append({"item": merged[i]["item"],
                           "score": float(row["relevance_score"])})
        return result


def render_reference(row, max_raw_chars=2400):
    """Produce substantive external conversational evidence, not tag-only memories."""
    flow = row.get("dialogue_flow", [])
    quotes = row.get("verbatim_excerpts", [])
    sections = [
        "【外部直播对话参考：不是角色本人经历或长期记忆】",
        "话题：" + row.get("topic", ""),
        "具体内容：" + row.get("details", ""),
    ]
    if row.get("trigger_context"):
        sections.append("明确触发背景：" + row["trigger_context"])
    if flow:
        sections.append("话语如何展开：\n" + "\n".join("- " + s for s in flow))
    if row.get("interaction_notes"):
        sections.append("真实互动方式：" + row["interaction_notes"])
    if quotes:
        sections.append("来自原文的逐字摘录：\n" + "\n".join("- " + q for q in quotes))
    original = row.get("raw_text", "")
    if original:
        suffix = "…[剩余原文请读取 raw_text 字段]" if len(original) > max_raw_chars else ""
        sections.append("原始直播文字：\n" + original[:max_raw_chars] + suffix)
    return "\n\n".join(sections)


def open_index(index_dir):
    dest = Path(index_dir)
    manifest = json.loads((dest / "stream_rag_manifest.json").read_text(encoding="utf-8"))
    table = _lancedb().connect(str(dest)).open_table(manifest["table"])
    return table, manifest


def search(index_dir, query, embeddings=None, *, top_k=3, candidates=15,
           reranker=None, lexical_only=False, verbose_errors=False,
           strict_reranker=False):
    table, manifest = open_index(index_dir)
    if not query.strip():
        return []
    candidates = max(top_k, candidates)
    vector_rows = []
    if not lexical_only:
        if embeddings is None:
            raise ValueError("Embeddings are required for vector retrieval")
        if embeddings.model != manifest["embedding_model"]:
            raise ValueError("Embedding model mismatch; rebuild or use original model")
        if embeddings.mode != manifest["embedding_mode"]:
            raise ValueError("Embedding mode mismatch")
        vector_rows = table.search(embeddings.embed(query), query_type="vector").limit(candidates).to_list()
    lexical_rows = []
    query_tokens = terms(query)
    if query_tokens:
        lexical_rows = table.search(
            " ".join(query_tokens), query_type="fts", fts_columns="search_terms"
        ).limit(candidates).to_list()
    merged = rrf_merge(vector_rows, lexical_rows, top_count=candidates)
    reranker_status = "disabled"
    if reranker:
        try:
            reranked = reranker.rerank(query, merged, top_k)
            if not reranked and merged:
                raise ValueError("Reranker returned no results")
            merged = reranked
            reranker_status = "ok"
        except Exception as exc:
            if strict_reranker:
                raise RuntimeError("Reranker failed during evaluation; metrics unavailable") from exc
            reranker_status = "fallback"
            warnings.warn(
                "Reranker failed; using RRF results instead "
                f"({type(exc).__name__}).", RuntimeWarning, stacklevel=2)
            if verbose_errors:
                print("Rerank failed; using RRF:", type(exc).__name__)
    output = []
    for item in merged[:top_k]:
        row = item["item"]
        dialogue_flow = json.loads(row.get("dialogue_flow_json", "[]"))
        excerpts = json.loads(row.get("verbatim_excerpts_json", "[]"))
        result = {
            "id": row["id"], "source_id": row["source_id"],
            "speaker_id": row["speaker_id"], "topic": row["topic"],
            "summary": row["summary"], "details": row["details"],
            "raw_text": row["raw_text"], "source_file": row["source_file"],
            "trigger_context": row.get("trigger_context") or None,
            "interaction_notes": row.get("interaction_notes", ""),
            "dialogue_flow": dialogue_flow,
            "verbatim_excerpts": excerpts,
            "scene_summaries": json.loads(row.get("scene_summaries_json", "[]")),
            "topics": json.loads(row.get("topics_json", "[]")),
            "source_sha256": row["source_sha256"],
            "source_meta": json.loads(row["source_meta_json"]),
            "start_ms": None if row["start_ms"] < 0 else row["start_ms"],
            "end_ms": None if row["end_ms"] < 0 else row["end_ms"],
            "score": item["score"],
            "reranker_status": reranker_status,
        }
        result["reference_text"] = render_reference(result)
        output.append(result)
    return output


def evaluate(index_dir, gold_file, embeddings, *, top_k=3, candidates=15,
             lexical_only=False, reranker=None):
    examples = read_jsonl(gold_file)
    if not examples:
        raise ValueError("Evaluation requires at least one example")
    recalled, hit, reciprocal, reports = 0.0, 0, 0.0, []
    for row in examples:
        expected = set(row.get("relevant_ids", []))
        if not expected:
            raise ValueError("Each golden query needs relevant_ids")
        results = search(index_dir, row["query"], embeddings, top_k=top_k,
                         candidates=candidates, lexical_only=lexical_only,
                         reranker=reranker, strict_reranker=True)
        ids = [x["id"] for x in results]
        rank = next((i for i, name in enumerate(ids, 1) if name in expected), None)
        matched = len(expected.intersection(ids))
        recalled += matched / len(expected)
        hit += int(rank is not None)
        reciprocal += 1.0/rank if rank else 0.0
        reports.append({"query": row["query"], "relevant_ids": sorted(expected),
                        "retrieved_ids": ids, "relevant_retrieved": matched,
                        "first_relevant_rank": rank})
    return {"queries": len(examples), "recall_at_k": recalled/len(examples),
            "hit_rate_at_k": hit/len(examples),
            "mrr_at_k": reciprocal/len(examples), "k": top_k,
            "reranker_enabled": reranker is not None, "results": reports}

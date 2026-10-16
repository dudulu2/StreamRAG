"""Source-pinned three-layer LanceDB retrieval using local embeddings."""
from __future__ import annotations
import gc
import json
import math
import tempfile
from pathlib import Path
from .attribution import digest, read_jsonl, validate
from .retrieval import _lancedb, _publish_index, terms, rrf_merge

SCOPES = {"context":("context","viewer"), "facts":("fact",),
          "style":("quote","interaction"), "attribution":("fact","viewer")}
EPISTEMIC = {"self_report","opinion","hypothetical","hearsay","expression","viewer_report","transcript"}


def load_reviewed(revision, windows_path):
    revision=Path(revision)
    manifest=json.loads((revision/"manifest.json").read_text(encoding="utf-8"))
    for field,path in [("records_sha256",revision/"records.jsonl"),
                       ("author_spans_sha256",revision/"author_spans.jsonl"),
                       ("source_windows_sha256",windows_path)]:
        if digest(path)!=manifest[field]:
            raise ValueError("Reviewed data digest mismatch: "+field)
    units=read_jsonl(revision/"records.jsonl")
    windows={w["id"]:w for w in read_jsonl(windows_path)}
    validate(units,windows,read_jsonl(revision/"author_spans.jsonl"))
    return units,manifest


def document(unit):
    parts=[unit["text"]]
    if unit["kind"]=="interaction":
        parts += [unit.get("scenario","")]
    parts += [s["text"] for s in unit.get("trigger_evidence",[])]
    parts += [s["text"] for s in unit["evidence"]]
    return "\n".join(p for p in parts if p.strip())


def projection(unit):
    # Mixed source context is fetched explicitly by ID, never appended to a fact/style answer.
    fields=("id","window_id","source_id","source_file","source_sha256","window_signature",
            "speaker_id","speaker_verified","verification_method","review_status","start_ms","end_ms",
            "layer","kind","author","text","evidence","epistemic_status","trigger_evidence",
            "trigger_kind","evidence_kind","cross_window_context_id","scenario","model_analysis")
    return {k:unit[k] for k in fields if k in unit}


def build(revision, windows_path, index_dir, embeddings, *, speaker_aliases=None):
    units,review=load_reviewed(revision,windows_path)
    contract=embeddings.contract
    if not contract or contract.get("metric")!="cosine":
        raise ValueError("An explicit cosine embedding contract is required")
    dest=Path(index_dir).resolve()
    if dest.exists():
        raise ValueError("Index exists; select a new revision path")
    dest.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=".three-layer-",dir=dest.parent))
    rows=[]
    for start in range(0,len(units),16):
        batch=units[start:start+16]
        texts=[document(u) for u in batch]
        vectors=embeddings.embed(texts)
        if len(vectors)!=len(batch): raise ValueError("Embedding count mismatch")
        for unit,text,vector in zip(batch,texts,vectors):
            if len(vector)!=contract["dimensions"] or not all(math.isfinite(v) for v in vector):
                raise ValueError("Invalid embedding vector")
            rows.append({"id":unit["id"],"kind":unit["kind"],"layer":unit["layer"],
                         "review_status":unit["review_status"],"epistemic_status":unit["epistemic_status"],
                         "speaker_id":unit["speaker_id"],"payload":json.dumps(projection(unit),ensure_ascii=False),
                         "search_terms":" ".join(terms(text)),"vector":vector})
        print(f"Embedded {len(rows)}/{len(units)}",flush=True)
    db=_lancedb().connect(str(staging))
    table=db.create_table("attributed_records",data=rows)
    from lancedb.index import FTS
    table.create_index("search_terms",config=FTS(base_tokenizer="whitespace",stem=False,remove_stop_words=False))
    manifest={"schema_version":2,"table":"attributed_records","row_count":len(rows),
              "embedding_contract":contract,"embedding_model":embeddings.model,
              "embedding_mode":embeddings.mode,"review_manifest":review,
              "retrieval_document_version":1,"scopes":SCOPES,
              "speaker_ids":sorted({u["speaker_id"] for u in units}),
              "speaker_aliases":speaker_aliases or {}}
    (staging/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    del table,db
    gc.collect()
    _publish_index(staging,dest)
    return manifest


def topical_query(query, manifest):
    """In a single-speaker archive, its name is a filter, not the retrieval topic."""
    speakers=manifest.get("speaker_ids",[])
    if len(speakers)!=1:return query
    aliases=manifest.get("speaker_aliases",{}).get(speakers[0],[])
    result=query
    for name in sorted(aliases,key=len,reverse=True):
        result=result.replace(name,"")
    return result.strip() if terms(result) else query


def search(index_dir, query, embeddings, *, scope="facts", top_k=5, candidates=30,
           epistemic=None, min_similarity=None):
    if scope not in SCOPES or not 1<=top_k<=100 or candidates<top_k:
        raise ValueError("Invalid scope or retrieval limits")
    if epistemic is not None and epistemic not in EPISTEMIC:
        raise ValueError("Invalid epistemic filter")
    if min_similarity is not None and not -1<=min_similarity<=1:
        raise ValueError("Invalid similarity threshold")
    if not query.strip(): return []
    dest=Path(index_dir)
    manifest=json.loads((dest/"manifest.json").read_text(encoding="utf-8"))
    if embeddings.contract!=manifest["embedding_contract"] or embeddings.model!=manifest["embedding_model"] or embeddings.mode!=manifest["embedding_mode"]:
        raise ValueError("Embedding contract mismatch")
    table=_lancedb().connect(str(dest)).open_table(manifest["table"])
    kinds=", ".join("'"+s+"'" for s in SCOPES[scope])
    predicate=f"kind IN ({kinds})"
    if scope in ("facts","style"):
        predicate+=" AND review_status = 'text_reviewed'"
    if scope=="style": predicate+=" AND epistemic_status != 'hearsay'"
    if epistemic: predicate+=f" AND epistemic_status = '{epistemic}'"
    retrieval_query=topical_query(query,manifest)
    vector=embeddings.embed_query(retrieval_query)
    if len(vector)!=manifest["embedding_contract"]["dimensions"]:
        raise ValueError("Query vector dimensions mismatch")
    vec=table.search(vector,query_type="vector").distance_type("cosine").where(predicate,prefilter=True).limit(candidates).to_list()
    tokens=terms(retrieval_query)
    lexical=table.search(" ".join(tokens),query_type="fts",fts_columns="search_terms").where(predicate,prefilter=True).limit(candidates).to_list() if tokens else []
    similarities={r["id"]:1-r["_distance"] for r in vec}
    # With a threshold, lexical-only candidates cannot silently bypass the vector gate.
    merged=rrf_merge(vec,lexical,top_count=2*candidates)
    out=[]
    for entry in merged:
        row=entry["item"];similarity=similarities.get(row["id"])
        if min_similarity is not None and (similarity is None or similarity<min_similarity): continue
        item=json.loads(row["payload"])
        item.update(rrf_score=entry["score"],cosine_similarity=similarity)
        out.append(item)
        if len(out)==top_k:break
    return out

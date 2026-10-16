"""Read-only, persistent JSONL worker for companion applications.

Only stdin/stdout pipes: no network listener, no LLM API, no memory writes.
Start once per application so Qwen is not loaded for every conversational turn.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamrag.local_embedding import LocalQwenEmbedding
from streamrag.three_layer import load_reviewed, search
from streamrag.attribution import read_jsonl


def validate_payload(request):
    if not isinstance(request, dict) or not isinstance(request.get("id"), str):
        raise ValueError("Invalid request id")
    if not 1 <= len(request["id"]) <= 120:
        raise ValueError("Invalid request id length")
    action = request.get("action", "search")
    if action not in ("search", "health", "shutdown", "context"):
        raise ValueError("Invalid action")
    if action == "search":
        query = request.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 1600:
            raise ValueError("Invalid query")
        if request.get("scope", "style") not in ("style", "facts", "attribution"):
            raise ValueError("Runtime scope not permitted")
        if type(request.get("top_k", 3)) is not int or not 1 <= request.get("top_k", 3) <= 5:
            raise ValueError("Invalid top_k")
    return request


def safe_candidate(row, window_map, *, max_chars=1600):
    """Return separated literal evidence. Classification strings aren't spoken text."""
    if row.get("review_status") != "text_reviewed":
        return None
    if row.get("epistemic_status") in ("hearsay", "hypothetical"):
        return None
    if not row.get("evidence"):
        return None
    result = {k: row[k] for k in (
        "id", "window_id", "source_id", "speaker_id", "author", "kind", "layer",
        "epistemic_status", "review_status", "speaker_verified", "verification_method",
        "text", "model_analysis", "scenario", "trigger_kind", "rrf_score",
        "cosine_similarity") if k in row}
    # Cap by selecting complete spans; never present spliced text as one literal quotation.
    def select(spans, allowed_author, budget):
        output, used = [], 0
        for span in spans:
            if span.get("author") != allowed_author:
                continue
            window = window_map.get(span.get("window_id"))
            lo, hi = span.get("start_char"), span.get("end_char")
            if (window is None or type(lo) is not int or type(hi) is not int
                    or not 0 <= lo < hi <= len(window["raw_text"])
                    or window["raw_text"][lo:hi] != span.get("text")):
                raise ValueError("Candidate source evidence mismatch")
            text = span["text"]
            if used + len(text) > budget:
                continue
            output.append({k: span[k] for k in ("window_id", "start_char", "end_char", "text", "author")})
            used += len(text)
        return output
    result["evidence"] = select(row["evidence"], row["author"], max_chars)
    result["trigger_evidence"] = select(row.get("trigger_evidence", []), "viewer", 600)
    if not result["evidence"]:
        return None
    result["source_date_is_not_current_schedule"] = True
    result["can_become_character_memory"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "qwen")
    parser.add_argument("--speaker-id", default="nailv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    protocol_out = sys.stdout
    def emit(value):
        protocol_out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        protocol_out.flush()
    with contextlib.redirect_stdout(sys.stderr):
        units, review = load_reviewed(args.revision, args.windows)
        manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("review_manifest") != review:
            raise ValueError("Index and reviewed revision do not match")
        if manifest.get("speaker_ids") != [args.speaker_id]:
            raise ValueError("Runtime requires a single matching source speaker")
        windows = {w["id"]: w for w in read_jsonl(args.windows)}
        embed = LocalQwenEmbedding(args.model_path, device=args.device)
        if embed.contract != manifest["embedding_contract"]:
            raise ValueError("Index embedding contract mismatch")
        embed.embed_query("conversation")
        # Warm the lexical tokenizer/table as well; first live turn must not pay this cost.
        search(args.index, "conversation", embed, scope="style", top_k=1, candidates=5)
    emit({"event": "ready", "protocol": 1, "records": len(units),
          "source_speaker": args.speaker_id, "model": embed.model,
          "revision": review["records_sha256"], "readonly": True})
    for line in sys.stdin:
        request = None
        try:
            if len(line) > 32768:
                raise ValueError("Request too large")
            request = validate_payload(json.loads(line))
            action = request.get("action", "search")
            if action == "shutdown":
                emit({"id": request["id"], "ok": True, "status": "shutdown"})
                return
            if action == "health":
                emit({"id": request["id"], "ok": True, "status": "ready"})
                continue
            if action == "context":
                # Explicit context only, never automatically included in a fact/style result.
                window = windows.get(request.get("window_id"))
                if window is None:
                    raise ValueError("Unknown context id")
                emit({"id": request["id"], "ok": True, "status": "context",
                      "window_id": window["id"], "text": window["raw_text"],
                      "author": "mixed_context", "can_become_character_memory": False})
                continue
            with contextlib.redirect_stdout(sys.stderr):
                rows = search(args.index, request["query"], embed,
                              scope=request.get("scope", "style"),
                              top_k=min(100, max(10, request.get("top_k", 3) * 4)),
                              candidates=30)
            selected = []
            for row in rows:
                candidate = safe_candidate(row, windows)
                if candidate is not None:
                    selected.append(candidate)
                if len(selected) >= request.get("top_k", 3):
                    break
            emit({"id": request["id"], "ok": True,
                  "status": "candidates" if selected else "no_candidates",
                  "scope": request.get("scope", "style"), "items": selected,
                  "answer_supported": None,
                  "warning": "Candidates are not proof of answerability; verify relevance and attribution.",
                  "revision": review["records_sha256"], "source_speaker": args.speaker_id})
        except Exception as exc:
            # Do not echo prompts, user text, credentials, or raw exception payloads.
            emit({"id": request.get("id") if isinstance(request, dict) else None,
                  "ok": False, "error": type(exc).__name__, "status": "rejected"})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as error:
        print(json.dumps({"event": "startup_error", "error": type(error).__name__}), flush=True)
        sys.exit(1)

"""Prepare and refine real streamer transcripts into Eidolon-style five-minute memories."""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

CLOCK = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})")
SENTENCE = re.compile(r"[^\r\n。！？!?；;]+[。！？!?；;]*")
SUFFIXES = {".txt", ".srt", ".jsonl"}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                result.append(row)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return result


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ms(clock):
    h, m, s = clock.replace(",", ".").split(":")
    sec, frac = s.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000 + int(frac.ljust(3, "0")[:3])


def parse_source(path, speaker_id):
    path = Path(path)
    original = path.read_text(encoding="utf-8-sig")
    segments = []
    if path.suffix.lower() == ".srt":
        for block_no, block in enumerate(re.split(r"\n\s*\n", original.replace("\r\n", "\n")), 1):
            lines = block.strip().splitlines()
            clock_index = next((i for i, value in enumerate(lines) if CLOCK.search(value)), None)
            if clock_index is None:
                continue
            match = CLOCK.search(lines[clock_index])
            value = "\n".join(lines[clock_index + 1:]).strip()
            if value:
                segments.append({"segment_id": len(segments) + 1, "text": value,
                                 "start_ms": ms(match[1]), "end_ms": ms(match[2]),
                                 "original_cue": lines[0] if clock_index else str(block_no)})
    elif path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(original.splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"{path}:{line_no}: expected a text field")
            raw_speaker = item.get("speaker_id", item.get("speaker"))
            if raw_speaker and str(raw_speaker) != speaker_id:
                continue  # no attribution of someone else's words
            if not item["text"].strip():
                continue
            a, b = item.get("start_ms"), item.get("end_ms")
            if a is None and isinstance(item.get("start"), (float, int)):
                a = round(item["start"] * 1000)
            if b is None and isinstance(item.get("end"), (float, int)):
                b = round(item["end"] * 1000)
            segments.append({"segment_id": len(segments) + 1, "text": item["text"],
                             "start_ms": a, "end_ms": b, "original_line": line_no,
                             "speaker_explicit": bool(raw_speaker)})
    elif path.suffix.lower() == ".txt":
        for match in SENTENCE.finditer(original):
            raw = match.group()
            for offset in range(0, len(raw), 400):
                piece = raw[offset:offset + 400]
                if piece.strip():
                    segments.append({"segment_id": len(segments) + 1, "text": piece,
                                     "char_start": match.start() + offset,
                                     "char_end": match.start() + offset + len(piece),
                                     "start_ms": None, "end_ms": None})
    else:
        raise ValueError(f"Unsupported transcript: {path}")
    if not segments:
        raise ValueError(f"No usable text in {path}")
    return segments, digest(original)


def make_windows(segments, source_id, speaker_id, source_file, source_sha256,
                 metadata=None, duration_ms=300000, max_chars=3600):
    if max_chars < 400 or duration_ms <= 0:
        raise ValueError("Invalid window settings")
    timed = [row.get("start_ms") is not None for row in segments]
    if any(timed) and not all(timed):
        raise ValueError("Mixed timed and untimed evidence")
    is_timed = all(timed)
    groups, group, size, current_bucket = [], [], 0, None
    for row in segments:
        bucket = int(row["start_ms"]) // duration_ms if is_timed else None
        length = len(row["text"]) + 1
        if group and ((is_timed and bucket != current_bucket) or size + length > max_chars):
            groups.append(group)
            group, size = [], 0
        group.append(row)
        size += length
        current_bucket = bucket
    if group:
        groups.append(group)
    meta = {key: str(val) for key, val in (metadata or {}).items()
            if key in ("date", "bvid", "title", "url") and val is not None}
    output = []
    for num, rows in enumerate(groups, 1):
        output.append({
            "id": f"{source_id}__{num:05d}", "source_id": source_id,
            "speaker_id": speaker_id,
            "speaker_verified": all(r.get("speaker_explicit", False) for r in rows),
            "source_file": source_file, "source_sha256": source_sha256,
            "source_meta": meta,
            "window_strategy": "five_minutes" if is_timed else "character_fallback",
            "segment_start": rows[0]["segment_id"], "segment_end": rows[-1]["segment_id"],
            "start_char": rows[0].get("char_start"), "end_char": rows[-1].get("char_end"),
            "start_ms": rows[0].get("start_ms"), "end_ms": rows[-1].get("end_ms"),
            "raw_text": "\n".join(row["text"] for row in rows),
        })
    for i, row in enumerate(output):
        row["context_before"] = output[i-1]["raw_text"][-280:] if i else ""
        row["context_after"] = output[i+1]["raw_text"][:280] if i+1 < len(output) else ""
    return output


def prepare(input_path, output_file, speaker_id, source_id=None,
            metadata_file=None, max_chars=3600, duration_seconds=300):
    input_path = Path(input_path)
    if not speaker_id.strip():
        raise ValueError("A nonempty speaker_id is mandatory")
    files = (sorted(p for p in input_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUFFIXES)
             if input_path.is_dir() else [input_path])
    if not files:
        raise ValueError(f"No supported source files in {input_path}")
    if source_id and len(files) != 1:
        raise ValueError("source_id is only for a single file")
    metadata = (json.loads(Path(metadata_file).read_text(encoding="utf-8"))
                if metadata_file else {})
    output = []
    for file in files:
        segments, source_hash = parse_source(file, speaker_id)
        name = file.name if input_path.is_file() else file.relative_to(input_path).as_posix()
        meta = metadata.get(name, {})
        if not isinstance(meta, dict):
            raise ValueError(f"Bad metadata for {name}")
        file_id = source_id or meta.get("source_id") or (
            re.sub("[^A-Za-z0-9_-]", "_", file.stem)[:45] + "_" + digest(name)[:8])
        if not re.fullmatch("[A-Za-z0-9_-]+", file_id):
            raise ValueError(f"Bad source_id: {file_id}")
        output.extend(make_windows(
            segments, file_id, speaker_id, name, source_hash, meta,
            duration_ms=duration_seconds*1000, max_chars=max_chars))
    if len({row["id"] for row in output}) != len(output):
        raise ValueError("Duplicate window IDs")
    write_jsonl(output_file, output)
    return output


REFINE_PROMPT = """你是直播档案整理器，不是角色扮演者。
将主片段中的真实直播原文精炼为可检索的结构化记忆，不编造原文之外的事实。
前后文仅供消歧，不要将前后文内容写成主片段发生的事。
不能把游戏台词、引用、歌词、读弹幕当作主播本人经历。说话人不明时标记 unclear_speaker。
观众问题缺失则不能推断提问内容；不确定内容明确保留不确定性。
纯礼物播报、纯歌词、无意义重复、纯日程通知可设 keep=false。
返回严格 JSON 对象，键名为：
keep(boolean), topic(string), summary(string), details(string),
topics(string[]), scene_summaries(string[]), activity(string), risk_flags(string[])。
details 保留故事和讨论的展开、观点、转折、回应方式，但不要改写出不存在的剧情。
summary 简短，details 具体。不要在输出中重新生成原始逐字稿。"""


def validate_refinement(value):
    if not isinstance(value, dict) or type(value.get("keep")) is not bool:
        raise ValueError("LLM output needs a boolean keep")
    result = {"keep": value["keep"]}
    for key, max_length in (("topic", 180), ("summary", 1000),
                            ("details", 4000), ("activity", 100)):
        text = value.get(key, "")
        if not isinstance(text, str):
            raise ValueError(f"Bad LLM {key}")
        result[key] = text.strip()[:max_length]
    for key, max_length in (("topics", 90), ("scene_summaries", 300),
                            ("risk_flags", 90)):
        values = value.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"Bad LLM list {key}")
        result[key] = [v.strip()[:max_length] for v in values if v.strip()][:30]
    if result["keep"] and not (result["summary"] and result["details"]):
        raise ValueError("Retained memories require summary and details")
    return result


class ChatClient:
    def __init__(self, base_url, model, api_key, timeout=90):
        if not model or not api_key:
            raise ValueError("Set LLM model and API key")
        if not base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Use an HTTPS provider or localhost HTTP")
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.model, self.key, self.timeout = model, api_key, timeout

    def refine(self, window):
        user = json.dumps({
            "context_before": window.get("context_before", ""),
            "MAIN_SOURCE_TEXT": window["raw_text"],
            "context_after": window.get("context_after", "")
        }, ensure_ascii=False)
        request = urllib.request.Request(
            self.url, method="POST",
            data=json.dumps({
                "model": self.model, "temperature": 0.15, "stream": False,
                "messages": [{"role": "system", "content": REFINE_PROMPT},
                             {"role": "user", "content": user}]
            }).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            answer = json.load(response)["choices"][0]["message"]["content"].strip()
        fence = chr(96)*3
        if answer.startswith(fence):
            answer = re.sub("^" + fence + "(?:json)?\\s*", "", answer)
            answer = re.sub("\\s*" + fence + "$", "", answer)
        return validate_refinement(json.loads(answer))


def refine(windows_file, output_file, client, limit=None, retries=2):
    windows = read_jsonl(windows_file)
    path = Path(output_file)
    prior_rows = read_jsonl(path) if path.exists() else []
    refiner_id = str(getattr(client, "model", type(client).__name__))
    refiner_signature = digest(REFINE_PROMPT + refiner_id)
    prior = {row["id"]: row for row in prior_rows}
    if len(prior) != len(prior_rows):
        raise ValueError("Duplicate checkpoint IDs")
    for window in windows:
        existing = prior.get(window["id"])
        if existing and (existing.get("source_sha256") != window["source_sha256"]
                         or existing.get("speaker_id") != window["speaker_id"]):
            raise ValueError(f"Source changed since previous refinement: {window['id']}")
        if existing and existing.get("refiner_signature") != refiner_signature:
            raise ValueError(f"LLM model or prompt changed for {window['id']}; use a new workdir")
    done = set(prior)
    path.parent.mkdir(parents=True, exist_ok=True)
    complete, errors, skipped = 0, [], 0
    with path.open("a", encoding="utf-8", newline="\n") as dest:
        for window in windows:
            if window["id"] in done:
                skipped += 1
                continue
            if limit is not None and complete + len(errors) >= limit:
                break
            for attempt in range(retries + 1):
                try:
                    result = validate_refinement(client.refine(window))
                    break
                except (ValueError, KeyError, urllib.error.URLError) as exc:
                    if attempt == retries:
                        errors.append({"id": window["id"], "error": str(exc)[:180]})
                    else:
                        time.sleep(0.4 * (2 ** attempt))
            else:
                continue
            row = {k: v for k, v in window.items()
                   if k not in ("context_before", "context_after")}
            row.update(result)
            row["review_status"] = "auto_candidate"
            row["refiner_model"] = refiner_id
            row["refiner_signature"] = refiner_signature
            row["persona_memory_allowed"] = False
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            dest.flush()
            complete += 1
    if errors:
        write_jsonl(path.with_suffix(".errors.jsonl"), errors)
    return {"total_windows": len(windows), "refined": complete,
            "resume_skipped": skipped, "errors": len(errors),
            "error_ids": [item["id"] for item in errors]}


def build_catalog(refined_file, output_file):
    groups = defaultdict(list)
    for row in read_jsonl(refined_file):
        groups[row["source_id"]].append(row)
    entries = []
    for source_id, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x["id"])
        count = Counter(topic for row in rows if row.get("keep", True)
                        for topic in row.get("topics", []))
        first = rows[0]
        entries.append({
            "source_id": source_id, "speaker_id": first["speaker_id"],
            "source_file": first["source_file"], "source_sha256": first["source_sha256"],
            "source_meta": first.get("source_meta", {}),
            "total_windows": len(rows),
            "kept_windows": sum(bool(row.get("keep", True)) for row in rows),
            "top_topics": [{"topic": t, "count": n} for t, n in count.most_common(20)],
            "scenes": [{"id": row["id"], "summary": row.get("summary", ""),
                        "start_ms": row.get("start_ms"), "end_ms": row.get("end_ms")}
                       for row in rows if row.get("keep", True)]})
    catalog = {"schema_version": 1, "broadcasts": entries}
    dest = Path(output_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return catalog


def quality_report(refined_file, output_file):
    rows = read_jsonl(refined_file)
    report = {
        "rows": len(rows),
        "kept": sum(bool(x.get("keep")) for x in rows),
        "source_count": len({x["source_id"] for x in rows}),
        "source_characters": sum(len(x.get("raw_text", "")) for x in rows),
        "speaker_unverified": sum(not x.get("speaker_verified", False) for x in rows),
        "risky": sum(bool(x.get("risk_flags")) for x in rows),
        "review_statuses": dict(Counter(x.get("review_status", "unknown") for x in rows)),
        "note": "Automated counts; not human transcript or fact verification."
    }
    dest = Path(output_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

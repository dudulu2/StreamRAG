"""Exact-span attribution repair from explicit, source-pinned semantic decisions."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short_id(wid):
    kind = "e" if "_emotion_" in wid else "m" if "_memories_" in wid else "d"
    return kind + str(int(wid.rsplit("__", 1)[1]))


def resolve(text, selector):
    lines = text.splitlines(keepends=True)
    first, last = selector
    if not 1 <= first <= len(lines):
        raise ValueError(f"Invalid line: {selector}")
    start = sum(map(len, lines[:first-1]))
    if isinstance(last, str):
        line = lines[first-1]
        if line.count(last) != 1:
            raise ValueError(f"Nonunique/missing exact selection: {selector}: {line!r}")
        start += line.index(last)
        return start, start + len(last)
    if not first <= last <= len(lines):
        raise ValueError(f"Invalid line range: {selector}")
    return start, sum(map(len, lines[:last])) - len(lines[last-1]) + len(lines[last-1].rstrip("\r\n"))


def span(window, start, end, author):
    text = window["raw_text"]
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end-1].isspace():
        end -= 1
    if start == end:
        return None
    return {"window_id": window["id"], "start_char": start, "end_char": end,
            "line_start": text.count("\n", 0, start)+1,
            "line_end": text.count("\n", 0, end-1)+1,
            "text": text[start:end], "author": author,
            "spoken_by": "unverified"}


def selected(window, labels, source_span, author):
    if not source_span:
        return []
    start, end = source_span["start_char"], source_span["end_char"]
    result, pos = [], start
    while pos < end:
        stop = pos + 1
        while stop < end and labels[stop] == labels[pos]:
            stop += 1
        if labels[pos] == author:
            item = span(window, pos, stop, author)
            if item:
                result.append(item)
        pos = stop
    return result


def context(window, source_span):
    return [span(window, source_span["start_char"], source_span["end_char"], "context")] if source_span else []


def prepare(windows_path, attribution_path, patch_path):
    patch = json.loads(Path(patch_path).read_text(encoding="utf-8"))
    if digest(windows_path) != patch["source_windows_sha256"] or digest(attribution_path) != patch["source_attribution_sha256"]:
        raise ValueError("Source changed; semantic patch must be reviewed again")
    windows = {w["id"]: w for w in read_jsonl(windows_path)}
    original = read_jsonl(attribution_path)
    if len(windows) != len(original) or set(windows) != {a["window_id"] for a in original}:
        raise ValueError("Window coverage mismatch")
    if set(patch["viewer_segments"]) != {short_id(w) for w in windows}:
        raise ValueError("Semantic review coverage mismatch")
    annotations, labels_by_id = [], {}
    for a in original:
        w = windows[a["window_id"]]
        if a["window_signature"] != w["window_signature"]:
            raise ValueError("Window signature mismatch")
        key, text = short_id(w["id"]), w["raw_text"]
        labels = ["streamer"] * len(text)
        for field, author in [("viewer_segments", "viewer"), ("external_segments", "external"), ("uncertain_segments", "uncertain")]:
            for selection in patch[field].get(key, []):
                lo, hi = resolve(text, selection)
                labels[lo:hi] = [author] * (hi-lo)
        for u in a["uncertain_or_contested_spans"]:
            lo, hi = u["source_span"]["start_char"], u["source_span"]["end_char"]
            labels[lo:hi] = ["uncertain"] * (hi-lo)
        labels_by_id[w["id"]] = labels
        for author in ["streamer", "viewer", "external", "uncertain"]:
            annotations += selected(w, labels, {"start_char":0, "end_char":len(text)}, author)
    units, changes, audit = [], [], []
    ep = {"streamer_self_reported_fact":"self_report", "streamer_subjective_opinion":"opinion",
          "streamer_unverified_science_pop_or_hearsay":"hearsay"}
    for a in original:
        w = windows[a["window_id"]]; wid = w["id"]; key = short_id(wid); labels = labels_by_id[wid]
        base = {"window_id":wid, "source_id":w["source_id"], "source_file":w["source_file"],
                "source_sha256":w["source_sha256"], "window_signature":w["window_signature"],
                "speaker_id":w["speaker_id"], "speaker_verified":False,
                "verification_method":"subtitle_semantic_review", "review_status":"text_reviewed",
                "start_ms":w["start_ms"], "end_ms":w["end_ms"]}
        units.append(dict(base, id=wid, layer=1, kind="context", author="context", text=w["raw_text"],
                          evidence=[], context_evidence=[], epistemic_status="transcript", trigger_evidence=[]))
        for section, kind, id_field, text_field, author, layer in [
            ("streamer_facts","fact","fact_id","claim","streamer",2),
            ("viewer_submissions","viewer","submission_id","viewer_experience_summary","viewer",1),
            ("streamer_spoken_quotes","quote","quote_id","verbatim_quote","streamer",3),
            ("interaction_examples","interaction","interaction_id","viewer_trigger_summary","streamer",3)]:
            for i, item in enumerate(a[section],1):
                rid, pkey = item[id_field], f"{key}.{i}"
                source = item.get("streamer_response_span" if kind == "interaction" else "source_span")
                evidence = selected(w, labels, source, author)
                if kind == "fact" and pkey in patch["fact_evidence"]:
                    evidence = []
                    for sel in patch["fact_evidence"][pkey]:
                        lo,hi=resolve(w["raw_text"],sel)
                        evidence += selected(w,labels,{"start_char":lo,"end_char":hi},author)
                text = item[text_field]
                if kind == "fact":
                    text = patch["fact_text"].get(pkey, text)
                if kind == "viewer":
                    text = patch["viewer_text"].get(pkey, text)
                # Quotes remain arrays of exact original slices; no synthetic joined quote.
                if kind == "quote":
                    text = item.get("speech_act", "") + " " + "、".join(item.get("style_tags", []))
                unit = dict(base, id=rid, kind=kind, layer=layer, author=author, text=text,
                            evidence=evidence, context_evidence=context(w, source), trigger_evidence=[],
                            epistemic_status=ep.get(item.get("epistemic_status"), "expression" if layer==3 else "viewer_report"))
                if kind == "fact":
                    unit["epistemic_status"] = patch["fact_epistemic"].get(pkey,unit["epistemic_status"])
                if not evidence:
                    unit["review_status"] = "context_only"
                if kind == "viewer":
                    unit["evidence_kind"] = "direct" if evidence else "indirect"
                    if key == "e1" and i == 1:
                        unit["evidence_kind"] = "streamer_report_with_viewer_question"
                if kind == "interaction":
                    unit["trigger_evidence"] = selected(w,labels,item.get("viewer_trigger_span"),"viewer")
                    unit["trigger_context"] = context(w,item.get("viewer_trigger_span"))
                    cross = item.get("cross_window_trigger_ref")
                    if cross:
                        if cross not in windows:
                            raise ValueError("Missing cross-window trigger")
                        unit["cross_window_context_id"] = cross
                    if key == "e14" and i == 1:
                        cw=windows[cross];lo,hi=resolve(cw["raw_text"],[120,122])
                        unit["trigger_evidence"] = selected(cw, labels_by_id[cross], {"start_char":lo,"end_char":hi}, "viewer")
                        unit["trigger_context"] = context(cw,{"start_char":lo,"end_char":hi})
                    if key == "e1" and i == 1:
                        unit["text"] = patch["viewer_text"]["e1.1"]
                    if key == "e14" and i == 2:
                        unit["text"] = patch["viewer_text"]["e14.1"]
                    unit["trigger_kind"] = "direct" if unit["trigger_evidence"] else "context_inferred"
                    unit["model_analysis"] = item["interaction_mechanism_model_analysis"]
                    unit["scenario"] = item["scenario_category"]
                if key == "e1" and kind=="quote" and i==3:
                    unit["epistemic_status"]="hearsay"
                # Changed boundaries are explicit and inspectable.
                if source and [(s["start_char"],s["end_char"]) for s in evidence] != [(source["start_char"],source["end_char"])]:
                    changes.append({"id":rid,"old_span":source,"new_spans":[{k:s[k] for k in ("window_id","start_char","end_char")} for s in evidence]})
                units.append(unit)
        for item in a["v1_excerpts_audit"]:
            audit.append({"window_id":wid,"excerpt_index":item["excerpt_index"],
                          "original_text":item["verbatim_text"],
                          "streamer_evidence":selected(w,labels,item["source_span"],"streamer"),
                          "viewer_evidence":selected(w,labels,item["source_span"],"viewer"),
                          "uncertain_evidence":selected(w,labels,item["source_span"],"uncertain")})
    report = validate(units, windows, annotations)
    report["changed_boundaries"] = len(changes)
    return units, annotations, changes, audit, report


def validate(units, windows, annotations):
    errors, counts, seen = [], Counter(), set()
    def check(s):
        w = windows.get(s.get("window_id"))
        if w is None:
            errors.append("Unknown evidence window"); return
        text=w["raw_text"];lo=s.get("start_char",-1);hi=s.get("end_char",-1)
        if not (0<=lo<hi<=len(text)) or text[lo:hi]!=s.get("text"):
            errors.append("Evidence text/bounds mismatch"); return
        if s.get("line_start")!=text.count("\n",0,lo)+1 or s.get("line_end")!=text.count("\n",0,hi-1)+1:
            errors.append("Evidence line mismatch")
    maps = {wid:[None]*len(w["raw_text"]) for wid,w in windows.items()}
    for a in annotations:
        check(a)
        if a["window_id"] not in maps: continue
        for pos in range(a["start_char"],a["end_char"]):
            if maps[a["window_id"]][pos] is not None:
                errors.append("Overlapping author annotations");break
            maps[a["window_id"]][pos]=a["author"]
    for u in units:
        if u["id"] in seen: errors.append("Duplicate record ID")
        seen.add(u["id"]);counts[u["kind"]]+=1
        expected={"context":(1,"context"),"viewer":(1,"viewer"),"fact":(2,"streamer"),"quote":(3,"streamer"),"interaction":(3,"streamer")}
        if expected.get(u["kind"]) != (u["layer"],u["author"]):
            errors.append("Record kind/layer/author mismatch")
        if u["review_status"] not in ("text_reviewed","context_only"):
            errors.append("Unknown review status")
        w=windows.get(u["window_id"])
        if not w or u["window_signature"] != w["window_signature"] or u["source_sha256"] != w["source_sha256"]:
            errors.append("Record provenance mismatch")
        if u["speaker_verified"] is not False:
            errors.append("Unsupported audio verification")
        if u["kind"] in ("fact","quote","interaction") and u["review_status"]=="text_reviewed" and not u["evidence"]:
            errors.append("Eligible record lacks direct evidence")
        for field in ("evidence","trigger_evidence","context_evidence","trigger_context"):
            for s in u.get(field,[]):
                check(s)
                if field in ("context_evidence","trigger_context"): continue
                required="viewer" if field=="trigger_evidence" else u["author"]
                if s["author"] != required: errors.append("Wrong evidence author")
                for pos in range(s["start_char"],s["end_char"]):
                    if not windows[s["window_id"]]["raw_text"][pos].isspace() and maps[s["window_id"]][pos] != required:
                        errors.append(f"Attribution conflict: {u['id']}");break
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return {"windows":len(windows),"records":len(units),"counts":dict(counts),"conflicts":0,
            "exact_evidence_spans":sum(len(u.get(f,[])) for u in units for f in ("evidence","trigger_evidence")),
            "context_only_records":sum(u["review_status"]=="context_only" for u in units)}


def repair(windows_path, attribution_path, patch_path, output_dir):
    units, annotations, changes, audit, report = prepare(windows_path,attribution_path,patch_path)
    dest=Path(output_dir)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Repair output must be empty; choose a new revision directory")
    dest.mkdir(parents=True,exist_ok=True)
    for name,value in [("records.jsonl",units),("author_spans.jsonl",annotations),("boundary_changes.jsonl",changes),("v1_audit.jsonl",audit)]:
        (dest/name).write_text("".join(json.dumps(v,ensure_ascii=False)+"\n" for v in value),encoding="utf-8")
    report.update({"schema_version":2,"source_windows_sha256":digest(windows_path),
                   "source_attribution_sha256":digest(attribution_path),"semantic_patch_sha256":digest(patch_path),
                   "records_sha256":digest(dest/"records.jsonl"),"author_spans_sha256":digest(dest/"author_spans.jsonl")})
    (dest/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

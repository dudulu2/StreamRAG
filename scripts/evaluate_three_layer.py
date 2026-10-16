"""Run disclosed retrieval regressions and layer-isolation checks with local Qwen."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from streamrag.attribution import read_jsonl
from streamrag.local_embedding import LocalQwenEmbedding
from streamrag.three_layer import search

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,default=ROOT/"data/lancedb/nailv_three_layer_v1")
    p.add_argument("--revision",type=Path,default=ROOT/"data/attribution/nailv_20261016/revision_02")
    p.add_argument("--gold",type=Path,default=ROOT/"data/attribution/nailv_20261016/golden_queries.json")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cases=json.loads(args.gold.read_text(encoding="utf-8"))["test_cases"]
    units={u["id"]:u for u in read_jsonl(args.revision/"records.jsonl")}
    for c in cases:
        if not set(c["expected_ids"])<=units.keys():raise ValueError("Missing expected evidence ID")
    embed=LocalQwenEmbedding(ROOT/"qwen",device="auto")
    results=[]
    for c in cases:
        started=time.perf_counter()
        rows=search(args.index,c["query"],embed,scope=c["scope"],top_k=5)
        ids=[r["id"] for r in rows]
        rank=next((i for i,rid in enumerate(ids,1) if rid in c["expected_ids"]),None)
        results.append(dict(c,retrieved_ids=ids,first_relevant_rank=rank,
                            seconds=round(time.perf_counter()-started,3)))
        print(c["id"],rank,flush=True)
    checks={}
    rows=search(args.index,"孟加拉达卡工作三年",embed,scope="facts",top_k=5)
    checks["facts_exclude_viewer_records"]=all(r["kind"]=="fact" and r["author"]=="streamer" for r in rows)
    rows=search(args.index,"断骨增高百分之六十",embed,scope="style",top_k=30)
    checks["style_excludes_hearsay"]=all(r["epistemic_status"]!="hearsay" for r in rows)
    checks["context_not_injected_in_fact_style"]=all("context_evidence" not in r for r in rows)
    rows=search(args.index,"雪松香味",embed,scope="facts",min_similarity=1.0)
    checks["strict_threshold_returns_empty"]=not rows
    # A negative query is reported, not assigned an uncalibrated production threshold.
    unknown=search(args.index,"明前奶绿的护照号码是多少？",embed,scope="facts",top_k=3)
    report={"suite":"development_regression","queries":len(results),"k":5,
            "hit_at_5":sum(r["first_relevant_rank"] is not None for r in results)/len(results),
            "mrr_at_5":sum(1/r["first_relevant_rank"] if r["first_relevant_rank"] else 0 for r in results)/len(results),
            "checks":checks,"results":results,
            "unanswerable_probe":{"query":"明前奶绿的护照号码是多少？","expected":"no_supported_answer",
                                  "candidate_ids":[r["id"] for r in unknown],
                                  "candidate_similarities":[r["cosine_similarity"] for r in unknown]},
            "answer_generation_tested":False,"independent_holdout":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("queries","hit_at_5","mrr_at_5","checks")},ensure_ascii=False))
    if not all(checks.values()) or report["hit_at_5"]<1:sys.exit(1)

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

"""Repair, validate, index and query the attributed corpus."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from streamrag.attribution import repair
from streamrag.three_layer import load_reviewed,build,search
from streamrag.local_embedding import LocalQwenEmbedding


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["repair","validate","index","query"])
    parser.add_argument("--windows",type=Path,default=ROOT/"corpus_sources/gemini_batches/nailv_20261016/windows.jsonl")
    parser.add_argument("--original",type=Path,default=ROOT/"gemini_output/gemini_output/gemini_output/nailv_20261016/attribution/nailv_three_layer_attribution.jsonl")
    parser.add_argument("--patch",type=Path,default=ROOT/"data/attribution/nailv_20261016/semantic_patch.json")
    parser.add_argument("--revision",type=Path,default=ROOT/"data/attribution/nailv_20261016/revision_02")
    parser.add_argument("--index",type=Path,default=ROOT/"data/lancedb/nailv_three_layer_v1")
    parser.add_argument("--model-path",type=Path,default=ROOT/"qwen")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--scope",choices=["context","facts","style","attribution"],default="facts")
    parser.add_argument("--epistemic")
    parser.add_argument("--query",default="")
    parser.add_argument("--top-k",type=int,default=5)
    parser.add_argument("--min-similarity",type=float)
    args=parser.parse_args()
    if args.action=="repair": result=repair(args.windows,args.original,args.patch,args.revision)
    elif args.action=="validate": result=load_reviewed(args.revision,args.windows)[1]
    else:
        embed=LocalQwenEmbedding(args.model_path,device=args.device)
        if args.action=="index":result=build(args.revision,args.windows,args.index,embed,speaker_aliases={"nailv":["明前奶绿"]})
        else:result=search(args.index,args.query,embed,scope=args.scope,top_k=args.top_k,epistemic=args.epistemic,min_similarity=args.min_similarity)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

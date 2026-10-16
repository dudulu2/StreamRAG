"""Real local model -> temporary LanceDB -> hybrid search. Synthetic corpus only."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streamrag.local_embedding import LocalQwenEmbedding
from streamrag.corpus import prepare, write_jsonl
from streamrag.retrieval import build_index, search


def main():
    client = LocalQwenEmbedding(os.getenv("STREAM_RAG_EMBED_MODEL_PATH",
                               str(Path(__file__).resolve().parents[1] / "qwen")))
    examples = [
        ("comfort", "朋友因为工作出了差错很难过。主播先听完，告诉他一次失误不代表没有能力，再一起想怎么补救。", "事情没做好，觉得自己很没用，应该怎么安慰？"),
        ("cooking", "主播做番茄鸡蛋面，先炒鸡蛋，再炒番茄出汁，加入水和面条，最后调味。", "晚饭想吃面条，有什么简单做法？"),
        ("game", "主播玩射击游戏，强调用掩体保护自己，注意换弹时间，和队友配合，不要独自冲锋。", "打枪总被敌人击倒，如何提高生存率？"),
    ]
    started = time.perf_counter()
    vectors = client.embed([text for _, text, _ in examples])
    load_and_encode = time.perf_counter() - started
    report = {"synthetic_smoke_only": True, "offline": True,
              "model": client.model, "device": str(client._encoder.device),
              "dimensions": len(vectors[0]), "load_and_3_documents_seconds": round(load_and_encode, 3),
              "queries": []}
    with tempfile.TemporaryDirectory(prefix="streamrag-qwen-smoke-") as td:
        root = Path(td)
        rows = []
        for name, text, _ in examples:
            source = root / (name + ".txt")
            source.write_text(text, encoding="utf-8")
            row = prepare(source, root / (name + ".jsonl"), "smoke", source_id=name)[0]
            row.update(keep=True, summary=text, details=text)
            rows.append(row)
        write_jsonl(root / "refined.jsonl", rows)
        build_index(root / "refined.jsonl", root / "db", client)
        for expected, _, query in examples:
            start = time.perf_counter()
            vector = client.embed_query(query)
            similarities = [sum(a*b for a, b in zip(vector, doc)) for doc in vectors]
            vector_best = examples[max(range(len(vectors)), key=similarities.__getitem__)][0]
            results = search(root / "db", query, client, top_k=1)
            actual = results[0]["source_id"]
            report["queries"].append({"expected": expected, "vector_top1": vector_best,
                                      "hybrid_top1": actual, "seconds": round(time.perf_counter()-start, 3)})
            if actual != expected or vector_best != expected:
                raise RuntimeError(json.dumps(report, ensure_ascii=False))
    import torch
    if torch.cuda.is_available():
        report["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated()/1024**2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Fast regression tests for the isolated Eidolon-style stream corpus pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from streamrag import corpus, retrieval  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.seen = []

    def refine(self, window):
        self.seen.append(window["id"])
        return {
            "keep": True,
            "topic": "朋友之间发生误会",
            "summary": "主播回应误会并建议澄清",
            "details": "先回应听众的情绪，再讨论如何直接询问对方的想法。",
            "topics": ["友情", "误会"], "scene_summaries": ["从困惑到主动沟通"],
            "activity": "chat", "risk_flags": []
        }


class CorpusTests(unittest.TestCase):
    def test_srt_five_minute_windows_with_real_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "recording.srt"
            file.write_text(
                "1\n00:00:10,000 --> 00:00:15,000\n我今天有点困。\n\n"
                "2\n00:04:59,000 --> 00:05:00,000\n你们呢？\n\n"
                "3\n00:05:02,000 --> 00:05:06,000\n有人想聊点什么？\n",
                encoding="utf-8")
            output = Path(temp) / "windows.jsonl"
            windows = corpus.prepare(file, output, "test_broadcaster",
                                     source_id="srt_001")
            self.assertEqual(2, len(windows))
            self.assertEqual("five_minutes", windows[0]["window_strategy"])
            self.assertEqual((10000, 300000),
                             (windows[0]["start_ms"], windows[0]["end_ms"]))
            self.assertEqual(302000, windows[1]["start_ms"])
            self.assertEqual(windows[0]["raw_text"],
                             "我今天有点困。\n你们呢？")
            self.assertNotIn("有人想聊", windows[0]["raw_text"])
            self.assertFalse(windows[0]["speaker_verified"])
            self.assertEqual(windows, corpus.read_jsonl(output))

    def test_plaintext_character_windows_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "long.txt"
            original = ("刚才我们聊到工作上的误会。" * 135
                        + "\n换个话题，假如你突然心情不好怎么办？" * 50)
            source.write_text(original, encoding="utf-8")
            output = Path(temp) / "w.jsonl"
            windows = corpus.prepare(source, output, "single", max_chars=900)
            self.assertGreater(len(windows), 2)
            self.assertTrue(all(w["start_ms"] is None for w in windows))
            self.assertTrue(all(w["source_sha256"] == corpus.digest(original)
                                for w in windows))
            self.assertEqual(len(set(x["id"] for x in windows)), len(windows))
            self.assertTrue(all(len(w["raw_text"]) <= 925 for w in windows))
            self.assertTrue(all(not w["speaker_verified"] for w in windows))

    def test_input_speaker_avoid_mixing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mixed.jsonl"
            corpus.write_jsonl(path, [
                {"speaker": "nailv", "text": "先聊点别的"},
                {"speaker": "other", "text": "我也来聊"},
                {"speaker": "nailv", "text": "再来说工作"}
            ])
            segments, _ = corpus.parse_source(path, "nailv")
            self.assertEqual(["先聊点别的", "再来说工作"],
                             [x["text"] for x in segments])
            self.assertTrue(all(x["speaker_explicit"] for x in segments))

    def test_refine_resume_and_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text(("今天我们聊朋友间的误会。应该怎么沟通。" * 70),
                              encoding="utf-8")
            windows = root / "windows.jsonl"
            refined = root / "refined.jsonl"
            catalog_path = root / "catalog.json"
            corpus.prepare(source, windows, "nailv", max_chars=900)
            fake = FakeLLM()
            first = corpus.refine(windows, refined, fake, limit=1)
            self.assertEqual(1, first["refined"])
            second = corpus.refine(windows, refined, fake)
            self.assertEqual(len(corpus.read_jsonl(windows)),
                             1 + second["refined"])
            self.assertEqual(len(corpus.read_jsonl(windows)), len(fake.seen))
            self.assertEqual(len(fake.seen),
                             corpus.refine(windows, refined, fake)["resume_skipped"])
            catalog = corpus.build_catalog(refined, catalog_path)
            self.assertEqual(1, len(catalog["broadcasts"]))
            self.assertEqual(len(fake.seen), catalog["broadcasts"][0]["total_windows"])
            report = corpus.quality_report(refined, root / "quality.json")
            self.assertEqual(len(fake.seen), report["kept"])
            for r in corpus.read_jsonl(refined):
                self.assertFalse(r["persona_memory_allowed"])
                self.assertEqual("auto_candidate", r["review_status"])
                self.assertEqual("朋友之间发生误会", r["topic"])
                self.assertTrue(r["raw_text"].strip())
                self.assertEqual(corpus.digest(source.read_text(encoding="utf-8")),
                                 r["source_sha256"])

    def test_refinement_rejects_fabrication_friendly_empty_outputs(self):
        with self.assertRaises(ValueError):
            corpus.validate_refinement({"keep": True, "summary": "yes", "details": ""})
        with self.assertRaises(ValueError):
            corpus.validate_refinement({"keep": "yes", "summary": "x", "details": "y"})
        self.assertFalse(corpus.validate_refinement({"keep": False})["keep"])

    def test_hash_test_embeddings_are_deterministic(self):
        embed = retrieval.EmbeddingClient("hash-test", "ignored")
        original = retrieval.terms
        retrieval.terms = lambda s: [s[i:i+2] for i in range(max(0, len(s)-1))]
        try:
            one = embed.embed("我好累，今天工作做不完了")
            self.assertEqual(one, embed.embed("我好累，今天工作做不完了"))
            self.assertEqual(128, len(one))
            self.assertNotEqual(one, embed.embed("天气真好，太阳出来了"))
        finally:
            retrieval.terms = original

    def test_rrf_keeps_distinct_documents(self):
        vec = [{"id": "b"}, {"id": "a"}]
        fts = [{"id": "a"}, {"id": "c"}]
        merged = retrieval.rrf_merge(vec, fts, top_count=3)
        self.assertEqual(["a", "b", "c"], [x["item"]["id"] for x in merged])
        self.assertGreater(merged[0]["score"], merged[1]["score"])


    def test_refine_rejects_changed_checkpoint_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "test.txt"
            windows = root / "windows.jsonl"
            refined = root / "refined.jsonl"
            source.write_text("昨天聊天时对方有点不高兴。", encoding="utf-8")
            corpus.prepare(source, windows, "solo", source_id="s1")
            corpus.refine(windows, refined, FakeLLM())
            source.write_text("今天聊天时对方有点不高兴。", encoding="utf-8")
            corpus.prepare(source, windows, "solo", source_id="s1")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                corpus.refine(windows, refined, FakeLLM())

    def test_lancedb_ingest_and_local_hybrid_search(self):
        try:
            import lancedb  # noqa: F401
            import jieba  # noqa: F401
        except ImportError:
            self.skipTest("LanceDB/jieba not installed in this Python environment")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sample.txt"
            source.write_text(
                "朋友之间遇到误会不要急着争吵。先认真问清楚原因再去理解对方。" * 26
                + "\n"
                + "去公园散步吧。今天天气很好，天空很蓝，微风很舒服。" * 25,
                encoding="utf-8")
            wpath, rpath, dbpath = root / "w.jsonl", root / "r.jsonl", root / "lance"
            corpus.prepare(source, wpath, "solo", source_id="s1", max_chars=800)
            corpus.refine(wpath, rpath, FakeLLM())
            client = retrieval.EmbeddingClient("hash-test", "not-used")
            manifest = retrieval.build_index(rpath, dbpath, client, batch_size=3)
            self.assertEqual(len(corpus.read_jsonl(rpath)), manifest["row_count"])
            self.assertEqual("solo", manifest["speaker_id"])
            lex = retrieval.search(dbpath, "朋友误会", None, lexical_only=True, top_k=3)
            self.assertGreaterEqual(len(lex), 1)
            both = retrieval.search(dbpath, "朋友误会", client, top_k=3)
            self.assertGreaterEqual(len(both), 1)
            gold = root / "gold.jsonl"
            corpus.write_jsonl(gold, [{"query": "朋友误会",
                                       "relevant_ids": [lex[0]["id"]]}])
            evaluation = retrieval.evaluate(dbpath, gold, None, lexical_only=True)
            self.assertEqual(1.0, evaluation["recall_at_k"])

if __name__ == "__main__":
    unittest.main()

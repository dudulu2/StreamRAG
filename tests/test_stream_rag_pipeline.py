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


    def test_checkpoint_rejects_changed_partition_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = root / "source.txt"
            text.write_text("We were talking about everyday topics. " * 100, encoding="utf-8")
            windows, refined = root / "w.jsonl", root / "r.jsonl"
            corpus.prepare(text, windows, "solo", max_chars=420)
            corpus.refine(windows, refined, FakeLLM())
            corpus.prepare(text, windows, "solo", max_chars=1000)
            with self.assertRaisesRegex(ValueError, "Orphaned|Partition/content"):
                corpus.refine(windows, refined, FakeLLM())
            with self.assertRaisesRegex(ValueError, "Incomplete|Stale"):
                corpus.check_refined_complete(windows, refined)

    def test_checkpoint_rejects_same_boundaries_with_different_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "one.txt"
            source.write_text("Hello there! How is your week?", encoding="utf-8")
            windows, refined = root / "w.jsonl", root / "r.jsonl"
            corpus.prepare(source, windows, "solo", max_chars=450)
            corpus.refine(windows, refined, FakeLLM())
            corpus.prepare(source, windows, "solo", max_chars=550)
            with self.assertRaisesRegex(ValueError, "Partition/content"):
                corpus.refine(windows, refined, FakeLLM())

    def test_rejects_tampered_prepared_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows = root / "one.txt", root / "w.jsonl"
            source.write_text("This conversation is about working with friends.", encoding="utf-8")
            corpus.prepare(source, windows, "solo")
            original = corpus.read_jsonl(windows)
            original[0]["raw_text"] = "Fake transcript."
            corpus.write_jsonl(windows, original)
            with self.assertRaisesRegex(ValueError, "window changed"):
                corpus.refine(windows, root / "r.jsonl", FakeLLM())

    def test_failed_llm_window_keeps_running_then_recovers_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows, refined = root / "source.txt", root / "w.jsonl", root / "r.jsonl"
            source.write_text("Support your friends when they are upset. " * 90, encoding="utf-8")
            prepared = corpus.prepare(source, windows, "solo", max_chars=500)
            self.assertGreaterEqual(len(prepared), 2)
            failed_id = prepared[0]["id"]
            normal = FakeLLM()

            class Flaky:
                model = "stable-test-model"

                def refine(self, window):
                    if window["id"] == failed_id:
                        raise TimeoutError("temporary connection timeout")
                    return normal.refine(window)

            first = corpus.refine(windows, refined, Flaky(), retries=0)
            self.assertEqual(1, first["errors"])
            self.assertEqual(len(prepared) - 1, first["refined"])
            error_file = refined.with_suffix(".errors.jsonl")
            self.assertTrue(error_file.exists())
            self.assertEqual([failed_id], first["pending_failed_ids"])
            self.assertEqual(failed_id, corpus.read_jsonl(error_file)[0]["id"])
            self.assertEqual("TimeoutError", corpus.read_jsonl(error_file)[0]["exception"])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                corpus.check_refined_complete(windows, refined)
            restored_client = FakeLLM()
            restored_client.model = "stable-test-model"
            restored = corpus.refine(windows, refined, restored_client, retries=0)
            self.assertEqual(1, restored["refined"])
            self.assertEqual([], restored["pending_failed_ids"])
            self.assertFalse(error_file.exists())
            self.assertEqual(len(prepared), corpus.check_refined_complete(windows, refined))

    def test_http_chat_client_invalid_responses_are_retryable_validation_errors(self):
        import io
        from unittest.mock import patch
        client = corpus.ChatClient("https://example.com/v1", "testing", "placeholder")

        for bad in (b'{"choices":[]}',
                    b'{"choices":[{"message":{"content":null}}]}',
                    b'{"choices":[{"message":null}]}'):
            with self.subTest(payload=bad):
                with patch("streamrag.corpus.urllib.request.urlopen",
                           return_value=io.BytesIO(bad)):
                    with self.assertRaises(ValueError):
                        client.refine({"raw_text": "test"})

    def test_jsonl_foreign_speaker_creates_discontinuity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows = root / "chat.jsonl", root / "w.jsonl"
            corpus.write_jsonl(source, [
                {"speaker_id": "target", "text": "Before disagreement", "start": 1.0},
                {"speaker_id": "someone_else", "text": "Interrupted", "start": 2.0},
                {"speaker_id": "target", "text": "After disagreement", "start": 3.0}
            ])
            rows = corpus.prepare(source, windows, "target", max_chars=800)
            self.assertEqual(2, len(rows))
            self.assertEqual("", rows[0]["context_after"])
            self.assertEqual("", rows[1]["context_before"])
            self.assertTrue(rows[1]["discontinuous_before"])
            self.assertNotIn("Interrupted", " ".join(r["raw_text"] for r in rows))

    def test_txt_uses_exact_original_source_slice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows = root / "lines.txt", root / "w.jsonl"
            original = "A short thought.\n\n  Another thought with spaces!  \n\nA third line."
            source.write_text(original, encoding="utf-8")
            rows = corpus.prepare(source, windows, "target")
            self.assertEqual(1, len(rows))
            self.assertEqual(original[rows[0]["start_char"]:rows[0]["end_char"]],
                             rows[0]["raw_text"])
            self.assertIn("\n\n", rows[0]["raw_text"])

    def test_embedding_document_includes_bounded_raw_evidence(self):
        record = {"topic": "friendship", "summary": "A discussion",
                  "details": "A conclusion", "raw_text": "specific original phrase " * 400}
        doc = retrieval.embedding_document(record)
        self.assertIn("specific original phrase", doc)
        self.assertLessEqual(len(doc), 6500)

    def test_evaluation_recall_and_hit_rate_are_distinct(self):
        import jieba  # noqa: F401
        import lancedb  # noqa: F401
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sample.txt"
            source.write_text("Friends share their fears and hopes. " * 150,
                              encoding="utf-8")
            wpath, rpath, dbpath, gold = (
                root / "w.jsonl", root / "r.jsonl", root / "db", root / "gold.jsonl")
            corpus.prepare(source, wpath, "solo", max_chars=500)
            corpus.refine(wpath, rpath, FakeLLM())
            client = retrieval.EmbeddingClient("hash-test", "ignored")
            retrieval.build_index(rpath, dbpath, client)
            lex = retrieval.search(dbpath, "朋友误会", None, lexical_only=True, top_k=2)
            self.assertGreaterEqual(len(lex), 2)
            corpus.write_jsonl(gold, [
                {"query": "朋友误会", "relevant_ids": [x["id"] for x in lex[:2]]}
            ])
            metrics = retrieval.evaluate(dbpath, gold, None, lexical_only=True, top_k=1)
            self.assertEqual(0.5, metrics["recall_at_k"])
            self.assertEqual(1.0, metrics["hit_rate_at_k"])

            class OppositeReranker:
                def __init__(self):
                    self.calls = 0

                def rerank(self, query, merged, top_k):
                    self.calls += 1
                    return list(reversed(merged))[:top_k]

            reranker = OppositeReranker()
            metrics = retrieval.evaluate(dbpath, gold, None,
                                         lexical_only=True, top_k=1,
                                         reranker=reranker)
            self.assertEqual(1, reranker.calls)
            self.assertTrue(metrics["reranker_enabled"])

    def test_empty_reranker_result_falls_back_to_rrf(self):
        import jieba  # noqa: F401
        import lancedb  # noqa: F401
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sample.txt"
            source.write_text("Discussing how people resolve everyday conflicts. " * 35,
                              encoding="utf-8")
            wpath, rpath, db = root / "w.jsonl", root / "r.jsonl", root / "index"
            corpus.prepare(source, wpath, "solo", max_chars=500)
            corpus.refine(wpath, rpath, FakeLLM())
            retrieval.build_index(rpath, db, retrieval.EmbeddingClient("hash-test", "x"))
            normal = retrieval.search(db, "朋友误会", None, lexical_only=True)

            class Empty:
                def rerank(self, query, merged, top_k):
                    return []

            with self.assertWarnsRegex(RuntimeWarning, "Reranker failed"):
                fallback = retrieval.search(
                    db, "朋友误会", None, lexical_only=True, reranker=Empty())
            self.assertEqual([r["id"] for r in normal],
                             [r["id"] for r in fallback])
            self.assertTrue(all(row["reranker_status"] == "fallback"
                                for row in fallback))
            self.assertTrue(all(row["reranker_status"] == "disabled"
                                for row in normal))




    def test_successful_publish_is_not_reported_as_failed_when_backup_locked(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing, staging = root / "db", root / "staged"
            existing.mkdir()
            staging.mkdir()
            (existing / "old.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            real_rmtree = retrieval.shutil.rmtree

            def deny_backup_cleanup(path, *args, **kwargs):
                if ".previous-" in str(path):
                    raise PermissionError("Old files are still open")
                return real_rmtree(path, *args, **kwargs)

            with patch("streamrag.retrieval.shutil.rmtree",
                       side_effect=deny_backup_cleanup):
                with self.assertWarnsRegex(RuntimeWarning, "published successfully"):
                    retrieval._publish_index(staging, existing)
            self.assertEqual("new", (existing / "new.txt").read_text(encoding="utf-8"))
            backups = list(root.glob(".db.previous-*"))
            self.assertEqual(1, len(backups))
            self.assertEqual("old", (backups[0] / "old.txt").read_text(encoding="utf-8"))

    def test_failed_index_publish_restores_existing_directory(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / "db"
            staging = root / "new-staging"
            existing.mkdir()
            staging.mkdir()
            (existing / "old.txt").write_text("safe", encoding="utf-8")
            (staging / "new.txt").write_text("incomplete", encoding="utf-8")
            original_rename = Path.rename

            def fail_new_publication(path, target):
                if path == staging and target == existing:
                    raise OSError("simulated failed publication")
                return original_rename(path, target)

            with patch.object(Path, "rename", new=fail_new_publication):
                with self.assertRaisesRegex(OSError, "failed publication"):
                    retrieval._publish_index(staging, existing)
            self.assertEqual("safe", (existing / "old.txt").read_text(encoding="utf-8"))
            self.assertFalse((existing / "new.txt").exists())
            self.assertEqual([], list(root.glob(".db.previous-*")))

    def test_rebuild_index_preserves_old_on_embedding_error(self):
        import jieba  # noqa: F401
        import lancedb  # noqa: F401
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows, refined, index = (
                root / "sample.txt", root / "w.jsonl", root / "r.jsonl", root / "db")
            source.write_text("Work with patience. " * 120, encoding="utf-8")
            corpus.prepare(source, windows, "solo", max_chars=500)
            corpus.refine(windows, refined, FakeLLM())
            client = retrieval.EmbeddingClient("hash-test", "not-used")
            before = retrieval.build_index(refined, index, client)
            self.assertGreater(before["row_count"], 0)

            class BrokenEmbedding(retrieval.EmbeddingClient):
                def embed(self, texts):
                    raise RuntimeError("mock remote embedding unavailable")

            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                retrieval.build_index(refined, index, BrokenEmbedding("hash-test", "x"))
            after = retrieval.open_index(index)[1]
            self.assertEqual(before, after)
            self.assertTrue(retrieval.search(
                index, "朋友误会", None, lexical_only=True))

            new_manifest = retrieval.build_index(refined, index, client)
            self.assertEqual(before, new_manifest)
            self.assertEqual([], list(root.glob(".db.building-*")))

    def test_eval_fails_when_reranker_does_not_work(self):
        import jieba  # noqa: F401
        import lancedb  # noqa: F401
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text("Talking about friendship. " * 110, encoding="utf-8")
            windows, refined, index, gold = (
                root / "w.jsonl", root / "r.jsonl", root / "index", root / "gold.jsonl")
            corpus.prepare(source, windows, "solo")
            corpus.refine(windows, refined, FakeLLM())
            retrieval.build_index(refined, index,
                                  retrieval.EmbeddingClient("hash-test", "x"))
            result = retrieval.search(index, "朋友误会", None, lexical_only=True)
            corpus.write_jsonl(gold, [{"query": "朋友误会",
                                       "relevant_ids": [result[0]["id"]]}])

            class BrokenReranker:
                def rerank(self, query, merged, top_k):
                    raise OSError("Rerank endpoint unavailable")

            with self.assertRaisesRegex(RuntimeError, "Reranker failed"):
                retrieval.evaluate(index, gold, None, reranker=BrokenReranker(),
                                   lexical_only=True)




    def test_truncated_checkpoint_tail_is_salvaged_for_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows, refined = root / "sample.txt", root / "w.jsonl", root / "r.jsonl"
            source.write_text("Learning to listen to a friend. " * 150, encoding="utf-8")
            prepared = corpus.prepare(source, windows, "solo", max_chars=500)
            corpus.refine(windows, refined, FakeLLM(), limit=1)
            with refined.open("ab") as handle:
                handle.write(b'{"id":"cut-off')
            result = corpus.refine(windows, refined, FakeLLM(), retries=0)
            self.assertEqual(len(prepared) - 1, result["refined"])
            self.assertEqual(1, result["resume_skipped"])
            self.assertEqual(len(prepared), corpus.check_refined_complete(windows, refined))
            self.assertEqual(b'{"id":"cut-off',
                             (root / "r.jsonl.truncated-tail").read_bytes())

    def test_corruption_in_complete_checkpoint_line_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows, refined = root / "sample.txt", root / "w.jsonl", root / "r.jsonl"
            source.write_text("Small everyday conversation.", encoding="utf-8")
            corpus.prepare(source, windows, "solo")
            refined.write_text('{"id":}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                corpus.refine(windows, refined, FakeLLM())
            self.assertEqual('{"id":}\n', refined.read_text(encoding="utf-8"))

    def test_unknown_jsonl_speakers_do_not_merge_into_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file, windows = root / "src.jsonl", root / "w.jsonl"
            corpus.write_jsonl(input_file, [
                {"speaker_id": "target", "text": "A first thought.", "start": 1.0},
                {"speaker_id": None, "text": "Unverified utterance.", "start": 2.0},
                {"speaker_id": "target", "text": "A second thought.", "start": 3.0},
                {"text": "Unknown without speaker key.", "start": 4.0},
                {"speaker_id": "target", "text": "A third thought.", "start": 5.0}
            ])
            prepared = corpus.prepare(input_file, windows, "target")
            self.assertEqual(3, len(prepared))
            self.assertTrue(all(w["speaker_verified"] for w in prepared))
            self.assertEqual(
                ["A first thought.", "A second thought.", "A third thought."],
                [w["raw_text"] for w in prepared])
            self.assertTrue(all(w["discontinuous_before"] for w in prepared[1:]))




    def test_modified_refined_raw_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, windows, refined = root / "sample.txt", root / "w.jsonl", root / "r.jsonl"
            source.write_text("A conversation about supporting a friend.",
                              encoding="utf-8")
            corpus.prepare(source, windows, "solo")
            corpus.refine(windows, refined, FakeLLM())
            rows = corpus.read_jsonl(refined)
            rows[0]["raw_text"] = "This was never in the transcript."
            corpus.write_jsonl(refined, rows)
            with self.assertRaisesRegex(ValueError, "raw_text"):
                corpus.check_refined_complete(windows, refined)
            with self.assertRaisesRegex(ValueError, "raw_text"):
                corpus.refine(windows, refined, FakeLLM())


if __name__ == "__main__":
    unittest.main()

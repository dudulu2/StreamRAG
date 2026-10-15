"""Regression tests for the API-free Gemini/Astra batch workflow."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streamrag import corpus, manual, retrieval


def make_analysis(window):
    return {
        "id": window["id"],
        "window_signature": window["window_signature"],
        "keep": True,
        "topic": "遇到工作挫折时怎样回应朋友",
        "summary": "主播谈到朋友因工作挫折而低落时，如何认真倾听和回应。",
        "details": (
            "这段直播先谈到朋友在工作中受到挫折的情境，指出急着给建议可能让对方更不舒服。"
            "接下来通过倾听和慢慢询问的方式解释，先把对方想说的话听完，"
            "等对方觉得被理解之后再继续讨论具体可行的做法，而不是只给一句空泛的鼓励。"
        ),
        "trigger_context": None,
        "dialogue_flow": [
            "先谈到朋友在工作中遇到挫折。",
            "随后解释为什么先听完对方的想法，再慢慢讨论下一步。"
        ],
        "interaction_notes": "先回应对方的感受，随后再从具体困扰进入可操作的讨论。",
        "verbatim_excerpts": ["先听对方说完"],
        "topics": ["工作", "沟通"],
        "scene_summaries": ["聊如何处理朋友遇到挫折后的对话"],
        "activity": "chat",
        "risk_flags": [],
    }


class ManualWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "stream.txt"
        self.source.write_text(
            "工作碰到挫折时，应该先听对方说完，再慢慢讨论可以做些什么。" * 160,
            encoding="utf-8"
        )
        self.windows = self.root / "windows.jsonl"
        self.prepared = corpus.prepare(self.source, self.windows, "solo",
                                       source_id="live01", max_chars=800)
        self.export_dir = self.root / "exchange"
        self.manifest = None

    def export(self):
        result = manual.export_batches(
            self.windows, self.export_dir, max_windows=2, max_chars=3600)
        self.manifest = json.loads(
            (self.export_dir / "manifest.json").read_text(encoding="utf-8"))
        return result

    def response_for(self, batch):
        all_rows = {w["id"]: w for w in self.prepared}
        return {
            "batch_id": batch["batch_id"],
            "prompt_sha256": self.manifest["prompt_sha256"],
            "items": [make_analysis(all_rows[id_]) for id_ in batch["window_ids"]],
        }

    def save_response(self, batch, filename=None, fenced=False):
        result = self.response_for(batch)
        path = (self.export_dir / "responses" /
                (filename or batch["batch_id"] + ".json"))
        content = json.dumps(result, ensure_ascii=False, indent=2)
        if fenced:
            fence = chr(96) * 3
            content = fence + "json\n" + content + "\n" + fence
        path.write_text(content, encoding="utf-8")
        return path

    def test_full_api_free_gemini_or_astra_workflow_and_rag_narrative(self):
        report = self.export()
        self.assertGreater(report["batches"], 1)
        request_text = next((self.export_dir / "requests").glob("*.md")).read_text(
            encoding="utf-8")
        self.assertIn("main_source_text", request_text)
        self.assertIn("dialogue_flow", request_text)
        self.assertIn("verbatim_excerpts", request_text)
        self.assertIn("先听对方说完", request_text)

        self.save_response(self.manifest["batches"][0], fenced=True)
        out = self.root / "refined.jsonl"
        first = manual.import_responses(
            self.windows, self.export_dir / "manifest.json",
            self.export_dir / "responses", out, "gemini4-web")
        self.assertEqual(2, first["added"])
        self.assertGreater(first["remaining"], 0)
        partial = manual.validate_import(self.windows, out)
        self.assertEqual("incomplete", partial["status"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            manual.validate_import(self.windows, out, strict=True)

        for batch in self.manifest["batches"][1:]:
            self.save_response(batch)
        second = manual.import_responses(
            self.windows, self.export_dir / "manifest.json",
            self.export_dir / "responses", out, "gemini4-web")
        self.assertEqual(len(self.prepared) - 2, second["added"])
        self.assertEqual(2, second["unchanged"])
        result = manual.validate_import(self.windows, out, strict=True)
        self.assertEqual("complete", result["status"])
        self.assertEqual(len(self.prepared), result["verbatim_quotes"])
        self.assertEqual(len(self.prepared) * 2, result["narrative_flow_steps"])
        self.assertEqual(len(self.prepared), corpus.check_refined_complete(
            self.windows, out))
        for row in corpus.read_jsonl(out):
            self.assertEqual("web_import_candidate", row["review_status"])
            self.assertFalse(row["persona_memory_allowed"])
            self.assertEqual("先听对方说完", row["verbatim_excerpts"][0])
            self.assertEqual(self.prepared[int(row["id"].split("__")[1])-1]["raw_text"],
                             row["raw_text"])

        embedder = retrieval.EmbeddingClient("hash-test", "only-for-test")
        db = self.root / "lancedb"
        retrieval.build_index(out, db, embedder)
        found = retrieval.search(db, "工作挫折", None, lexical_only=True)
        self.assertTrue(found)
        self.assertIn("先听对方说完", found[0]["reference_text"])
        self.assertIn("这段直播先谈到", found[0]["reference_text"])
        self.assertIn("原始直播文字", found[0]["reference_text"])
        self.assertEqual("来自原文的逐字摘录" in found[0]["reference_text"], True)
        self.assertEqual(["先听对方说完"], found[0]["verbatim_excerpts"])
        self.assertIn("对话参考", found[0]["reference_text"])

    def test_model_name_change_cannot_mix_two_web_processing_strategies(self):
        self.export()
        for batch in self.manifest["batches"]:
            self.save_response(batch)
        out = self.root / "result.jsonl"
        manual.import_responses(
            self.windows, self.export_dir / "manifest.json",
            self.export_dir / "responses", out, "astra-local")
        with self.assertRaisesRegex(ValueError, "another model"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", out, "gemini4-web")

    def test_fabricated_verbatim_quote_rejected_without_partial_write(self):
        self.export()
        first = self.manifest["batches"][0]
        response = self.response_for(first)
        response["items"][0]["verbatim_excerpts"] = ["这句话从来不在原文里"]
        (self.export_dir / "responses" / "bad.json").write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8")
        out = self.root / "refined.jsonl"
        with self.assertRaisesRegex(ValueError, "fabricated"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", out, "gemini4-web")
        self.assertFalse(out.exists())

    def test_category_labels_alone_are_rejected(self):
        self.export()
        response = self.response_for(self.manifest["batches"][0])
        response["items"][0]["details"] = "安慰，追问，转移话题。"
        (self.export_dir / "responses" / "short.json").write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "details too shallow"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", self.root / "refined.jsonl", "astra-local")

    def test_invalid_source_metadata_fields_rejected(self):
        self.export()
        response = self.response_for(self.manifest["batches"][0])
        response["items"][0]["raw_text"] = "Rewritten source text"
        (self.export_dir / "responses" / "bad.json").write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", self.root / "refined.jsonl", "astra-local")

    def test_incomplete_model_response_batch_rejected(self):
        self.export()
        response = self.response_for(self.manifest["batches"][0])
        response["items"].pop()
        (self.export_dir / "responses" / "missing.json").write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", self.root / "refined.jsonl", "gemini4-web")

    def test_stale_export_manifest_rejected_if_original_window_changes(self):
        self.export()
        self.save_response(self.manifest["batches"][0])
        self.source.write_text(
            "新文件内容，原先的片段已经被修改。" * 160, encoding="utf-8")
        corpus.prepare(self.source, self.windows, "solo",
                       source_id="live01", max_chars=800)
        with self.assertRaisesRegex(ValueError, "differ"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", self.root / "refined.jsonl",
                "gemini4-web")

    def test_changed_batch_layout_requires_new_export_directory(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "another batch job"):
            manual.export_batches(
                self.windows, self.export_dir, max_windows=1, max_chars=1000)

    def test_duplicate_response_files_cannot_double_import(self):
        self.export()
        self.save_response(self.manifest["batches"][0], "response1.json")
        self.save_response(self.manifest["batches"][0], "response2.json")
        with self.assertRaisesRegex(ValueError, "duplicate batch"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                self.export_dir / "responses", self.root / "refined.jsonl",
                "gemini4-web")

    def test_reimport_conflicting_json_rejected_without_corruption(self):
        self.export()
        path = self.save_response(self.manifest["batches"][0])
        out = self.root / "refined.jsonl"
        manual.import_responses(
            self.windows, self.export_dir / "manifest.json",
            path, out, "astra-local")
        before = out.read_bytes()
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["items"][0]["details"] += " 添加一段未经原文验证的内容。"
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                path, out, "astra-local")
        self.assertEqual(before, out.read_bytes())

    def test_model_prompt_signature_mismatch_rejected(self):
        self.export()
        path = self.save_response(self.manifest["batches"][0])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["prompt_sha256"] = "wrong"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prompt signature"):
            manual.import_responses(
                self.windows, self.export_dir / "manifest.json",
                path, self.root / "refined.jsonl", "gemini4-web")


    def test_public_cli_supports_complete_manual_flow_without_llm_api(self):
        import subprocess
        cli = Path(__file__).resolve().parents[1] / "scripts" / "stream_rag.py"

        def call(*args):
            proc = subprocess.run([sys.executable, str(cli), *map(str, args)],
                                  text=True, capture_output=True, timeout=35)
            self.assertEqual(0, proc.returncode,
                             msg=proc.stdout + "\n" + proc.stderr)
            return json.loads(proc.stdout)

        exports = call("export-batches", "--windows", self.windows,
                       "--outdir", self.export_dir, "--max-windows", "2",
                       "--max-chars", "3600")
        self.assertEqual(exports["windows"], len(self.prepared))
        self.manifest = json.loads(
            (self.export_dir / "manifest.json").read_text(encoding="utf-8"))
        for group in self.manifest["batches"]:
            self.save_response(group)

        refined = self.root / "cli-refined.jsonl"
        imported = call("import-refined", "--windows", self.windows,
                        "--manifest", self.export_dir / "manifest.json",
                        "--responses", self.export_dir / "responses",
                        "--output", refined, "--model", "astra-local")
        self.assertEqual(len(self.prepared), imported["total_refined"])
        report = call("validate", "--windows", self.windows,
                      "--refined", refined, "--strict")
        self.assertEqual("complete", report["status"])

        db = self.root / "db"
        created = call("ingest", "--windows", self.windows, "--refined", refined,
                       "--db", db, "--embed-mode", "hash-test",
                       "--allow-test-embeddings")
        self.assertEqual(len(self.prepared), created["row_count"])
        matches = call("search", "--db", db, "--query", "工作挫折",
                       "--lexical-only", "--embed-mode", "hash-test",
                       "--allow-test-embeddings")
        self.assertTrue(matches)
        self.assertIn("原始直播文字", matches[0]["reference_text"])



if __name__ == "__main__":
    unittest.main()

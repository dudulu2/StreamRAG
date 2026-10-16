"""Local embedding contracts; no real model or network needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streamrag.embedding_setup import DEFAULT_EMBED_MODEL, embedding_status, resolve_api_key
from streamrag.local_embedding import LocalQwenEmbedding, QUERY_INSTRUCTION
from streamrag import corpus, retrieval

class Encoder:
    device = "cpu"
    max_seq_length = 32768
    def __init__(self):
        self.calls = []
    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [[1] * len(t) for t in texts]}
    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        out = np.zeros((len(texts), 1024))
        out[:, 0] = 1.0
        return out

class EmbeddingSetupTests(unittest.TestCase):
    def client(self, folder):
        c = LocalQwenEmbedding(folder)
        c._encoder = Encoder()
        return c

    def test_local_default_no_key_or_network_required(self):
        self.assertEqual(DEFAULT_EMBED_MODEL, "Qwen/Qwen3-Embedding-0.6B")
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_api_key(), ("", "missing"))
            c = LocalQwenEmbedding(d)
            status = embedding_status(c)
            self.assertFalse(status["api_key_required"])
            self.assertFalse(status["ready_to_ingest"])
            with self.assertRaisesRegex(ValueError, "embedding-download"):
                c.embed("test")

    def test_query_has_instruction_documents_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.client(d)
            self.assertEqual(len(c.embed(["document"])[0]), 1024)
            self.assertEqual(len(c.embed_query("question")), 1024)
            self.assertEqual(c._encoder.calls[0][0], ["document"])
            self.assertEqual(c._encoder.calls[1][0], [f"Instruct: {QUERY_INSTRUCTION}\nQuery: question"])
            self.assertEqual(c._encoder.calls[0][1]["prompt"], "")
            self.assertTrue(c._encoder.calls[0][1]["normalize_embeddings"])
            self.assertTrue(embedding_status(c, live=True)["ready_to_ingest"])

    def test_bad_input_and_overflow_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.client(d)
            for value in ("", [], [""], [None]):
                with self.assertRaises(ValueError):
                    c.embed(value)
            c._encoder.max_seq_length = 2
            with self.assertRaisesRegex(ValueError, "token limit"):
                c.embed("long")
            self.assertFalse(c._encoder.calls)

    def test_wrong_model_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text('{"model_type":"other"}')
            with self.assertRaisesRegex(ValueError, "Expected Qwen3"):
                LocalQwenEmbedding(d).embed("hello")

    def test_partial_weight_download_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text('{"model_type":"qwen3","hidden_size":1024}')
            (root / "1_Pooling").mkdir()
            (root / "1_Pooling/config.json").write_text('{"pooling_mode_lasttoken":true}')
            (root / "model.safetensors.aria2").touch()
            client = LocalQwenEmbedding(d)
            self.assertFalse(embedding_status(client)["model_files_present"])
            with self.assertRaisesRegex(ValueError, "download is incomplete"):
                client.embed("hello")

    def test_invalid_vectors_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.client(d)
            for bad in (np.zeros((1, 1024)), np.ones((1, 3)), np.full((1, 1024), np.nan)):
                with patch.object(c._encoder, "encode", return_value=bad):
                    with self.assertRaisesRegex(ValueError, "Invalid local Qwen"):
                        c.embed("test")

    def test_index_contract_query_and_model_config_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            model = p / "model"
            model.mkdir()
            (model / "config.json").write_text('{"version":1}')
            c = self.client(model)
            (p / "source.txt").write_text("主播说今天煮番茄鸡蛋面。", encoding="utf-8")
            row = corpus.prepare(p / "source.txt", p / "windows.jsonl", "host")[0]
            row.update(keep=True, summary="煮面", details="主播讲如何煮面。")
            corpus.write_jsonl(p / "refined.jsonl", [row])
            manifest = retrieval.build_index(p / "refined.jsonl", p / "db", c)
            self.assertEqual(manifest["embedding_contract"], c.contract)
            self.assertEqual(len(retrieval.search(p / "db", "怎么煮面", c)), 1)
            self.assertTrue(c._encoder.calls[-1][0][0].startswith("Instruct:"))
            (model / "config.json").write_text('{"version":2}')
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                retrieval.search(p / "db", "怎么煮面", c)
            self.assertTrue(retrieval.search(p / "db", "煮面", lexical_only=True))

if __name__ == "__main__":
    unittest.main()

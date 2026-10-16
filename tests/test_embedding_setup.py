"""Configuration regression tests; never use real credentials or network."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streamrag.embedding_setup import (
    DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL, configure_key, resolve_api_key,
    embedding_status,
)


class EmbeddingSetupTests(unittest.TestCase):
    def test_defaults_and_absent_key(self):
        self.assertEqual("BAAI/bge-m3", DEFAULT_EMBED_MODEL)
        self.assertEqual("https://api.siliconflow.cn/v1", DEFAULT_EMBED_URL)
        with tempfile.TemporaryDirectory() as d:
            key, source = resolve_api_key(env={}, secret_file=Path(d) / "absent.env")
            self.assertEqual(("", "missing"), (key, source))
            status = embedding_status(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL,
                                      env={}, secret_file=Path(d) / "absent.env")
            self.assertFalse(status["key_configured"])
            self.assertFalse(status["ready_to_ingest"])

    def test_private_saved_key_read_without_echo(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / ".local-secrets" / "siliconflow-embedding.env"
            private_test_key = "sk-" + ("test" * 10)
            result = configure_key(private_test_key, secret_file=file)
            self.assertTrue(result["saved"])
            self.assertNotIn(private_test_key, repr(result))
            key, source = resolve_api_key(env={}, secret_file=file)
            self.assertEqual(private_test_key, key)
            self.assertEqual("local-secret-file", source)
            status = embedding_status(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL,
                                      env={}, secret_file=file)
            self.assertTrue(status["key_configured"])
            self.assertEqual("local-secret-file", status["key_source"])
            self.assertFalse(status["live_checked"])
            self.assertNotIn(private_test_key, repr(status))
            self.assertEqual(("other-key", "environment"),
                             resolve_api_key(env={"STREAM_RAG_EMBED_API_KEY": "other-key"},
                                             secret_file=file))

    def test_live_requires_key_or_mock(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "absent.env"
            with self.assertRaisesRegex(ValueError, "Missing Embedding API Key"):
                embedding_status(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL,
                                 live=True, env={}, secret_file=file)

        class MockClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def embed(self, value):
                return [0.1, 0.2, 0.3]
        status = embedding_status(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL,
                                  live=True, env={"STREAM_RAG_EMBED_API_KEY": "mock-secret"},
                                  client_class=MockClient)
        self.assertTrue(status["live_checked"])
        self.assertTrue(status["ready_to_ingest"])
        self.assertEqual(3, status["returned_dimensions"])
        self.assertNotIn("mock-secret", repr(status))

    def test_invalid_secret_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "secret.env"
            with self.assertRaises(ValueError):
                configure_key("paste-your-key-here", secret_file=file)
            self.assertFalse(file.exists())


if __name__ == "__main__":
    unittest.main()

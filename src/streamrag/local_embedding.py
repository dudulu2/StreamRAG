"""Offline Qwen3-Embedding-0.6B with separate query/document encoding."""
import hashlib
import json
import math
from pathlib import Path

from .embedding_setup import DEFAULT_EMBED_MODEL, DEFAULT_MODEL_PATH

QUERY_INSTRUCTION = (
    "Given a conversation context or question, retrieve relevant livestream "
    "passages with concrete stories, opinions, or conversational responses."
)


class LocalQwenEmbedding:
    mode = "local"
    model = DEFAULT_EMBED_MODEL

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="auto", batch_size=1):
        self.model_path = Path(model_path).expanduser().resolve()
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if not 1 <= batch_size <= 16:
            raise ValueError("Local embedding batch size must be 1..16")
        self.device, self.batch_size = device, batch_size
        self._encoder = None
        self._file_state = None
        self._snapshot_digest = None

    @property
    def contract(self):
        files = sorted(p for p in self.model_path.rglob("*") if p.is_file()
                       and ".cache" not in p.relative_to(self.model_path).parts
                       and p.suffix in (".json", ".safetensors", ".txt", ".model", ".jinja"))
        state = [(p.relative_to(self.model_path).as_posix(), p.stat().st_size,
                  p.stat().st_mtime_ns) for p in files]
        if state != self._file_state:
            fingerprint = hashlib.sha256()
            for path in files:
                fingerprint.update(path.relative_to(self.model_path).as_posix().encode())
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        fingerprint.update(block)
            self._snapshot_digest = fingerprint.hexdigest()
            self._file_state = state
        return {"version": 1, "model": self.model, "dimensions": 1024,
                "normalize": True, "metric": "cosine", "pooling": "last_token",
                "query_instruction": QUERY_INSTRUCTION,
                "snapshot_sha256": self._snapshot_digest}

    def _load(self):
        if self._encoder is not None:
            return self._encoder
        config_path = self.model_path / "config.json"
        if not config_path.is_file():
            raise ValueError("Local Qwen model missing. Run embedding-download first, "
                             "or set --embed-model-path to the downloaded model folder.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3" or config.get("hidden_size") != 1024:
            raise ValueError("Expected Qwen3-Embedding-0.6B (qwen3, hidden_size=1024)")
        pooling = self.model_path / "1_Pooling" / "config.json"
        if not pooling.is_file() or not json.loads(pooling.read_text(encoding="utf-8")).get("pooling_mode_lasttoken"):
            raise ValueError("Qwen embedding model requires last-token pooling configuration")
        if (self.model_path / "model.safetensors.aria2").exists():
            raise ValueError("Qwen weight download is incomplete; finish the download first")
        import torch
        from sentence_transformers import SentenceTransformer
        device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self._encoder = SentenceTransformer(
            str(self.model_path), device=device, local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
            tokenizer_kwargs={"padding_side": "left"})
        self._encoder.max_seq_length = min(self._encoder.max_seq_length, 32768)
        return self._encoder

    def _encode(self, texts, query=False):
        is_batch = isinstance(texts, list)
        values = texts if is_batch else [texts]
        if not values or any(not isinstance(t, str) or not t.strip() for t in values):
            raise ValueError("Empty embedding input")
        if query:
            values = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in values]
        encoder = self._load()
        lengths = encoder.tokenizer(values, truncation=False, padding=False)["input_ids"]
        if any(len(tokens) > encoder.max_seq_length for tokens in lengths):
            raise ValueError("Embedding input exceeds model token limit; split the window")
        vectors = encoder.encode(values, prompt="", batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True).tolist()
        if len(vectors) != len(values) or any(
            len(v) != 1024 or not all(math.isfinite(x) for x in v)
            or not 0.99 <= math.sqrt(sum(x*x for x in v)) <= 1.01 for v in vectors):
            raise ValueError("Invalid local Qwen vectors: expected unit 1024-dimensional vectors")
        return vectors if is_batch else vectors[0]

    def embed(self, texts):
        return self._encode(texts)

    def embed_query(self, text):
        return self._encode(text, query=True)

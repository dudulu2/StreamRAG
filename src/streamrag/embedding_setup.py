"""Local Qwen setup; inference never downloads or reads cloud credentials."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MODEL_WEIGHT_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
DEFAULT_MODEL_PATH = ROOT / "qwen"
DEFAULT_EMBED_URL = "http://127.0.0.1:1234/v1"


def resolve_api_key(key_env="STREAM_RAG_EMBED_API_KEY"):
    key = os.getenv(key_env, "").strip()
    return key, "environment" if key else "missing"


def download_model(model_path=DEFAULT_MODEL_PATH):
    import hashlib
    from huggingface_hub import snapshot_download
    # Explicit download operation only; prefer ordinary HTTPS on Windows.
    from huggingface_hub import constants
    constants.HF_HUB_DISABLE_XET = True
    path = snapshot_download(repo_id=DEFAULT_EMBED_MODEL, revision=MODEL_REVISION,
                             local_dir=str(model_path),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    weight = Path(path) / "model.safetensors"
    with weight.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != MODEL_WEIGHT_SHA256:
        raise ValueError("Downloaded Qwen weight checksum mismatch")
    # A prior manual aria2 transfer may have left its resume marker.
    (Path(path) / "model.safetensors.aria2").unlink(missing_ok=True)
    return {"downloaded": True, "model_path": path, "model": DEFAULT_EMBED_MODEL}


def embedding_status(client, *, live=False):
    result = {"embedding_model": client.model, "mode": client.mode,
              "live_checked": False, "ready_to_ingest": False}
    if client.mode == "local":
        result.update(model_path=str(client.model_path),
                      model_files_present=all((client.model_path / name).is_file() for name in
                                              ("config.json", "model.safetensors", "tokenizer.json", "modules.json"))
                      and not (client.model_path / "model.safetensors.aria2").exists(),
                      device=client.device, api_key_required=False)
    if live:
        vectors = client.embed(["今天工作失误让我很难过。"])
        query = client.embed_query("工作失误后心情不好")
        result.update(live_checked=True, ready_to_ingest=True,
                      returned_dimensions=len(query), document_dimensions=len(vectors[0]))
        if client.mode == "local":
            result["device"] = str(client._encoder.device)
    return result

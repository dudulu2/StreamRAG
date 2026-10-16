"""Independent local configuration for StreamRAG's embedding API.

Never import AI-Wife internals or print/commit API keys.
"""
from __future__ import annotations

import getpass
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRET_FILE = ROOT / ".local-secrets" / "siliconflow-embedding.env"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_EMBED_URL = "https://api.siliconflow.cn/v1"
DEFAULT_KEY_ENV = "STREAM_RAG_EMBED_API_KEY"


def _parse_key_file(path: Path) -> str:
    if not path.exists():
        return ""
    if not path.is_file() or path.is_symlink():
        raise ValueError("Embedding key path must be a regular local file")
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(
            r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", stripped
        )
        if not match:
            raise ValueError("Malformed embedding secret file")
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[:1] == value[-1:] and value[0] in ("'", '"'):
            value = value[1:-1]
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("Invalid embedding secret file value")
        values[key] = value
    return values.get(DEFAULT_KEY_ENV, "").strip()


def resolve_api_key(key_env=DEFAULT_KEY_ENV, *, env=None, secret_file=None):
    """Environment wins; project-local secret fallback only for the default key name."""
    environment = os.environ if env is None else env
    value = str(environment.get(key_env, "") or "").strip()
    if value:
        return value, "environment"
    if key_env != DEFAULT_KEY_ENV:
        return "", "missing"
    value = _parse_key_file(Path(secret_file or SECRET_FILE))
    return (value, "local-secret-file") if value else ("", "missing")


def configure_key(key=None, *, secret_file=None, prompt=None):
    """Persist key without echoing it or passing it through shell arguments."""
    if key is None:
        key = (prompt or getpass.getpass)("SiliconFlow Embedding API Key (hidden): ")
    value = str(key).strip()
    if (not 16 <= len(value) <= 512
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
            or value.lower() in ("paste-your-key-here", "your-api-key", "your-key")):
        raise ValueError("Invalid API Key format; nothing was saved")
    destination = Path(secret_file or SECRET_FILE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("Refusing to write through a symlinked secret directory")
    try:
        os.chmod(destination.parent, 0o700)
    except OSError:
        pass
    if destination.is_symlink():
        raise ValueError("Refusing to overwrite a symlinked secret")
    tmp = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("# Private StreamRAG key; never commit or share this file.\n")
            stream.write(f"{DEFAULT_KEY_ENV}={value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, destination)
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"saved": True, "location": "project-local-secret", "key_exposed": False}


def embedding_status(model, url, key_env=DEFAULT_KEY_ENV, *, live=False,
                     client_class=None, env=None, secret_file=None):
    """Report configuration without exposing key; live=True sends a single API request."""
    key, source = resolve_api_key(key_env, env=env, secret_file=secret_file)
    result = {
        "embedding_model": model,
        "embedding_url": url,
        "key_configured": bool(key),
        "key_source": source,
        "live_checked": False,
        "ready_for_live_test": bool(key),
        "ready_to_ingest": False,
    }
    if live:
        if not key:
            raise ValueError(
                "Missing Embedding API Key. Run embedding-configure first, "
                "or set STREAM_RAG_EMBED_API_KEY.")
        if client_class is None:
            from .retrieval import EmbeddingClient
            client_class = EmbeddingClient
        client = client_class(mode="api", model=model, base_url=url, api_key=key)
        vector = client.embed("今天心情不好时，有人认真听我说话。")
        result["live_checked"] = True
        result["ready_to_ingest"] = True
        result["returned_dimensions"] = len(vector)
        result["embedding_valid"] = bool(vector) and all(
            isinstance(v, (int, float)) for v in vector)
    return result

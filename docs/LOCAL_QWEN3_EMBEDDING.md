# Qwen3-Embedding-0.6B local deployment (Windows)

**No Embedding subscription or actual API key is needed.** Install Ollama and pull its local qwen3-embedding:0.6b model once (approximately 639 MB).

## Setup

Open Windows PowerShell inside the independent StreamRAG repository:

~~~powershell
cd D:\project\ai\StreamRAG
.\scripts\setup_local_qwen3.ps1
~~~

The setup script checks for Ollama, optionally installs it using winget, starts the local server at 127.0.0.1:11434, downloads qwen3-embedding:0.6b and verifies that both the native Ollama and OpenAI-compatible embedding endpoints return 1024-dimensional vectors. When uv is available, it also checks the existing StreamRAG Python EmbeddingClient.

Official model: https://ollama.com/library/qwen3-embedding:0.6b
Official API compatibility: https://docs.ollama.com/api/openai-compatibility

If PowerShell blocks the script, inspect the source before running it and adjust the execution policy only if you trust it. The installer can also be run manually using ollama pull qwen3-embedding:0.6b.

## Build and search

After preparing/refining transcripts with Gemini or Astra, use this local launcher:

~~~powershell
.\scripts\run_local_qwen3.ps1 ingest --windows data\refined\host01\windows.jsonl --refined data\refined\host01\refined_gemini4.jsonl --db data\lancedb\host01
.\scripts\run_local_qwen3.ps1 search --db data\lancedb\host01 --query "朋友工作受挫后如何回应？"
~~~

Replace the host01 paths with your actual files. The launcher sets three STREAM_RAG_EMBED_* values only in the launcher process. The string "ollama" is a dummy API key ignored by the local Ollama endpoint; it is not a secret. Your AI-Wife project is not changed.

Keyword-only search does not require Ollama once an index has been created. New semantic queries and newly indexed documents require the local Ollama server to be running, but no Internet connectivity is needed after the model has been downloaded.

You can switch Embedding models in the future, but you must rebuild the vectors from the retained refined JSONL. You do not need to rerun Gemini/Astra text refinement.

**Verification:** An installation script being available on GitHub does not indicate that it has run on your PC; actual installation and download must be verified on the Windows host.

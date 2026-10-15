#requires -Version 5.1
# Run StreamRAG using local Ollama embeddings; no cloud API key.
$ErrorActionPreference = "Stop"
if ($args.Count -lt 1 -or @("ingest", "search", "eval", "pipeline") -notcontains $args[0]) {
    throw "Usage: .\scripts\run_local_qwen3.ps1 {ingest|search|eval|pipeline} [StreamRAG CLI flags]"
}
$subcommand = $args[0]
$rest = @()
if ($args.Count -gt 1) { $rest = @($args[1..($args.Count-1)]) }
$env:STREAM_RAG_EMBED_MODEL = "qwen3-embedding:0.6b"
$env:STREAM_RAG_EMBED_URL = "http://127.0.0.1:11434/v1"
$env:STREAM_RAG_EMBED_API_KEY = "ollama"  # placeholder ignored by local Ollama
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $project
try {
    & uv run python scripts/stream_rag.py $subcommand @rest
    if ($LASTEXITCODE -ne 0) { throw "StreamRAG failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }

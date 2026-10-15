#requires -Version 5.1
[CmdletBinding()]
param([switch]$SkipInstall, [switch]$SkipPull)
$ErrorActionPreference = "Stop"
$Model = "qwen3-embedding:0.6b"
$Base = "http://127.0.0.1:11434"

function Find-Ollama {
    $cmd = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @()
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe")
    }
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    }
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
    }
    return $null
}
function Is-Running {
    try {
        $null = Invoke-RestMethod -Uri "$Base/api/tags" -TimeoutSec 3
        return $true
    }
    catch { return $false }
}

$ollama = Find-Ollama
if (-not $ollama) {
    if ($SkipInstall) { throw "Ollama not installed. See https://ollama.com/download/windows" }
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "winget unavailable. Install Ollama from https://ollama.com/download/windows"
    }
    Write-Host "Installing Ollama (Windows installer may request confirmation)..."
    & winget.exe install --exact --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "Ollama installation failed." }
    $ollama = Find-Ollama
    if (-not $ollama) {
        throw "Reopen PowerShell to refresh the executable PATH, then run this script again."
    }
}
Write-Host "Ollama found: $ollama"

if (-not (Is-Running)) {
    Write-Host "Starting Ollama locally..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 2
        if (Is-Running) { $ready = $true; break }
    }
    if (-not $ready) {
        throw "Ollama did not start on 127.0.0.1:11434. Open the Ollama app manually and retry."
    }
}

if (-not $SkipPull) {
    Write-Host "Downloading $Model (first download is approximately 639 MB)..."
    & $ollama pull $Model
    if ($LASTEXITCODE -ne 0) { throw "Model download failed. Check your connection." }
}

$payload = @{
    model = $Model
    input = @("I feel discouraged at work.", "My friend feels upset and needs support.")
} | ConvertTo-Json -Depth 5
$native = Invoke-RestMethod -Uri "$Base/api/embed" -Method Post -ContentType "application/json" -Body ([Text.Encoding]::UTF8.GetBytes($payload)) -TimeoutSec 180
if (@($native.embeddings).Count -ne 2) { throw "Native Ollama embedding count mismatch." }
$dim = @($native.embeddings[0]).Count
if ($dim -ne 1024 -or @($native.embeddings[1]).Count -ne $dim) {
    throw "Embedding dimensions mismatch: $dim, expected 1024."
}
Write-Host "Ollama native /api/embed test passed: $dim dimensions."

$compatiblePayload = @{ model = $Model; input = @("A friend needs support.") } | ConvertTo-Json -Depth 5
$compatible = Invoke-RestMethod -Uri "$Base/v1/embeddings" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer ollama" } -Body ([Text.Encoding]::UTF8.GetBytes($compatiblePayload)) -TimeoutSec 180
if (@($compatible.data).Count -ne 1 -or @($compatible.data[0].embedding).Count -ne 1024) {
    throw "OpenAI-compatible /v1/embeddings response invalid."
}
Write-Host "Ollama /v1/embeddings test passed."

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path (Join-Path $root "scripts\stream_rag.py"))) {
    throw "Cannot find StreamRAG CLI at the expected project path."
}
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Push-Location $root
    try {
        $py = 'import sys;sys.path.insert(0,"src");from streamrag.retrieval import EmbeddingClient;c=EmbeddingClient("api","qwen3-embedding:0.6b","http://127.0.0.1:11434/v1","ollama");v=c.embed("Someone needs empathy.");assert len(v)==1024;print("StreamRAG Python embedding test passed.")'
        & uv run python -c $py
        if ($LASTEXITCODE -ne 0) { throw "StreamRAG Python embedding client test failed." }
    }
    finally { Pop-Location }
}
else { Write-Warning "uv not found; StreamRAG Python client test skipped." }
Write-Host "Setup completed. No paid API key is needed."
Write-Host "Use scripts/run_local_qwen3.ps1 to build or search indexes."

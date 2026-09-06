$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found. Run 'uv sync --group dev' first."
}

& $python -m ruff check (Join-Path $projectRoot "app") (Join-Path $projectRoot "tests") --no-cache
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m mypy --cache-dir (Join-Path $projectRoot "data\mypy-cache")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCommand) {
    foreach ($scriptName in @("app.js", "markdown.js", "stream.js")) {
        & $nodeCommand.Source --check (Join-Path $projectRoot "app\history_agent\web\static\$scriptName")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $nodeCommand.Source (Join-Path $projectRoot "scripts\test-stream.mjs")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Node.js is not installed; frontend stream checks skipped."
}

Write-Host "All project checks passed."

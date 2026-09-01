$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$ruff = Join-Path $projectRoot ".venv\Scripts\ruff.exe"
$mypy = Join-Path $projectRoot ".venv\Scripts\mypy.exe"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found. Run 'uv sync --group dev' first."
}

& $ruff check (Join-Path $projectRoot "app") (Join-Path $projectRoot "tests") --no-cache
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $mypy --cache-dir (Join-Path $projectRoot "data\mypy-cache")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All project checks passed."

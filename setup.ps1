param([string]$PythonPath = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# Dependency caches and installation temp files stay beside this repository on D:.
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache/pip'
$env:TEMP = Join-Path $PSScriptRoot '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $PythonPath -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
$dependencyFile = 'requirements.txt'
if (Test-Path -LiteralPath 'requirements-lock.txt') { $dependencyFile = 'requirements-lock.txt' }
# Do not roll back the optional Transformers-compatible environment on repeated setup.
if (Test-Path -LiteralPath '.venv/Lib/site-packages/llmlingua') { $dependencyFile = 'requirements.txt' }
& '.venv/Scripts/python.exe' -m pip install --disable-pip-version-check -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Ready. Run: .\run.ps1 -Mode Test'


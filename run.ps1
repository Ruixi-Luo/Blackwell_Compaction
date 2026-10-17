param(
    [ValidateSet('Run','Prepare','Replay','Report','Test')][string]$Mode = 'Report',
    [string]$Config = 'configs/pilot.json'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache/pip'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'
$env:TEMP = Join-Path $PSScriptRoot '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run .\setup.ps1 first.' }
if ($Mode -eq 'Test') {
    & $pythonExe -m unittest discover -s tests -v
} else {
    $extra = @()
    if ($Mode -eq 'Prepare') { $extra += '--prepare-only' }
    if ($Mode -eq 'Replay') { $extra += '--offline' }
    if ($Mode -eq 'Report') { $extra += '--report-only' }
    & $pythonExe run_experiment.py --config $Config @extra
}
if ($LASTEXITCODE -ne 0) { throw 'Experiment command failed. Review the error above; no automatic model substitution occurred.' }

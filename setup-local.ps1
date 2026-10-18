param([ValidateSet('cpu','cuda')][string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache/pip'
$env:TEMP = Join-Path $PSScriptRoot '.cache/tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run .\setup.ps1 first.' }
$torchChannel = if ($Device -eq 'cuda') { 'cu128' } else { 'cpu' }
& $pythonExe -m pip install --disable-pip-version-check --upgrade "torch==2.8.0+$torchChannel" --index-url "https://download.pytorch.org/whl/$torchChannel"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
$dependencyFile = 'requirements-local.txt'
if (Test-Path -LiteralPath 'requirements-local-lock.txt') { $dependencyFile = 'requirements-local-lock.txt' }
& $pythonExe -m pip install --disable-pip-version-check -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Local dependencies installation failed.' }
# Explicit compatibility exception: retain upstream SC code with current spaCy.
& $pythonExe -m pip install --disable-pip-version-check --no-deps 'selective-context==0.1.4' 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
if ($LASTEXITCODE -ne 0) { throw 'Selective Context / English parser installation failed.' }
& $pythonExe tools/prepare_local_models.py
if ($LASTEXITCODE -ne 0) { throw 'Local model preparation failed.' }
Write-Output 'Local compressors ready. See docs/Local_Compressors.html for use and comparison.'

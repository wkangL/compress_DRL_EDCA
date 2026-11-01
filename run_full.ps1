$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python run.py --config configs/full.yaml
exit $LASTEXITCODE

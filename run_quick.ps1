$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python run.py --config configs/quick.yaml
exit $LASTEXITCODE

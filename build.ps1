param(
    [string]$Config = 'configs/tetrahedron_dvd.json',
    [string]$Output = 'output/tetrahedron_dvd',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -m cdholo build --config $Config --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Hologram generation failed; see output above.' }
} finally { Pop-Location }

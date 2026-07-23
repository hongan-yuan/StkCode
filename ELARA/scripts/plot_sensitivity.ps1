[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

& (Join-Path $PSScriptRoot '_run_python.ps1') `
    -m ELARA.plot_sensitivity @Arguments
exit $LASTEXITCODE

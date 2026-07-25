[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

& (Join-Path $PSScriptRoot '_run_python.ps1') `
    -m ELARA.plot_paper_figures `
    --temporal-bin-slots 5 `
    --temporal-smoothing-window 7 `
    @Arguments
exit $LASTEXITCODE

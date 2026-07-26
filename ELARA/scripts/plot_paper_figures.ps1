[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

& (Join-Path $PSScriptRoot '_run_python.ps1') `
    -m ELARA.plot_paper_figures `
    --baseline-root ELARA/outputs/baseline-tests2 `
    --bandit-root ELARA/outputs/baseline-tests2 `
    --sensitivity-root ELARA/outputs/sensitivity `
    --output-dir ELARA/paper_figs3 `
    --exclude-model-seeds 44 `
    --temporal-bin-slots 5 `
    --temporal-smoothing-window 7 `
    --minimum-seeds 3 `
    @Arguments
exit $LASTEXITCODE

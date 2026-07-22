[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

& (Join-Path $PSScriptRoot '_run_python.ps1') -m ELARA.parallel_runner train @Arguments
exit $LASTEXITCODE

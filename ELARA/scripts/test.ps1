[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

& (Join-Path $PSScriptRoot '_run_python.ps1') `
    -m unittest discover -s ELARA/tests -v @Arguments
exit $LASTEXITCODE

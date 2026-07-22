[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArguments
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))

function Test-PythonInterpreter {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Executable,

        [string[]] $PrefixArguments = @()
    )

    try {
        & $Executable @PrefixArguments -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

$pythonExecutable = $null
$pythonPrefixArguments = @()

if ($env:ELARA_PYTHON) {
    if (-not (Test-PythonInterpreter -Executable $env:ELARA_PYTHON)) {
        [Console]::Error.WriteLine(
            '[ELARA] ELARA_PYTHON is not a working Python 3.10+ executable: "{0}"',
            $env:ELARA_PYTHON
        )
        exit 1
    }

    $pythonExecutable = $env:ELARA_PYTHON
}
else {
    $localCandidates = @()

    if ($env:VIRTUAL_ENV) {
        $localCandidates += Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
    }
    if ($env:CONDA_PREFIX) {
        $localCandidates += Join-Path $env:CONDA_PREFIX 'python.exe'
    }

    $localCandidates += Join-Path $repoRoot '.venv-elara\Scripts\python.exe'
    $localCandidates += Join-Path $repoRoot '.venv\Scripts\python.exe'

    foreach ($candidate in $localCandidates) {
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
            (Test-PythonInterpreter -Executable $candidate)) {
            $pythonExecutable = $candidate
            break
        }
    }

    if (-not $pythonExecutable) {
        $pythonCommand = Get-Command 'python.exe' -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($pythonCommand -and (Test-PythonInterpreter -Executable $pythonCommand.Source)) {
            $pythonExecutable = $pythonCommand.Source
        }
    }

    if (-not $pythonExecutable) {
        $pyCommand = Get-Command 'py.exe' -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($pyCommand -and
            (Test-PythonInterpreter -Executable $pyCommand.Source -PrefixArguments @('-3'))) {
            $pythonExecutable = $pyCommand.Source
            $pythonPrefixArguments = @('-3')
        }
    }
}

if (-not $pythonExecutable) {
    [Console]::Error.WriteLine('[ELARA] Python 3.10 or newer was not found.')
    [Console]::Error.WriteLine(
        '[ELARA] Install Python, activate a virtual environment, or set ELARA_PYTHON.'
    )
    exit 9009
}

Push-Location -LiteralPath $repoRoot
try {
    & $pythonExecutable @pythonPrefixArguments @PythonArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

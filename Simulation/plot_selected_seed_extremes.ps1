$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not $env:PYTHON_BIN) {
    $env:PYTHON_BIN = "python"
}
if (-not $env:INPUT_DIR) {
    $env:INPUT_DIR = Join-Path $ScriptDir "test_outputs\ablation_experiments"
}
if (-not $env:OUTPUT_DIR) {
    $env:OUTPUT_DIR = Join-Path $env:INPUT_DIR "selected_seed_plots"
}
if (-not $env:TARGET_ABLATION) {
    $env:TARGET_ABLATION = "full"
}
if (-not $env:ABLATIONS) {
    $env:ABLATIONS = "full no_bandit shortest_hop_routing nearest_replica service_pressure sc_nfv fairness_nfv_greedy"
}
if (-not $env:FORMAT) {
    $env:FORMAT = "auto"
}

Write-Host "Plotting selected seed extremes"
Write-Host "  input_dir: $env:INPUT_DIR"
Write-Host "  output_dir: $env:OUTPUT_DIR"
Write-Host "  target_ablation: $env:TARGET_ABLATION"
Write-Host "  ablations: $env:ABLATIONS"
Write-Host "  format: $env:FORMAT"

Push-Location -LiteralPath $ProjectRoot
try {
    & $env:PYTHON_BIN -m Simulation.pics.plot_selected_seed_extremes `
        --input-dir $env:INPUT_DIR `
        --output-dir $env:OUTPUT_DIR `
        --target-ablation $env:TARGET_ABLATION `
        --ablations $env:ABLATIONS `
        --format $env:FORMAT `
        @args
}
finally {
    Pop-Location
}

Write-Host "Done. Selected-seed plots are under $env:OUTPUT_DIR."

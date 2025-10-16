param(
    [string]$ExePath = ".\x64\Debug\VestaEngine.exe",
    [string]$OutputDir = "out\portfolio",
    [double]$WarmupSeconds = 0.75,
    [double]$CaptureSeconds = 1.25
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Engine executable not found: $ExePath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$captureDir = Join-Path $OutputDir "captures"
New-Item -ItemType Directory -Force -Path $captureDir | Out-Null

$runs = @(
    @{
        Name = "01_raster_path_split"
        Args = @("--scene", "assets\basicmesh.glb", "--compare", "split", "--compare-split", "0.50")
    },
    @{
        Name = "02_raster_path_difference"
        Args = @("--scene", "assets\basicmesh.glb", "--compare", "difference", "--compare-scale", "6.0")
    },
    @{
        Name = "03_pathtrace_reference"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "pathtrace", "--pt-backend", "auto", "--pt-scale", "1.0")
    },
    @{
        Name = "04_gaussian_debug"
        Args = @("--scene", "assets\demo\garden_input.ply", "--mode", "gaussian")
    }
)

foreach ($run in $runs) {
    $csvPath = Join-Path $OutputDir ($run.Name + ".csv")
    $pngPath = Join-Path $captureDir ($run.Name + ".png")
    $args = @(
        "--benchmark", $csvPath,
        "--screenshot", $pngPath,
        "--warmup-seconds", "$WarmupSeconds",
        "--benchmark-seconds", "$CaptureSeconds",
        "--no-ui"
    ) + $run.Args

    Write-Host "Capturing $($run.Name)..."
    & $ExePath @args
    if ($LASTEXITCODE -ne 0) {
        throw "Capture failed for $($run.Name) with exit code $LASTEXITCODE"
    }
}

Write-Host "Portfolio captures written to $OutputDir"

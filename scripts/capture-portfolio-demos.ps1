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
        Args = @("--scene", "assets\demo\garden_input.ply", "--mode", "gaussian", "--gaussian-debug", "overdraw")
    },
    @{
        Name = "04b_gaussian_radius_debug"
        Args = @("--scene", "assets\demo\garden_input.ply", "--mode", "gaussian", "--gaussian-debug", "radius")
    },
    @{
        Name = "04c_gaussian_contribution_debug"
        Args = @("--scene", "assets\demo\garden_input.ply", "--mode", "gaussian", "--gaussian-debug", "contribution-count")
    },
    @{
        Name = "05_raster_normal_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "normal")
    },
    @{
        Name = "06_raster_ssao_lighting"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--ssao", "on", "--ssao-radius", "1.1", "--ssao-intensity", "1.75")
    },
    @{
        Name = "07_raster_ao_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "ao", "--ssao", "on", "--ssao-radius", "1.1", "--ssao-intensity", "1.75")
    },
    @{
        Name = "08_raster_taa"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--taa", "on", "--taa-feedback", "0.90")
    },
    @{
        Name = "09_raster_motion_vector_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "motion-vector", "--taa", "on", "--taa-feedback", "0.90")
    },
    @{
        Name = "10_raster_ssr_lighting"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--ssr", "on", "--ssr-distance", "24.0", "--ssr-thickness", "0.22", "--ssr-intensity", "0.85")
    },
    @{
        Name = "11_raster_ssgi_lighting"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--ssgi", "on", "--ssgi-radius", "1.8", "--ssgi-intensity", "0.45", "--ssgi-samples", "12")
    },
    @{
        Name = "11b_raster_direct_lighting_aov"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "direct")
    },
    @{
        Name = "11c_raster_indirect_lighting_aov"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "indirect", "--ssgi", "on", "--ssgi-radius", "1.8", "--ssgi-intensity", "0.45")
    },
    @{
        Name = "11d_raster_reflection_aov"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "reflection", "--ssr", "on", "--ssr-distance", "24.0", "--ssr-intensity", "0.85")
    },
    @{
        Name = "12_pathtrace_indirect_aov"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "pathtrace", "--pt-debug", "indirect")
    },
    @{
        Name = "13_pathtrace_integrator_controls"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "pathtrace", "--pt-backend", "auto", "--pt-nee", "on", "--pt-rr", "on", "--pt-rr-depth", "3", "--pt-firefly-clamp", "6.0")
    },
    @{
        Name = "14_pathtrace_ray_count_heatmap"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "pathtrace", "--pt-backend", "auto", "--pt-debug", "ray-count", "--pt-nee", "on", "--pt-rr", "on")
    },
    @{
        Name = "15_pathtrace_denoised_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "composite", "--debug-view", "denoised", "--pt-backend", "auto")
    },
    @{
        Name = "16_raster_path_difference_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "composite", "--debug-view", "difference-reference", "--compare-scale", "6.0", "--pt-backend", "auto")
    },
    @{
        Name = "17_raster_wireframe_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "wireframe")
    },
    @{
        Name = "18_raster_mip_level_debug"
        Args = @("--scene", "assets\demo\DamagedHelmet.glb", "--mode", "raster", "--debug-view", "mip-level")
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

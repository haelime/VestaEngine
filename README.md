# VestaEngine

Vulkan/C++ portfolio renderer focused on modern real-time rendering structure: render graph orchestration, raster + deferred + gaussian + path tracing, hardware RT fallback, scene streaming, and engine-style runtime tuning.

## Features

- Multi-pass frame flow: `Geometry -> Deferred Lighting -> Gaussian Splat -> Path Tracing -> Composite`
- Hybrid path tracing backend: `Auto / Compute / Hardware RT`
- glTF/GLB scene loading with async parse/prepare and streaming GPU upload
- Surface-level frustum culling, optional distance culling, optional indirect draw path
- Texture upload path for base-color textures
- ImGui main menu, engine tuning controls, recent scenes, and benchmark overlay
- OS-native scene picker on Windows

## Build

### CMake

```powershell
cmake -S . -B build
cmake --build build --target engine --config Debug
```

### Visual Studio

Open [VestaEngine.sln](VestaEngine.sln) or [VestaEngine.vcxproj](VestaEngine.vcxproj) and build `Debug|x64`.

## Run

### Default

```powershell
.\x64\Debug\VestaEngine.exe
```

### Load a specific scene

```powershell
.\x64\Debug\VestaEngine.exe --scene assets\structure.glb --preset balanced --mode composite
```

### Automated benchmark

```powershell
.\x64\Debug\VestaEngine.exe --scene assets\structure.glb --benchmark out\benchmark.csv --warmup-seconds 2 --benchmark-seconds 10 --preset quality --mode composite --pt-backend auto --no-ui
```

The benchmark writes one CSV row with timing, scene stats, upload timings, and active PT backend.

## Controls

- `RMB + Mouse`: look
- `W/A/S/D/Q/E`: move
- `Shift`: faster move
- `1/2/3/4`: Composite / Deferred / Gaussian / Path Trace
- `G`: toggle Gaussian
- `P`: toggle Path Tracing
- `F1`: toggle debug overlay

## Runtime Options

- `File`: open scene, open recent, reload
- `View`: switch render modes
- `Options -> Engine Tuning`:
  - scene upload mode
  - frustum/distance culling
  - indirect draw
  - texture streaming
  - RT build on load
  - per-frame geometry/texture upload budgets

## Current Scope

- Scene formats: `glTF / GLB`
- Texture support: base-color textures
- Path tracing material model is intentionally lightweight and not yet texture-driven
- Benchmark mode currently captures a fixed camera view rather than a scripted camera path

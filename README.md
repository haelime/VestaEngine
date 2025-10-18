# VestaEngine

VestaEngine is a Vulkan-based rendering engine focused on real-time rendering, path tracing, ray tracing integration, Gaussian splatting, and engine-level debugging tools.

The project is built as a compact graphics engine rather than a game framework. Its main goal is to expose the full rendering pipeline, GPU resource state, profiling data, and debug visualization controls directly inside the runtime.

## Core Rendering

VestaEngine currently supports multiple rendering paths:

- Deferred rasterization with PBR-style material inputs
- Progressive path tracing
- Hardware ray tracing when the device supports Vulkan ray tracing extensions
- Gaussian splatting for point-based captured scenes
- Hybrid composite rendering that combines raster, path traced, and Gaussian outputs
- Rasterizer versus path tracer comparison views, including split-screen and difference heatmap modes

The renderer is organized around explicit render passes and a render graph. Each frame records pass order, input and output resources, resource formats, resolution, barriers, synchronization transitions, CPU time, and GPU timestamp timing.

## Runtime Debug UI

The engine includes an ImGui-based debug interface for controlling and inspecting the renderer at runtime.

Major panels include:

- Frame and engine overview
- Render pass and render graph inspector
- GPU profiler with pass timing and work counts
- Debug visualization controls
- Scene, camera, light, environment, and material controls
- Resource inspector for textures, buffers, frame textures, acceleration structures, and Gaussian buffers
- Log, validation, performance warning, and error console

The UI is designed to make the renderer inspectable while it is running. It exposes frame time, GPU timing, render mode, debug views, pass enable state, pass order, resource usage, barrier transitions, shader reload, screenshot capture, and path tracing accumulation reset.

## Render Graph and Profiling

The render graph tracks how each frame is built from individual passes. For every active pass, the engine records:

- CPU execution time
- GPU timestamp time
- Input and output resources
- Resource format and resolution
- Barrier count and resource usage transitions
- Estimated draw, dispatch, triangle, ray, and splat work

Benchmark captures write CSV files with frame statistics, scene statistics, pass GPU timings, Gaussian counters, path tracing state, camera state, environment state, VSync state, present mode, and comparison mode.

## Debug Visualization

The renderer exposes common graphics debugging views, including:

- Final color
- Albedo
- Normal
- World position
- Linear depth
- UV
- Material ID
- Object ID
- Roughness
- Metallic
- Emissive
- Path tracing AOVs
- Gaussian alpha, revealage, overdraw, depth, and tile occupancy

These views are available through the runtime UI and are routed through the same render graph resources used by the frame.

## Scene and Resource System

The scene system supports mesh scenes and Gaussian splat scenes. It tracks CPU-side scene data, GPU buffers, resident textures, ray tracing acceleration structures, upload timing, and scene loading status.

The resource inspector reports:

- Texture size, format, residency, bindless index, memory estimate, and preview
- Vertex, index, material, triangle, emissive triangle, and Gaussian buffers
- TLAS and BLAS residency, build time, memory, instance count, primitive count, and ray tracing feature support
- Current-frame transient render graph textures

## Capture and Benchmarking

The engine can capture screenshots and benchmark runs from the command line. Captures produce PNG images with JSON metadata sidecars. Benchmarks produce CSV files with frame timing and renderer state.

This makes rendering results reproducible and keeps performance data tied to the exact scene, render mode, backend, camera, environment, and debug state used for the run.

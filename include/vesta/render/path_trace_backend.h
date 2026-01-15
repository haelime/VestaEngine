#pragma once

#include <cstdint>

namespace vesta::render {
// Auto keeps the UI simple: prefer HW RT when available, otherwise fall back
// to the compute implementation without changing the pass interface.
enum class PathTraceBackend : uint32_t {
    Auto = 0,
    Compute = 1,
    HardwareRT = 2,
};

enum class PathTraceDebugView : uint32_t {
    Final = 0,
    Albedo = 1,
    Normal = 2,
    Depth = 3,
    Direct = 4,
    Indirect = 5,
    RayCountHeatmap = 6,
    DiffuseBounce = 7,
    SpecularBounce = 8,
    Throughput = 9,
    Pdf = 10,
};
} // namespace vesta::render

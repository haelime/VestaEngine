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
} // namespace vesta::render

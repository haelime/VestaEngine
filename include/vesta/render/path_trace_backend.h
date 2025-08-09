#pragma once

#include <cstdint>

namespace vesta::render {
enum class PathTraceBackend : uint32_t {
    Auto = 0,
    Compute = 1,
    HardwareRT = 2,
};
} // namespace vesta::render

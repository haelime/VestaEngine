#pragma once

#include <cstdint>
#include <limits>

namespace vesta::render {
inline constexpr uint32_t kInvalidResourceIndex = std::numeric_limits<uint32_t>::max();

struct BufferHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const BufferHandle&) const = default;
};

struct ImageHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const ImageHandle&) const = default;
};

struct SamplerHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const SamplerHandle&) const = default;
};

struct AccelerationStructureHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const AccelerationStructureHandle&) const = default;
};
} // namespace vesta::render

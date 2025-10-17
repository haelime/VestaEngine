#pragma once

#include <cstdint>
#include <limits>

namespace vesta::render {
inline constexpr uint32_t kInvalidResourceIndex = std::numeric_limits<uint32_t>::max();

// RHI 내부 배열 슬롯을 가리키는 얇은 버퍼 핸들이다.
// 외부 코드는 VkBuffer 자체 대신 이 핸들을 통해 자원을 참조한다.
struct BufferHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const BufferHandle&) const = default;
};

// RHI 내부 배열 슬롯을 가리키는 이미지 핸들이다.
// swapchain image와 일반 GPU image를 같은 방식으로 다루기 위해 쓴다.
struct ImageHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const ImageHandle&) const = default;
};

// 샘플러 핸들을 위한 자리다.
// 아직 구현은 비어 있지만 리소스 참조 방식을 handle 기반으로 맞추기 위해 미리 둔다.
struct SamplerHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const SamplerHandle&) const = default;
};
} // namespace vesta::render

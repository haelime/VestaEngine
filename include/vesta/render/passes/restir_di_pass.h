#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>
#include <vesta/render/resources/resource_handles.h>

namespace vesta::render {
// Lightweight ReSTIR DI candidate pass. It does not shade the final image yet;
// it writes deterministic current/history reservoir records for profiler,
// resource-inspector, and future resolve integration.
class RestirDiPass final : public IRenderPass {
public:
    void SetReservoirBuffers(BufferHandle current, BufferHandle history);
    void SetControls(uint32_t frameIndex,
        uint32_t width,
        uint32_t height,
        uint32_t candidateLightCount,
        uint32_t reservoirCount,
        uint32_t spatialSamples,
        uint32_t activeLightCount,
        uint32_t localLightCount,
        uint32_t emissiveTriangleCount,
        bool temporalReuse,
        bool spatialReuse);

    [[nodiscard]] bool IsBackendAvailable() const { return _backendAvailable; }

    [[nodiscard]] std::string_view Name() const override { return "ReSTIR DI CandidatePass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    BufferHandle _currentReservoir{};
    BufferHandle _historyReservoir{};
    uint32_t _frameIndex{ 0 };
    uint32_t _width{ 1 };
    uint32_t _height{ 1 };
    uint32_t _candidateLightCount{ 1 };
    uint32_t _reservoirCount{ 1 };
    uint32_t _spatialSamples{ 0 };
    uint32_t _activeLightCount{ 1 };
    uint32_t _localLightCount{ 1 };
    uint32_t _emissiveTriangleCount{ 0 };
    bool _temporalReuse{ true };
    bool _spatialReuse{ true };
    bool _backendAvailable{ false };

    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

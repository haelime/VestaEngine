#pragma once

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>
#include <vesta/render/resources/resource_handles.h>

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
// Ray-query DDGI probe update slice. It writes simple irradiance and visibility
// moment records into the existing DDGI storage buffers so the graph/profiler
// exposes a real probe update pass while the deferred composite remains stable.
class DdgiProbeUpdatePass final : public IRenderPass {
public:
    void SetProbeBuffers(BufferHandle irradiance, BufferHandle visibility);
    void SetScene(const vesta::scene::Scene* scene);
    void SetFrameSlot(uint32_t frameSlot);
    void SetFrameIndex(uint32_t frameIndex);
    void SetControls(uint32_t probeCountX,
        uint32_t probeCountY,
        uint32_t probeCountZ,
        uint32_t raysPerProbe,
        float probeSpacing,
        float hysteresis,
        glm::vec4 lightDirectionAndIntensity,
        glm::vec4 directionalLightColor,
        glm::vec4 environmentParams);

    [[nodiscard]] bool IsBackendAvailable() const { return _backendAvailable; }

    [[nodiscard]] std::string_view Name() const override { return "DDGI Probe UpdatePass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    BufferHandle _irradianceBuffer{};
    BufferHandle _visibilityBuffer{};
    const vesta::scene::Scene* _scene{ nullptr };
    uint32_t _frameSlot{ 0 };
    uint32_t _frameIndex{ 0 };
    uint32_t _probeCountX{ 8 };
    uint32_t _probeCountY{ 4 };
    uint32_t _probeCountZ{ 8 };
    uint32_t _raysPerProbe{ 128 };
    float _probeSpacing{ 2.0f };
    float _hysteresis{ 0.95f };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::vec4 _directionalLightColor{ 1.0f };
    glm::vec4 _environmentParams{ 1.0f, 0.0f, 0.0f, 0.22f };
    bool _backendAvailable{ false };

    VkDescriptorPool _descriptorPool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _descriptorSetLayout{ VK_NULL_HANDLE };
    std::array<VkDescriptorSet, 2> _descriptorSets{};
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

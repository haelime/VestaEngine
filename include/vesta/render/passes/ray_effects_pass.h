#pragma once

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
// Optional ray-query pass for hybrid raster + RT effects. It writes a compact
// visibility buffer consumed by deferred lighting:
// R = directional shadow visibility, G = ambient occlusion visibility,
// B = reflection miss visibility for masking IBL/SSR reflections.
class RayEffectsPass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle normal, GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetFrameSlot(uint32_t frameSlot);
    void SetFrameIndex(uint32_t frameIndex);
    void SetLight(glm::vec4 lightDirectionAndIntensity);
    void SetControls(bool shadowsEnabled,
        bool ambientOcclusionEnabled,
        bool reflectionsEnabled,
        bool globalIlluminationEnabled,
        uint32_t shadowSamples,
        uint32_t aoSamples,
        uint32_t reflectionSamples,
        uint32_t giSamples,
        float maxRayDistance,
        float aoRadius,
        float reflectionRoughnessCutoff);
    [[nodiscard]] bool IsBackendAvailable() const { return _backendAvailable; }

    [[nodiscard]] std::string_view Name() const override { return "RayEffectsPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _normal{};
    GraphTextureHandle _depth{};
    GraphTextureHandle _output{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    uint32_t _frameSlot{ 0 };
    uint32_t _frameIndex{ 0 };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    bool _shadowsEnabled{ false };
    bool _ambientOcclusionEnabled{ false };
    bool _reflectionsEnabled{ false };
    bool _globalIlluminationEnabled{ false };
    uint32_t _shadowSamples{ 1 };
    uint32_t _aoSamples{ 1 };
    uint32_t _reflectionSamples{ 1 };
    uint32_t _giSamples{ 1 };
    float _maxRayDistance{ 100.0f };
    float _aoRadius{ 2.0f };
    float _reflectionRoughnessCutoff{ 0.8f };
    bool _backendAvailable{ false };

    VkDescriptorPool _descriptorPool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _descriptorSetLayout{ VK_NULL_HANDLE };
    std::array<VkDescriptorSet, 2> _descriptorSets{};
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::render {
// Resolves the current ReSTIR DI reservoir buffer into a screen-space direct
// lighting image that deferred lighting can composite in the same frame.
class RestirDiResolvePass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);
    void SetReservoirBuffer(BufferHandle reservoir);
    void SetCamera(const Camera* camera);
    void SetLight(glm::vec4 lightDirectionAndIntensity);
    void SetLightColors(glm::vec4 directional, glm::vec4 point, glm::vec4 spot, glm::vec4 area);
    void SetPointLight(bool enabled, glm::vec4 positionAndIntensity);
    void SetSpotLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 directionAndAngle);
    void SetAreaLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 normalAndSize);
    void SetControls(uint32_t frameIndex,
        uint32_t reservoirCount,
        uint32_t candidateLightCount,
        uint32_t activeLightCount,
        uint32_t localLightCount,
        uint32_t emissiveTriangleCount,
        float intensity,
        bool showReservoirs,
        bool showSelectedLight);
    [[nodiscard]] bool IsBackendAvailable() const { return _backendAvailable; }

    [[nodiscard]] std::string_view Name() const override { return "ReSTIR DI ResolvePass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _albedo{};
    GraphTextureHandle _normal{};
    GraphTextureHandle _material{};
    GraphTextureHandle _depth{};
    GraphTextureHandle _output{};
    BufferHandle _reservoir{};
    BufferHandle _resolveConstantsBuffer{};
    const Camera* _camera{ nullptr };
    uint32_t _frameIndex{ 0 };
    uint32_t _reservoirCount{ 1 };
    uint32_t _candidateLightCount{ 1 };
    uint32_t _activeLightCount{ 1 };
    uint32_t _localLightCount{ 1 };
    uint32_t _emissiveTriangleCount{ 0 };
    float _intensity{ 0.18f };
    bool _showReservoirs{ false };
    bool _showSelectedLight{ false };
    bool _pointLightEnabled{ false };
    bool _spotLightEnabled{ false };
    bool _areaLightEnabled{ false };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::vec4 _directionalLightColor{ 1.0f, 1.0f, 1.0f, 0.0f };
    glm::vec4 _pointLightPositionAndIntensity{ 0.0f, 2.0f, 0.0f, 0.0f };
    glm::vec4 _pointLightColor{ 1.0f, 0.82f, 0.55f, 0.0f };
    glm::vec4 _spotLightPositionAndIntensity{ 0.0f, 3.0f, 2.5f, 0.0f };
    glm::vec4 _spotLightDirectionAndAngle{ 0.0f, -0.8f, -0.6f, 28.0f };
    glm::vec4 _spotLightColor{ 1.0f, 0.88f, 0.68f, 0.0f };
    glm::vec4 _areaLightPositionAndIntensity{ 0.0f, 3.2f, 0.0f, 0.0f };
    glm::vec4 _areaLightNormalAndSize{ 0.0f, -1.0f, 0.0f, 2.0f };
    glm::vec4 _areaLightColor{ 0.86f, 0.92f, 1.0f, 0.0f };
    bool _backendAvailable{ false };

    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

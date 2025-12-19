#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::render {
// Reads the GBuffer and writes lit scene color into a storage image.
// Because lighting happens after geometry, one mesh pass can feed many lighting models.
class DeferredLightingPass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);
    void SetDebugOutput(GraphTextureHandle output, uint32_t debugView);
    void SetCamera(const Camera* camera);
    void SetLight(glm::vec4 lightDirectionAndIntensity);
    void SetPointLight(bool enabled, glm::vec4 positionAndIntensity);
    void SetSpotLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 directionAndAngle);
    void SetAreaLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 normalAndSize);
    void SetEnvironment(glm::vec4 environmentParams);
    void SetAmbientOcclusion(bool enabled, float radius, float intensity);
    void SetScreenSpaceReflections(bool enabled, float maxDistance, float thickness, float intensity);
    void SetScreenSpaceGlobalIllumination(bool enabled, float radius, float intensity, uint32_t sampleCount);
    void SetShadowMap(GraphTextureHandle shadowMap, glm::mat4 lightViewProjection, float bias, float normalBias, float strength);

    [[nodiscard]] std::string_view Name() const override { return "DeferredLightingPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _albedo{};
    GraphTextureHandle _normal{};
    GraphTextureHandle _material{};
    GraphTextureHandle _depth{};
    GraphTextureHandle _shadowMap{};
    GraphTextureHandle _output{};
    GraphTextureHandle _debugOutput{};
    uint32_t _debugView{ 0 };
    const Camera* _camera{ nullptr };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::vec4 _pointLightPositionAndIntensity{ 0.0f, 2.0f, 0.0f, 0.0f };
    glm::vec4 _spotLightPositionAndIntensity{ 0.0f, 3.0f, 2.5f, 0.0f };
    glm::vec4 _spotLightDirectionAndAngle{ 0.0f, -0.8f, -0.6f, 28.0f };
    glm::vec4 _areaLightPositionAndIntensity{ 0.0f, 3.2f, 0.0f, 0.0f };
    glm::vec4 _areaLightNormalAndSize{ 0.0f, -1.0f, 0.0f, 2.0f };
    glm::vec4 _environmentParams{ 1.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 _ssaoParams{ 1.0f, 0.75f, 1.35f, 0.0f };
    glm::vec4 _ssrParams{ 1.0f, 18.0f, 0.18f, 0.65f };
    glm::vec4 _ssgiParams{ 1.0f, 1.4f, 0.32f, 10.0f };
    glm::mat4 _lightViewProjection{ 1.0f };
    glm::vec4 _shadowParams{ 0.0015f, 0.015f, 0.82f, 0.0f };
    BufferHandle _lightingConstantsBuffer{};
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>
#include <vesta/render/passes/shadow_map_pass.h>

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
    void SetLightColors(glm::vec4 directional, glm::vec4 point, glm::vec4 spot, glm::vec4 area);
    void SetPointLight(bool enabled, glm::vec4 positionAndIntensity);
    void SetSpotLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 directionAndAngle);
    void SetAreaLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 normalAndSize);
    void SetEnvironment(glm::vec4 environmentParams);
    void SetEnvironmentImage(uint32_t sampledImageIndex);
    void SetEnvironmentCubeImage(uint32_t sampledCubeImageIndex);
    void SetIblDiffuseIrradianceImage(uint32_t sampledImageIndex);
    void SetIblBrdfLutImage(uint32_t sampledImageIndex);
    void SetIblSpecularPrefilterImage(uint32_t sampledImageIndex);
    void SetEnvironmentSpecularStrength(float strength);
    void SetRayEffects(GraphTextureHandle rayEffects,
        GraphTextureHandle rayReflection,
        GraphTextureHandle rayGlobalIllumination,
        bool shadowsEnabled,
        bool ambientOcclusionEnabled,
        bool reflectionsEnabled,
        bool denoiserEnabled,
        bool temporalEnabled);
    void SetRestirDiResolve(GraphTextureHandle restirDirectLighting, bool enabled);
    void SetAmbientOcclusion(bool enabled, float radius, float intensity);
    void SetScreenSpaceReflections(bool enabled, float maxDistance, float thickness, float intensity);
    void SetScreenSpaceGlobalIllumination(bool enabled, float radius, float intensity, uint32_t sampleCount);
    void SetDdgi(bool enabled,
        uint32_t probeCountX,
        uint32_t probeCountY,
        uint32_t probeCountZ,
        float spacing,
        float hysteresis,
        float intensity,
        BufferHandle irradianceBuffer,
        BufferHandle visibilityBuffer);
    void SetContactShadows(bool enabled, float length, float intensity);
    void SetShadowMap(GraphTextureHandle shadowMap,
        const std::array<DirectionalShadowCascade, 4>& cascades,
        uint32_t cascadeCount,
        float splitLambda,
        float bias,
        float normalBias,
        float strength,
        bool pcssEnabled,
        float filterRadius);

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
    GraphTextureHandle _rayEffects{};
    GraphTextureHandle _rayReflection{};
    GraphTextureHandle _rayGlobalIllumination{};
    GraphTextureHandle _restirDirectLighting{};
    GraphTextureHandle _output{};
    GraphTextureHandle _debugOutput{};
    uint32_t _debugView{ 0 };
    const Camera* _camera{ nullptr };
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
    glm::vec4 _environmentParams{ 1.0f, 0.0f, 0.0f, 0.0f };
    uint32_t _environmentImageIndex{ kInvalidResourceIndex };
    uint32_t _environmentCubeImageIndex{ kInvalidResourceIndex };
    uint32_t _iblDiffuseIrradianceImageIndex{ kInvalidResourceIndex };
    uint32_t _iblBrdfLutImageIndex{ kInvalidResourceIndex };
    uint32_t _iblSpecularPrefilterImageIndex{ kInvalidResourceIndex };
    float _environmentSpecularStrength{ 0.45f };
    glm::uvec4 _rayEffectsFlags{ 0u, 0u, 0u, 0u };
    glm::uvec2 _rayGiFlags{ 0u, 0u };
    bool _restirDirectLightingEnabled{ false };
    glm::vec4 _ssaoParams{ 1.0f, 0.75f, 1.35f, 0.0f };
    glm::vec4 _ssrParams{ 1.0f, 18.0f, 0.18f, 0.65f };
    glm::vec4 _ssgiParams{ 1.0f, 1.4f, 0.32f, 10.0f };
    glm::uvec4 _ddgiGrid{ 8u, 4u, 8u, 0u };
    glm::vec4 _ddgiParams{ 2.0f, 0.95f, 0.28f, 0.0f };
    BufferHandle _ddgiIrradianceBuffer{};
    BufferHandle _ddgiVisibilityBuffer{};
    glm::vec4 _contactShadowParams{ 1.0f, 1.2f, 0.35f, 0.0f };
    std::array<DirectionalShadowCascade, 4> _shadowCascades{};
    uint32_t _shadowCascadeCount{ 1 };
    float _shadowCascadeLambda{ 0.65f };
    glm::vec4 _shadowParams{ 0.0015f, 0.015f, 0.82f, 0.0f };
    glm::vec4 _shadowFilterParams{ 1.0f, 0.0f, 0.0f, 0.0f };
    BufferHandle _lightingConstantsBuffer{};
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

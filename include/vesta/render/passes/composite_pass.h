#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
enum class BloomPassStage : uint32_t {
    Extract = 0,
    Downsample = 1,
    Upsample = 2,
};

class BloomPass final : public IRenderPass {
public:
    explicit BloomPass(BloomPassStage stage);

    void SetInput(GraphTextureHandle input);
    void SetSecondaryInput(GraphTextureHandle input);
    void SetOutput(GraphTextureHandle output);
    void SetParameters(float threshold, float intensity);
    [[nodiscard]] BloomPassStage Stage() const { return _stage; }

    [[nodiscard]] std::string_view Name() const override;
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    BloomPassStage _stage{ BloomPassStage::Extract };
    GraphTextureHandle _input{};
    GraphTextureHandle _secondaryInput{};
    GraphTextureHandle _output{};
    float _threshold{ 1.0f };
    float _intensity{ 0.1f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};

// Final full-screen pass. It chooses which intermediate image to show, or blends
// several of them together for the portfolio "composite" presentation mode.
class CompositePass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle deferredLighting,
        GraphTextureHandle pathTrace,
        GraphTextureHandle gaussianAccum,
        GraphTextureHandle gaussianReveal,
        GraphTextureHandle gaussianDebug);
    void SetGBufferInputs(GraphTextureHandle albedo,
        GraphTextureHandle normalRoughness,
        GraphTextureHandle material,
        GraphTextureHandle debug,
        GraphTextureHandle motion,
        GraphTextureHandle lightingDebug,
        GraphTextureHandle depth);
    void SetGaussianDebugResources(uint32_t tileRangeBufferIndex, uint32_t tileCountX, uint32_t tileCountY);
    void SetShadowMap(GraphTextureHandle shadowMap);
    void SetOverdraw(GraphTextureHandle overdraw);
    void SetBloomInput(GraphTextureHandle bloom);
    void SetOutput(GraphTextureHandle output);
    void SetMode(uint32_t mode, float gaussianMix, uint32_t debugView, uint32_t gaussianDebugView);
    void SetCompare(uint32_t compareMode, float splitPosition, float differenceScale);
    void SetExposure(float exposureEv);
    void SetToneMapping(uint32_t toneMappingMode);
    void SetPostProcess(float saturation,
        float contrast,
        bool vignetteEnabled,
        float vignetteStrength,
        bool bloomEnabled,
        float bloomThreshold,
        float bloomIntensity,
        bool fxaaEnabled,
        bool motionBlurEnabled,
        float motionBlurStrength);
    void SetAmbientOcclusion(bool enabled, bool screenSpaceEnabled, float radius, float intensity);
    void SetShadowCascadeDebug(uint32_t cascadeCount, float splitLambda, bool overlayEnabled);
    void SetCameraMatrices(const glm::mat4& viewProjection, const glm::mat4& inverseViewProjection);
    void SetDepthRange(float nearPlane, float farPlane);

    [[nodiscard]] std::string_view Name() const override { return "CompositePass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _deferredLighting{};
    GraphTextureHandle _pathTrace{};
    GraphTextureHandle _gaussianAccum{};
    GraphTextureHandle _gaussianReveal{};
    GraphTextureHandle _gaussianDebug{};
    GraphTextureHandle _gbufferAlbedo{};
    GraphTextureHandle _gbufferNormalRoughness{};
    GraphTextureHandle _gbufferMaterial{};
    GraphTextureHandle _gbufferDebug{};
    GraphTextureHandle _gbufferMotion{};
    GraphTextureHandle _lightingDebug{};
    GraphTextureHandle _gbufferDepth{};
    GraphTextureHandle _shadowMap{};
    GraphTextureHandle _overdraw{};
    GraphTextureHandle _bloom{};
    GraphTextureHandle _output{};
    uint32_t _mode{ 0 };
    uint32_t _debugView{ 0 };
    uint32_t _gaussianDebugView{ 0 };
    uint32_t _compareMode{ 0 };
    uint32_t _gaussianTileRangeBufferIndex{ kInvalidResourceIndex };
    uint32_t _gaussianTileCountX{ 0 };
    uint32_t _gaussianTileCountY{ 0 };
    float _gaussianMix{ 0.25f };
    float _compareSplitPosition{ 0.5f };
    float _compareDifferenceScale{ 4.0f };
    float _exposureEv{ 0.0f };
    uint32_t _toneMappingMode{ 2 };
    float _saturation{ 1.0f };
    float _contrast{ 1.0f };
    bool _vignetteEnabled{ false };
    float _vignetteStrength{ 0.0f };
    bool _bloomEnabled{ false };
    float _bloomThreshold{ 1.0f };
    float _bloomIntensity{ 0.1f };
    bool _fxaaEnabled{ false };
    bool _motionBlurEnabled{ false };
    float _motionBlurStrength{ 0.35f };
    glm::vec4 _ssaoParams{ 1.0f, 0.75f, 1.35f, 0.0f };
    glm::vec4 _shadowCascadeParams{ 4.0f, 0.92f, 0.0f, 0.0f };
    glm::mat4 _viewProjection{ 1.0f };
    glm::mat4 _inverseViewProjection{ 1.0f };
    glm::mat4 _previousViewProjection{ 1.0f };
    bool _hasPreviousViewProjection{ false };
    float _nearPlane{ 0.05f };
    float _farPlane{ 500.0f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

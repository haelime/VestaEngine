#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
// Final full-screen pass. It chooses which intermediate image to show, or blends
// several of them together for the portfolio "composite" presentation mode.
class CompositePass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle deferredLighting,
        GraphTextureHandle pathTrace,
        GraphTextureHandle gaussianAccum,
        GraphTextureHandle gaussianReveal,
        GraphTextureHandle gaussianDebug);
    void SetGBufferInputs(GraphTextureHandle albedo, GraphTextureHandle normalRoughness, GraphTextureHandle material);
    void SetGaussianDebugResources(uint32_t tileRangeBufferIndex, uint32_t tileCountX, uint32_t tileCountY);
    void SetOutput(GraphTextureHandle output);
    void SetMode(uint32_t mode, float gaussianMix, uint32_t debugView, uint32_t gaussianDebugView);

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
    GraphTextureHandle _output{};
    uint32_t _mode{ 0 };
    uint32_t _debugView{ 0 };
    uint32_t _gaussianDebugView{ 0 };
    uint32_t _gaussianTileRangeBufferIndex{ kInvalidResourceIndex };
    uint32_t _gaussianTileCountX{ 0 };
    uint32_t _gaussianTileCountY{ 0 };
    float _gaussianMix{ 0.25f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

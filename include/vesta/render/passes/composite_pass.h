#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
// Final full-screen pass. It chooses which intermediate image to show, or blends
// several of them together for the portfolio "composite" presentation mode.
class CompositePass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle deferredLighting, GraphTextureHandle pathTrace, GraphTextureHandle gaussian);
    void SetOutput(GraphTextureHandle output);
    void SetMode(uint32_t mode, float gaussianMix);

    [[nodiscard]] std::string_view Name() const override { return "CompositePass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _deferredLighting{};
    GraphTextureHandle _pathTrace{};
    GraphTextureHandle _gaussian{};
    GraphTextureHandle _output{};
    uint32_t _mode{ 0 };
    float _gaussianMix{ 0.25f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

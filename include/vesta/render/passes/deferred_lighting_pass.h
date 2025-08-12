#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
// Reads the GBuffer and writes lit scene color into a storage image.
// Because lighting happens after geometry, one mesh pass can feed many lighting models.
class DeferredLightingPass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal);
    void SetOutput(GraphTextureHandle output);

    [[nodiscard]] std::string_view Name() const override { return "DeferredLightingPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _albedo{};
    GraphTextureHandle _normal{};
    GraphTextureHandle _output{};
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

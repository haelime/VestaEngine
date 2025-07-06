#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
class GaussianSplatPass final : public IRenderPass {
public:
    void SetDepthInput(GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);

    [[nodiscard]] std::string_view Name() const override { return "GaussianSplatPass"; }
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;

private:
    GraphTextureHandle _depthInput{};
    GraphTextureHandle _output{};
};
} // namespace vesta::render

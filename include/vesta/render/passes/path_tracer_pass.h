#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
class PathTracerPass final : public IRenderPass {
public:
    void SetOutput(GraphTextureHandle output);
    void SetDepthInput(GraphTextureHandle depth);

    [[nodiscard]] std::string_view Name() const override { return "PathTracerPass"; }
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;

private:
    GraphTextureHandle _output{};
    GraphTextureHandle _depthInput{};
};
} // namespace vesta::render

#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
class CompositePass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle deferredLighting, GraphTextureHandle pathTrace, GraphTextureHandle gaussian);
    void SetOutput(GraphTextureHandle output);

    [[nodiscard]] std::string_view Name() const override { return "CompositePass"; }
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;

private:
    GraphTextureHandle _deferredLighting{};
    GraphTextureHandle _pathTrace{};
    GraphTextureHandle _gaussian{};
    GraphTextureHandle _output{};
};
} // namespace vesta::render

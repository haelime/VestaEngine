#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
class DeferredRasterPass final : public IRenderPass {
public:
    void SetGBufferTargets(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle depth);
    void SetLightingTarget(GraphTextureHandle lighting);

    [[nodiscard]] std::string_view Name() const override { return "DeferredRasterPass"; }
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;

private:
    GraphTextureHandle _gbufferAlbedo{};
    GraphTextureHandle _gbufferNormal{};
    GraphTextureHandle _sceneDepth{};
    GraphTextureHandle _lightingTarget{};
};
} // namespace vesta::render

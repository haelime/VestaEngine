#pragma once

#include <string_view>

namespace vesta::render {
class RenderGraphBuilder;
class RenderGraphContext;
class RenderDevice;

// Common interface shared by all render passes.
class IRenderPass {
public:
    virtual ~IRenderPass() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;
    virtual void Initialize(RenderDevice& device) {}
    virtual void Setup(RenderGraphBuilder& builder) = 0;
    virtual void Execute(const RenderGraphContext& context) = 0;
    virtual void Shutdown(RenderDevice& device) {}
};
} // namespace vesta::render

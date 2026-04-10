#pragma once

#include <string_view>

namespace vesta::render {
class RenderGraphBuilder;
class RenderGraphContext;
class RenderDevice;

// Every render pass follows the same lifecycle:
// 1. Initialize GPU objects that survive across frames.
// 2. Describe graph reads/writes in Setup().
// 3. Record commands in Execute() after the graph inserted barriers.
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

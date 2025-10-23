#pragma once

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
class PathDenoisePass final : public IRenderPass {
public:
    void SetInput(GraphTextureHandle input);
    void SetGuides(GraphTextureHandle normalGuide, GraphTextureHandle depthGuide);
    void SetOutput(GraphTextureHandle output);
    void SetStrength(float strength);
    void SetTemporalBlend(float blend);
    void SetIterations(uint32_t iterations);
    void SetFrameIndex(uint32_t frameIndex);

    [[nodiscard]] std::string_view Name() const override { return "PathDenoisePass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _input{};
    GraphTextureHandle _normalGuide{};
    GraphTextureHandle _depthGuide{};
    GraphTextureHandle _output{};
    float _strength{ 0.65f };
    float _temporalBlend{ 0.88f };
    uint32_t _iterations{ 3 };
    uint32_t _frameIndex{ 0 };
    ImageHandle _historyImage{};
    VkExtent3D _historyExtent{};
    bool _historyInitialized{ false };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };

    void EnsureHistoryImage(RenderDevice& device, VkExtent3D extent);
    void DestroyHistoryImage(RenderDevice& device);
};
} // namespace vesta::render

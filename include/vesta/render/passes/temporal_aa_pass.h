#pragma once

#include <cstdint>

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

namespace vesta::render {
enum class RendererDebugView : uint32_t;

class TemporalAAPass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle input, GraphTextureHandle normalRoughness, GraphTextureHandle motion, GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);
    void SetEnabled(bool enabled);
    void SetFeedback(float feedback);
    void SetFrameIndex(uint32_t frameIndex);
    void SetCameraMatrices(const glm::mat4& viewProjection, const glm::mat4& inverseViewProjection);
    void SetDebugView(RendererDebugView debugView);

    [[nodiscard]] std::string_view Name() const override { return "TemporalAAPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _input{};
    GraphTextureHandle _normalRoughness{};
    GraphTextureHandle _motion{};
    GraphTextureHandle _depth{};
    GraphTextureHandle _output{};
    bool _enabled{ true };
    float _feedback{ 0.88f };
    uint32_t _frameIndex{ 0 };
    RendererDebugView _debugView{};
    glm::mat4 _viewProjection{ 1.0f };
    glm::mat4 _inverseViewProjection{ 1.0f };
    glm::mat4 _previousViewProjection{ 1.0f };
    bool _hasPreviousViewProjection{ false };
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

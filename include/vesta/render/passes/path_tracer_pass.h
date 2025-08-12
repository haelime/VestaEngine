#pragma once

#include <vesta/render/path_trace_backend.h>
#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
// Progressive path tracing pass with two backends:
// - compute shader fallback for portability
// - hardware ray tracing pipeline when the GPU supports it
class PathTracerPass final : public IRenderPass {
public:
    void SetOutput(GraphTextureHandle output);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetFrameIndex(uint32_t frameIndex);
    void SetFrameSlot(uint32_t frameSlot);
    void SetEnabled(bool enabled);
    void SetBackendPreference(PathTraceBackend backend);
    [[nodiscard]] PathTraceBackend GetActiveBackend() const { return _activeBackend; }

    [[nodiscard]] std::string_view Name() const override { return "PathTracerPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _output{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    // _frameIndex controls accumulation. When the camera moves, the renderer
    // resets it so history from the old view is not blended into the new one.
    uint32_t _frameIndex{ 0 };
    uint32_t _frameSlot{ 0 };
    bool _enabled{ true };
    PathTraceBackend _backendPreference{ PathTraceBackend::Auto };
    PathTraceBackend _activeBackend{ PathTraceBackend::Compute };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
    VkDescriptorPool _rtDescriptorPool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _rtDescriptorSetLayout{ VK_NULL_HANDLE };
    std::array<VkDescriptorSet, 2> _rtDescriptorSets{};
    VkPipelineLayout _rtPipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _rtPipeline{ VK_NULL_HANDLE };
    VkShaderModule _raygenShader{ VK_NULL_HANDLE };
    VkShaderModule _missShader{ VK_NULL_HANDLE };
    VkShaderModule _closestHitShader{ VK_NULL_HANDLE };
    BufferHandle _shaderBindingTable{};
    VkStridedDeviceAddressRegionKHR _raygenSbt{};
    VkStridedDeviceAddressRegionKHR _missSbt{};
    VkStridedDeviceAddressRegionKHR _hitSbt{};
    VkStridedDeviceAddressRegionKHR _callableSbt{};
};
} // namespace vesta::render

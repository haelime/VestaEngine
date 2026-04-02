#pragma once

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
// Draws points from the loaded scene as soft splats. This is intentionally kept
// separate from the mesh raster pass so it can be blended or disabled at runtime.
class GaussianSplatPass final : public IRenderPass {
public:
    void SetDepthInput(GraphTextureHandle depth);
    void SetOutputs(GraphTextureHandle accum, GraphTextureHandle reveal);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetParams(float opacity,
        bool enabled,
        uint32_t shDegree,
        bool viewDependentColor,
        bool antialiasing,
        bool fastCulling);

    [[nodiscard]] std::string_view Name() const override { return "GaussianSplatPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    static constexpr uint32_t kGaussianTileSize = 8;
    static constexpr uint32_t kGaussianMaxEntriesPerTile = 4096;

    void EnsureComputeResources(RenderDevice& device, VkExtent2D extent, uint32_t gaussianCount);
    void DestroyComputeResources(RenderDevice& device);
    void ExecuteComputePath(const RenderGraphContext& context);
    void ExecuteGraphicsPath(const RenderGraphContext& context);

    GraphTextureHandle _depthInput{};
    GraphTextureHandle _accumOutput{};
    GraphTextureHandle _revealOutput{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    float _opacity{ 1.0f };
    bool _enabled{ true };
    uint32_t _shDegree{ 0 };
    bool _viewDependentColor{ true };
    bool _antialiasing{ true };
    bool _fastCulling{ true };
    VkPipelineLayout _graphicsPipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _graphicsPipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
    VkPipelineLayout _computePipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _binPipeline{ VK_NULL_HANDLE };
    VkPipeline _tilePipeline{ VK_NULL_HANDLE };
    VkShaderModule _binShader{ VK_NULL_HANDLE };
    VkShaderModule _tileShader{ VK_NULL_HANDLE };
    VkDescriptorPool _computeDescriptorPool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _computeDescriptorSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSet _computeDescriptorSet{ VK_NULL_HANDLE };
    BufferHandle _projectedGaussianBuffer{};
    BufferHandle _tileCountBuffer{};
    BufferHandle _tileEntryBuffer{};
    uint32_t _cachedGaussianCount{ 0 };
    VkExtent2D _cachedExtent{};
    uint32_t _cachedTileCount{ 0 };
};
} // namespace vesta::render

#pragma once

#include <array>
#include <vector>

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
}
namespace vesta::core {
class JobSystem;
}

namespace vesta::render {
class OfficialGaussianRasterPass final : public IRenderPass {
public:
    void SetDepthInput(GraphTextureHandle depth);
    void SetOutputs(GraphTextureHandle accum, GraphTextureHandle reveal);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetJobSystem(vesta::core::JobSystem* jobs);
    void SetParams(float opacity, uint32_t shDegree, bool viewDependentColor, bool antialiasing, bool fastCulling);

    [[nodiscard]] std::string_view Name() const override { return "OfficialGaussianRasterPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    static constexpr uint32_t kTileSize = 8;

    struct ProjectedGaussianCPU {
        glm::vec4 centerRadiusDepth{ 0.0f };
        glm::vec4 conicOpacity{ 0.0f };
        glm::vec4 color{ 0.0f };
        glm::uvec4 tileRect{ 0u };
        glm::uvec4 tileOffset{ 0u };
    };

    struct Statistics {
        uint32_t projectedCount{ 0 };
        uint32_t duplicateCount{ 0 };
        uint32_t paddedDuplicateCount{ 0 };
        uint32_t tileCount{ 0 };
        float averageTilesTouched{ 0.0f };
        uint64_t rebuildCount{ 0 };
        float preprocessMs{ 0.0f };
        float scanMs{ 0.0f };
        float duplicateMs{ 0.0f };
        float sortMs{ 0.0f };
        float rangeMs{ 0.0f };
        float rasterMs{ 0.0f };
        float totalBuildMs{ 0.0f };
    };

public:
    [[nodiscard]] const Statistics& GetStatistics() const { return _statistics; }

private:
    void EnsureResources(RenderDevice& device, VkExtent2D extent, size_t projectedCount, size_t duplicateCount, size_t duplicateCapacity, size_t tileCount);
    void DestroyResources(RenderDevice& device);
    void RebuildFrameDataIfNeeded(VkExtent2D extent);
    bool NeedsFrameDataRebuild(VkExtent2D extent) const;
    [[nodiscard]] bool ReadTimestampResults(RenderDevice& device, uint32_t slot);

    GraphTextureHandle _depthInput{};
    GraphTextureHandle _accumOutput{};
    GraphTextureHandle _revealOutput{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    float _opacity{ 1.0f };
    uint32_t _shDegree{ 0 };
    bool _viewDependentColor{ true };
    bool _antialiasing{ true };
    bool _fastCulling{ true };

    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _preprocessPipeline{ VK_NULL_HANDLE };
    VkPipeline _duplicatePipeline{ VK_NULL_HANDLE };
    VkPipeline _scanPipeline{ VK_NULL_HANDLE };
    VkPipeline _sortPipeline{ VK_NULL_HANDLE };
    VkPipeline _rangePipeline{ VK_NULL_HANDLE };
    VkPipeline _rasterPipeline{ VK_NULL_HANDLE };
    VkShaderModule _preprocessShader{ VK_NULL_HANDLE };
    VkShaderModule _duplicateShader{ VK_NULL_HANDLE };
    VkShaderModule _scanShader{ VK_NULL_HANDLE };
    VkShaderModule _sortShader{ VK_NULL_HANDLE };
    VkShaderModule _rangeShader{ VK_NULL_HANDLE };
    VkShaderModule _rasterShader{ VK_NULL_HANDLE };
    VkDescriptorPool _descriptorPool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _descriptorSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSet _descriptorSet{ VK_NULL_HANDLE };
    static constexpr uint32_t kTimestampFrameSlots = 2;
    std::array<VkQueryPool, kTimestampFrameSlots> _timestampQueryPools{};
    float _timestampPeriodNs{ 0.0f };
    bool _timestampsSupported{ false };
    std::array<bool, kTimestampFrameSlots> _timestampResultsPending{};
    std::array<bool, kTimestampFrameSlots> _timestampResultsIncludeBuild{};
    uint32_t _timestampWriteSlot{ 0 };

    BufferHandle _projectedBuffer{};
    BufferHandle _duplicateKeyBuffer{};
    BufferHandle _duplicateValueBuffer{};
    BufferHandle _scanBlockSumBuffer{};
    BufferHandle _duplicateCountBuffer{};
    BufferHandle _duplicateScratchKeyBuffer{};
    BufferHandle _duplicateScratchValueBuffer{};
    BufferHandle _radixHistogramBuffer{};
    BufferHandle _radixBinBaseBuffer{};
    BufferHandle _tileRangeBuffer{};
    size_t _projectedCapacity{ 0 };
    size_t _duplicateCapacity{ 0 };
    size_t _scanBlockCapacity{ 0 };
    size_t _radixBlockCapacity{ 0 };
    size_t _tileCapacity{ 0 };
    size_t _duplicateCount{ 0 };
    size_t _duplicatePaddedCount{ 0 };

    const vesta::scene::Scene* _cachedScene{ nullptr };
    uint64_t _cachedSceneVersion{ 0 };
    VkExtent2D _cachedExtent{};
    glm::mat4 _cachedViewMatrix{ 1.0f };
    glm::mat4 _cachedViewProjection{ 1.0f };
    uint32_t _cachedShDegree{ 0 };
    bool _cachedViewDependentColor{ true };
    bool _cachedAntialiasing{ true };
    bool _cachedFastCulling{ true };
    float _cachedOpacity{ 1.0f };
    bool _gpuBuildDirty{ true };
    Statistics _statistics{};
    vesta::core::JobSystem* _jobs{ nullptr };
};
} // namespace vesta::render

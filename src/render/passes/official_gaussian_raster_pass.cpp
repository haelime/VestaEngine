#include <vesta/render/passes/official_gaussian_raster_pass.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
constexpr uint32_t kInvalidImageIndex = kInvalidResourceIndex;
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kShC1 = 0.4886025119029199f;
constexpr float kShC2_0 = 1.0925484305920792f;
constexpr float kShC2_1 = -1.0925484305920792f;
constexpr float kShC2_2 = 0.31539156525252005f;
constexpr float kShC2_3 = -1.0925484305920792f;
constexpr float kShC2_4 = 0.5462742152960396f;
constexpr float kShC3_0 = -0.5900435899266435f;
constexpr float kShC3_1 = 2.890611442640554f;
constexpr float kShC3_2 = -0.4570457994644658f;
constexpr float kShC3_3 = 0.3731763325901154f;
constexpr float kShC3_4 = -0.4570457994644658f;
constexpr float kShC3_5 = 1.445305721320277f;
constexpr float kShC3_6 = -0.5900435899266435f;

struct GaussianComputePushConstants {
    glm::uvec4 params0{ 0u };
    glm::uvec4 params1{ 0u };
    glm::vec4 params2{ 0.0f };
    glm::mat4 viewMatrix{ 1.0f };
    glm::mat4 viewProjection{ 1.0f };
    glm::vec4 cameraPositionAndSceneType{ 0.0f };
};

struct ProjectedGaussianGPU {
    glm::vec4 centerRadiusDepth{ 0.0f };
    glm::vec4 conicOpacity{ 0.0f };
    glm::vec4 color{ 0.0f };
    glm::uvec4 tileRect{ 0u };
    glm::uvec4 tileOffset{ 0u };
};

uint32_t NextPowerOfTwo(uint32_t value)
{
    if (value <= 1u) {
        return 1u;
    }
    value -= 1u;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

bool MatApproximatelyEqual(const glm::mat4& lhs, const glm::mat4& rhs)
{
    for (int column = 0; column < 4; ++column) {
        const glm::vec4 delta = glm::abs(lhs[column] - rhs[column]);
        if (delta.x >= 1.0e-5f || delta.y >= 1.0e-5f || delta.z >= 1.0e-5f || delta.w >= 1.0e-5f) {
            return false;
        }
    }
    return true;
}

glm::vec3 QuatRotate(const glm::vec4& quaternion, const glm::vec3& vector)
{
    const glm::vec3 t = 2.0f * glm::cross(glm::vec3(quaternion), vector);
    return vector + quaternion.w * t + glm::cross(glm::vec3(quaternion), t);
}

glm::vec3 EvaluateShColor(const vesta::scene::GaussianPrimitive& gaussian, uint32_t degree, bool viewDependentColor, const glm::vec3& viewDir)
{
    glm::vec3 result = kShC0 * glm::vec3(gaussian.shCoefficients[0]);
    if (!viewDependentColor || degree == 0u) {
        return glm::max(result + 0.5f, glm::vec3(0.0f));
    }
    const float x = viewDir.x;
    const float y = viewDir.y;
    const float z = viewDir.z;
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    result += (-kShC1 * y) * glm::vec3(gaussian.shCoefficients[1]);
    result += (kShC1 * z) * glm::vec3(gaussian.shCoefficients[2]);
    result += (-kShC1 * x) * glm::vec3(gaussian.shCoefficients[3]);
    if (degree >= 2u) {
        result += (kShC2_0 * x * y) * glm::vec3(gaussian.shCoefficients[4]);
        result += (kShC2_1 * y * z) * glm::vec3(gaussian.shCoefficients[5]);
        result += (kShC2_2 * (2.0f * zz - xx - yy)) * glm::vec3(gaussian.shCoefficients[6]);
        result += (kShC2_3 * x * z) * glm::vec3(gaussian.shCoefficients[7]);
        result += (kShC2_4 * (xx - yy)) * glm::vec3(gaussian.shCoefficients[8]);
    }
    if (degree >= 3u) {
        result += (kShC3_0 * y * (3.0f * xx - yy)) * glm::vec3(gaussian.shCoefficients[9]);
        result += (kShC3_1 * x * y * z) * glm::vec3(gaussian.shCoefficients[10]);
        result += (kShC3_2 * y * (4.0f * zz - xx - yy)) * glm::vec3(gaussian.shCoefficients[11]);
        result += (kShC3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy)) * glm::vec3(gaussian.shCoefficients[12]);
        result += (kShC3_4 * x * (4.0f * zz - xx - yy)) * glm::vec3(gaussian.shCoefficients[13]);
        result += (kShC3_5 * z * (xx - yy)) * glm::vec3(gaussian.shCoefficients[14]);
        result += (kShC3_6 * x * (xx - 3.0f * yy)) * glm::vec3(gaussian.shCoefficients[15]);
    }
    return glm::max(result + 0.5f, glm::vec3(0.0f));
}

void ClearStorageOutput(const RenderGraphContext& context, GraphTextureHandle accumOutput, GraphTextureHandle revealOutput)
{
    VkClearColorValue accumClear{};
    VkClearColorValue revealClear{};
    revealClear.float32[0] = 1.0f;
    revealClear.float32[1] = 1.0f;
    revealClear.float32[2] = 1.0f;
    revealClear.float32[3] = 1.0f;
    const VkImageSubresourceRange range = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(context.GetCommandBuffer(), context.GetDevice().GetImage(context.GetTextureHandle(accumOutput)), VK_IMAGE_LAYOUT_GENERAL, &accumClear, 1, &range);
    vkCmdClearColorImage(context.GetCommandBuffer(), context.GetDevice().GetImage(context.GetTextureHandle(revealOutput)), VK_IMAGE_LAYOUT_GENERAL, &revealClear, 1, &range);
}

void InsertMemoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
{
    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}
} // namespace

void OfficialGaussianRasterPass::SetDepthInput(GraphTextureHandle depth) { _depthInput = depth; }
void OfficialGaussianRasterPass::SetOutputs(GraphTextureHandle accum, GraphTextureHandle reveal)
{
    _accumOutput = accum;
    _revealOutput = reveal;
}
void OfficialGaussianRasterPass::SetScene(const vesta::scene::Scene* scene) { _scene = scene; }
void OfficialGaussianRasterPass::SetCamera(const Camera* camera) { _camera = camera; }
void OfficialGaussianRasterPass::SetJobSystem(vesta::core::JobSystem* jobs) { _jobs = jobs; }
void OfficialGaussianRasterPass::SetParams(
    float opacity, float alphaCutoff, uint32_t shDegree, bool viewDependentColor, bool antialiasing, bool fastCulling)
{
    _opacity = opacity;
    _alphaCutoff = alphaCutoff;
    _shDegree = shDegree;
    _viewDependentColor = viewDependentColor;
    _antialiasing = antialiasing;
    _fastCulling = fastCulling;
}

void OfficialGaussianRasterPass::Initialize(RenderDevice& device)
{
    if (_rasterPipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }
    VkDevice vkDevice = device.GetDevice();
    _preprocessShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/official_gaussian_preprocess.comp.spv"));
    _duplicateShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/official_gaussian_duplicate.comp.spv"));
    _scanShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/official_gaussian_scan.comp.spv"));
    _sortShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/official_gaussian_sort.comp.spv"));
    _rangeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/official_gaussian_ranges.comp.spv"));
    _rasterShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/official_gaussian_raster.comp.spv"));

    std::array<VkDescriptorPoolSize, 2> poolSizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &_descriptorPool));

    std::array<VkDescriptorSetLayoutBinding, 12> bindings{
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 10),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 11),
    };
    std::array<VkDescriptorBindingFlags, 12> bindingFlags{
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo = vkinit::descriptorset_layout_create_info(bindings.data(), static_cast<uint32_t>(bindings.size()));
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    VK_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &_descriptorSetLayout));

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(vkDevice, &allocInfo, &_descriptorSet));

    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{ device.GetBindless().GetLayout(), _descriptorSetLayout };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(GaussianComputePushConstants) },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);
    _preprocessPipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{ .layout = _pipelineLayout, .computeShader = _preprocessShader });
    _duplicatePipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{ .layout = _pipelineLayout, .computeShader = _duplicateShader });
    _scanPipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{ .layout = _pipelineLayout, .computeShader = _scanShader });
    _sortPipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{ .layout = _pipelineLayout, .computeShader = _sortShader });
    _rangePipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{ .layout = _pipelineLayout, .computeShader = _rangeShader });
    _rasterPipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{ .layout = _pipelineLayout, .computeShader = _rasterShader });
}

void OfficialGaussianRasterPass::Setup(RenderGraphBuilder& builder)
{
    if (_depthInput) {
        builder.Read(_depthInput, ResourceUsage::DepthRead);
    }
    builder.Write(_accumOutput, ResourceUsage::StorageWrite);
    builder.Write(_revealOutput, ResourceUsage::StorageWrite);
}

bool OfficialGaussianRasterPass::NeedsFrameDataRebuild(VkExtent2D extent) const
{
    if (_scene == nullptr || _camera == nullptr) {
        return false;
    }
    if (_scene != _cachedScene || _scene->GetContentVersion() != _cachedSceneVersion || extent.width != _cachedExtent.width
        || extent.height != _cachedExtent.height || _shDegree != _cachedShDegree
        || _viewDependentColor != _cachedViewDependentColor || _antialiasing != _cachedAntialiasing
        || _fastCulling != _cachedFastCulling || std::abs(_opacity - _cachedOpacity) >= 1.0e-4f
        || std::abs(_alphaCutoff - _cachedAlphaCutoff) >= 1.0e-6f) {
        return true;
    }
    return !MatApproximatelyEqual(_camera->GetViewMatrix(), _cachedViewMatrix)
        || !MatApproximatelyEqual(_camera->GetViewProjection(), _cachedViewProjection);
}

void OfficialGaussianRasterPass::RebuildFrameDataIfNeeded(VkExtent2D extent)
{
    const uint32_t tileCountX = std::max(1u, (extent.width + kTileSize - 1u) / kTileSize);
    const uint32_t tileCountY = std::max(1u, (extent.height + kTileSize - 1u) / kTileSize);
    _statistics.projectedCount = _scene->GetGaussianCount();
    _statistics.tileCount = tileCountX * tileCountY;
    ++_statistics.rebuildCount;
    _cachedScene = _scene;
    _cachedSceneVersion = _scene->GetContentVersion();
    _cachedExtent = extent;
    _cachedViewMatrix = _camera->GetViewMatrix();
    _cachedViewProjection = _camera->GetViewProjection();
    _cachedShDegree = _shDegree;
    _cachedViewDependentColor = _viewDependentColor;
    _cachedAntialiasing = _antialiasing;
    _cachedFastCulling = _fastCulling;
    _cachedOpacity = _opacity;
    _cachedAlphaCutoff = _alphaCutoff;
    _gpuBuildDirty = true;
}

void OfficialGaussianRasterPass::EnsureResources(
    RenderDevice& device, VkExtent2D extent, size_t projectedCount, size_t duplicateCount, size_t duplicateCapacity, size_t tileCount)
{
    const bool needProjected = !_projectedBuffer || projectedCount > _projectedCapacity;
    const bool needDuplicates = !_duplicateKeyBuffer || duplicateCapacity > _duplicateCapacity;
    const size_t scanBlockCount = std::max<size_t>((projectedCount + 255u) / 256u, 1u);
    const bool needScan = !_scanBlockSumBuffer || scanBlockCount > _scanBlockCapacity;
    const size_t radixBlockCount = std::max<size_t>((duplicateCount + 255u) / 256u, 1u);
    const bool needRadix = !_radixHistogramBuffer || radixBlockCount > _radixBlockCapacity;
    const bool needTiles = !_tileRangeBuffer || tileCount > _tileCapacity;
    if (!needProjected && !needDuplicates && !needScan && !needRadix && !needTiles) {
        return;
    }

    DestroyResources(device);

    if (projectedCount > 0) {
        _projectedBuffer = device.CreateBuffer(BufferDesc{
            .size = sizeof(ProjectedGaussianGPU) * projectedCount,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .debugName = "OfficialGaussianProjected",
        });
        _projectedCapacity = projectedCount;
    }
    if (duplicateCapacity > 0) {
        _duplicateKeyBuffer = device.CreateBuffer(BufferDesc{
            .size = sizeof(uint32_t) * duplicateCapacity,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .debugName = "OfficialGaussianDuplicateKeys",
        });
        _duplicateValueBuffer = device.CreateBuffer(BufferDesc{
            .size = sizeof(uint32_t) * duplicateCapacity,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .debugName = "OfficialGaussianDuplicateValues",
        });
        _duplicateScratchKeyBuffer = device.CreateBuffer(BufferDesc{
            .size = sizeof(uint32_t) * duplicateCapacity,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .debugName = "OfficialGaussianDuplicateScratchKeys",
        });
        _duplicateScratchValueBuffer = device.CreateBuffer(BufferDesc{
            .size = sizeof(uint32_t) * duplicateCapacity,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .debugName = "OfficialGaussianDuplicateScratchValues",
        });
        _duplicateCapacity = duplicateCapacity;
    }
    _scanBlockSumBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(uint32_t) * scanBlockCount,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "OfficialGaussianScanBlockSums",
    });
    _scanBlockCapacity = scanBlockCount;
    _duplicateCountBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(uint32_t) * 16u,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "OfficialGaussianCounts",
    });
    _radixHistogramBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(uint32_t) * radixBlockCount * 16u,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "OfficialGaussianRadixHistogram",
    });
    _radixBinBaseBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(uint32_t) * 16u,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "OfficialGaussianRadixBinBase",
    });
    _radixBlockCapacity = radixBlockCount;
    if (tileCount > 0) {
        _tileRangeBuffer = device.CreateBuffer(BufferDesc{
            .size = sizeof(glm::uvec2) * tileCount,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .debugName = "OfficialGaussianTileRanges",
        });
        _tileCapacity = tileCount;
    }

    _duplicateCount = duplicateCount;
    _duplicatePaddedCount = duplicateCapacity;
    _cachedExtent = extent;
    _gpuBuildDirty = true;
}

void OfficialGaussianRasterPass::DestroyResources(RenderDevice& device)
{
    if (_projectedBuffer) {
        device.DestroyBuffer(_projectedBuffer);
        _projectedBuffer = {};
    }
    if (_duplicateKeyBuffer) {
        device.DestroyBuffer(_duplicateKeyBuffer);
        _duplicateKeyBuffer = {};
    }
    if (_duplicateValueBuffer) {
        device.DestroyBuffer(_duplicateValueBuffer);
        _duplicateValueBuffer = {};
    }
    if (_duplicateScratchKeyBuffer) {
        device.DestroyBuffer(_duplicateScratchKeyBuffer);
        _duplicateScratchKeyBuffer = {};
    }
    if (_duplicateScratchValueBuffer) {
        device.DestroyBuffer(_duplicateScratchValueBuffer);
        _duplicateScratchValueBuffer = {};
    }
    if (_scanBlockSumBuffer) {
        device.DestroyBuffer(_scanBlockSumBuffer);
        _scanBlockSumBuffer = {};
    }
    if (_duplicateCountBuffer) {
        device.DestroyBuffer(_duplicateCountBuffer);
        _duplicateCountBuffer = {};
    }
    if (_radixHistogramBuffer) {
        device.DestroyBuffer(_radixHistogramBuffer);
        _radixHistogramBuffer = {};
    }
    if (_radixBinBaseBuffer) {
        device.DestroyBuffer(_radixBinBaseBuffer);
        _radixBinBaseBuffer = {};
    }
    if (_tileRangeBuffer) {
        device.DestroyBuffer(_tileRangeBuffer);
        _tileRangeBuffer = {};
    }
    _projectedCapacity = 0;
    _duplicateCapacity = 0;
    _scanBlockCapacity = 0;
    _radixBlockCapacity = 0;
    _tileCapacity = 0;
    _duplicateCount = 0;
    _duplicatePaddedCount = 0;
    _gpuBuildDirty = true;
}

void OfficialGaussianRasterPass::Execute(const RenderGraphContext& context)
{
    if (_scene == nullptr || _camera == nullptr || !_scene->HasTrainedGaussians() || !_scene->HasGaussianSplats()) {
        ClearStorageOutput(context, _accumOutput, _revealOutput);
        return;
    }
    const uint32_t sourceGaussianCount = _scene->GetGaussianCount();
    if (sourceGaussianCount == 0u || !_scene->GetGaussianBuffer()) {
        ClearStorageOutput(context, _accumOutput, _revealOutput);
        return;
    }

    if (NeedsFrameDataRebuild(context.GetRenderExtent())) {
        RebuildFrameDataIfNeeded(context.GetRenderExtent());
    }
    const uint32_t tileCountX = std::max(1u, (context.GetRenderExtent().width + kTileSize - 1u) / kTileSize);
    const uint32_t tileCountY = std::max(1u, (context.GetRenderExtent().height + kTileSize - 1u) / kTileSize);
    const size_t tileCount = static_cast<size_t>(tileCountX) * tileCountY;
    if (_statistics.projectedCount == 0u || tileCount == 0) {
        ClearStorageOutput(context, _accumOutput, _revealOutput);
        return;
    }

    RenderDevice& device = context.GetDevice();
    const float estimatedTilesTouched = _statistics.averageTilesTouched > 0.0f ? _statistics.averageTilesTouched : 6.0f;
    const size_t estimatedDuplicateCapacity = std::max<size_t>(
        _duplicateCapacity,
        std::max<size_t>(static_cast<size_t>(std::ceil(double(sourceGaussianCount) * std::max(estimatedTilesTouched * 1.5f, 6.0f))), 1u));
    EnsureResources(device, context.GetRenderExtent(), sourceGaussianCount, estimatedDuplicateCapacity, estimatedDuplicateCapacity, tileCount);
    if (!_projectedBuffer || !_duplicateKeyBuffer || !_duplicateValueBuffer || !_duplicateScratchKeyBuffer || !_duplicateScratchValueBuffer
        || !_scanBlockSumBuffer || !_duplicateCountBuffer
        || !_radixHistogramBuffer || !_radixBinBaseBuffer || !_tileRangeBuffer) {
        ClearStorageOutput(context, _accumOutput, _revealOutput);
        return;
    }

    VkDescriptorBufferInfo projectedInfo = vkinit::buffer_info(device.GetBuffer(_projectedBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo duplicateKeyInfo = vkinit::buffer_info(device.GetBuffer(_duplicateKeyBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo duplicateValueInfo = vkinit::buffer_info(device.GetBuffer(_duplicateValueBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo duplicateScratchKeyInfo = vkinit::buffer_info(device.GetBuffer(_duplicateScratchKeyBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo duplicateScratchValueInfo = vkinit::buffer_info(device.GetBuffer(_duplicateScratchValueBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo scanBlockSumInfo = vkinit::buffer_info(device.GetBuffer(_scanBlockSumBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo duplicateCountInfo = vkinit::buffer_info(device.GetBuffer(_duplicateCountBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo radixHistogramInfo = vkinit::buffer_info(device.GetBuffer(_radixHistogramBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo radixBinBaseInfo = vkinit::buffer_info(device.GetBuffer(_radixBinBaseBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo tileRangeInfo = vkinit::buffer_info(device.GetBuffer(_tileRangeBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorImageInfo accumInfo{};
    accumInfo.imageView = context.GetTextureView(_accumOutput);
    accumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo revealInfo{};
    revealInfo.imageView = context.GetTextureView(_revealOutput);
    revealInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 12> writes{
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &projectedInfo, 0),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &duplicateKeyInfo, 1),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &duplicateValueInfo, 2),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &duplicateScratchKeyInfo, 3),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &duplicateScratchValueInfo, 4),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &radixHistogramInfo, 5),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &radixBinBaseInfo, 6),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &tileRangeInfo, 7),
        vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _descriptorSet, &accumInfo, 8),
        vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _descriptorSet, &revealInfo, 9),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &scanBlockSumInfo, 10),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _descriptorSet, &duplicateCountInfo, 11),
    };
    vkUpdateDescriptorSets(device.GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    ClearStorageOutput(context, _accumOutput, _revealOutput);

    const uint32_t depthImageIndex =
        _depthInput ? device.GetImageResource(context.GetTextureHandle(_depthInput)).bindless.sampledImage : kInvalidImageIndex;
    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    const std::array<VkDescriptorSet, 2> descriptorSets{ device.GetBindless().GetSet(), _descriptorSet };
    vkCmdBindDescriptorSets(commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        _pipelineLayout,
        0,
        static_cast<uint32_t>(descriptorSets.size()),
        descriptorSets.data(),
        0,
        nullptr);
    GaussianComputePushConstants pushConstants{};

    if (_gpuBuildDirty) {
        vkCmdFillBuffer(commandBuffer, device.GetBuffer(_tileRangeBuffer), 0, sizeof(glm::uvec2) * tileCount, 0xFFFFFFFFu);
        vkCmdFillBuffer(commandBuffer, device.GetBuffer(_duplicateCountBuffer), 0, sizeof(uint32_t) * 16u, 0u);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        const uint32_t sourceGaussianBufferIndex = device.GetBufferResource(_scene->GetGaussianBuffer()).bindless.storageBuffer;
        const uint32_t preprocessFlags = (_viewDependentColor ? 1u : 0u) | (_antialiasing ? 2u : 0u) | (_fastCulling ? 4u : 0u);
        const uint32_t preprocessBlockCount = static_cast<uint32_t>((sourceGaussianCount + 127u) / 128u);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _preprocessPipeline);
        pushConstants = {};
        pushConstants.params0 = glm::uvec4(sourceGaussianCount, context.GetRenderExtent().width, context.GetRenderExtent().height, sourceGaussianBufferIndex);
        pushConstants.params1 = glm::uvec4(_shDegree, tileCountX, tileCountY, preprocessFlags);
        pushConstants.params2 = glm::vec4(_opacity, _alphaCutoff, std::max(_alphaCutoff * 1.5f, 0.00075f), 0.0f);
        pushConstants.viewMatrix = _camera->GetViewMatrix();
        pushConstants.viewProjection = _camera->GetViewProjection();
        pushConstants.cameraPositionAndSceneType = glm::vec4(_camera->GetPosition(), 0.0f);
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, preprocessBlockCount, 1, 1);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        const uint32_t projectedBlockCount = static_cast<uint32_t>((sourceGaussianCount + 255u) / 256u);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _scanPipeline);
        pushConstants = {};
        pushConstants.params0 = glm::uvec4(sourceGaussianCount, projectedBlockCount, 0u, static_cast<uint32_t>(_duplicateCapacity));
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, projectedBlockCount, 1, 1);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        pushConstants.params0.z = 1u;
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        pushConstants.params0.z = 2u;
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, projectedBlockCount, 1, 1);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

    if (_gpuBuildDirty) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _duplicatePipeline);
        pushConstants = {};
        pushConstants.params0 = glm::uvec4(sourceGaussianCount, tileCountX, tileCountY, static_cast<uint32_t>(_duplicateCapacity));
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, static_cast<uint32_t>((sourceGaussianCount + 63u) / 64u), 1, 1);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        if (_duplicateCapacity > 1) {
            const uint32_t radixBlockCount = static_cast<uint32_t>((_duplicateCapacity + 255u) / 256u);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _sortPipeline);
            uint32_t inputIndex = 0u;
            uint32_t outputIndex = 1u;
            for (uint32_t shift = 0u; shift < 32u; shift += 4u) {
                pushConstants = {};
                pushConstants.params0 = glm::uvec4(0u, shift, 0u, 0u);
                pushConstants.params1 = glm::uvec4(inputIndex, outputIndex, 0u, 0u);
                vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
                vkCmdDispatch(commandBuffer, radixBlockCount, 1, 1);
                InsertMemoryBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                pushConstants.params0.w = 1u;
                vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                InsertMemoryBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                pushConstants.params0.w = 2u;
                vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                InsertMemoryBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                pushConstants.params0.w = 3u;
                vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
                vkCmdDispatch(commandBuffer, radixBlockCount, 1, 1);
                InsertMemoryBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                std::swap(inputIndex, outputIndex);
            }
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _rangePipeline);
        pushConstants = {};
        pushConstants.params0 = glm::uvec4(0u, tileCountX * tileCountY, 0u, 0u);
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, static_cast<uint32_t>((_duplicateCapacity + 255u) / 256u), 1, 1);
        InsertMemoryBarrier(commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        _gpuBuildDirty = false;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _rasterPipeline);
    pushConstants = {};
    pushConstants.params0 = glm::uvec4(tileCountX, tileCountY, depthImageIndex, 0u);
    pushConstants.params1 = glm::uvec4(static_cast<uint32_t>(context.GetRenderExtent().width), static_cast<uint32_t>(context.GetRenderExtent().height), 0u, 0u);
    pushConstants.params2 = glm::vec4(1.0f, _alphaCutoff, std::max(_alphaCutoff * 1.5f, 0.00075f), 0.0f);
    vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, tileCountX, tileCountY, 1);
}

void OfficialGaussianRasterPass::Shutdown(RenderDevice& device)
{
    DestroyResources(device);
    const VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }
    if (_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vkDevice, _descriptorPool, nullptr);
        _descriptorPool = VK_NULL_HANDLE;
    }
    if (_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vkDevice, _descriptorSetLayout, nullptr);
        _descriptorSetLayout = VK_NULL_HANDLE;
    }
    _descriptorSet = VK_NULL_HANDLE;
    vkutil::destroy_pipeline(vkDevice, _preprocessPipeline);
    vkutil::destroy_pipeline(vkDevice, _duplicatePipeline);
    vkutil::destroy_pipeline(vkDevice, _scanPipeline);
    vkutil::destroy_pipeline(vkDevice, _sortPipeline);
    vkutil::destroy_pipeline(vkDevice, _rangePipeline);
    vkutil::destroy_pipeline(vkDevice, _rasterPipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _preprocessShader);
    vkutil::destroy_shader_module(vkDevice, _duplicateShader);
    vkutil::destroy_shader_module(vkDevice, _scanShader);
    vkutil::destroy_shader_module(vkDevice, _sortShader);
    vkutil::destroy_shader_module(vkDevice, _rangeShader);
    vkutil::destroy_shader_module(vkDevice, _rasterShader);
}

} // namespace vesta::render

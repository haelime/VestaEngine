#include <vesta/render/passes/restir_di_pass.h>

#include <algorithm>
#include <array>

#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
struct RestirDiPushConstants {
    glm::uvec4 bufferIndices{ kInvalidResourceIndex, kInvalidResourceIndex, 0u, 0u };
    glm::uvec4 dispatchParams{ 1u, 1u, 1u, 1u };
    glm::uvec4 lightParams{ 1u, 1u, 0u, 0u };
    glm::uvec4 flags{ 0u, 0u, 0u, 0u };
};

static_assert(sizeof(RestirDiPushConstants) <= 128, "ReSTIR push constants must stay compact.");
} // namespace

void RestirDiPass::SetReservoirBuffers(BufferHandle current, BufferHandle history)
{
    _currentReservoir = current;
    _historyReservoir = history;
}

void RestirDiPass::SetControls(uint32_t frameIndex,
    uint32_t width,
    uint32_t height,
    uint32_t candidateLightCount,
    uint32_t reservoirCount,
    uint32_t spatialSamples,
    uint32_t activeLightCount,
    uint32_t localLightCount,
    uint32_t emissiveTriangleCount,
    bool temporalReuse,
    bool spatialReuse)
{
    _frameIndex = frameIndex;
    _width = std::max(1u, width);
    _height = std::max(1u, height);
    _candidateLightCount = std::clamp(candidateLightCount, 1u, 64u);
    _reservoirCount = std::clamp(reservoirCount, 1u, 8u);
    _spatialSamples = std::clamp(spatialSamples, 0u, 16u);
    _activeLightCount = std::max(1u, activeLightCount);
    _localLightCount = std::min(localLightCount, _activeLightCount);
    _emissiveTriangleCount = emissiveTriangleCount;
    _temporalReuse = temporalReuse;
    _spatialReuse = spatialReuse;
}

void RestirDiPass::Initialize(RenderDevice& device)
{
    _backendAvailable = false;
    if (device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }
    if (_pipeline != VK_NULL_HANDLE) {
        _backendAvailable = true;
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/restir_di.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{
        device.GetBindless().GetLayout(),
    };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(RestirDiPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
    _backendAvailable = _pipeline != VK_NULL_HANDLE;
}

void RestirDiPass::Setup(RenderGraphBuilder&)
{
}

void RestirDiPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || !_currentReservoir) {
        return;
    }

    const auto& current = context.GetDevice().GetBufferResource(_currentReservoir);
    const bool writeHistory = _temporalReuse && _historyReservoir;
    const uint32_t historyIndex = writeHistory
        ? context.GetDevice().GetBufferResource(_historyReservoir).bindless.storageBuffer
        : kInvalidResourceIndex;
    const RestirDiPushConstants pushConstants{
        .bufferIndices = glm::uvec4(current.bindless.storageBuffer, historyIndex, 0u, 0u),
        .dispatchParams = glm::uvec4(_width, _height, _reservoirCount, _candidateLightCount),
        .lightParams = glm::uvec4(_activeLightCount, _localLightCount, _emissiveTriangleCount, _spatialSamples),
        .flags = glm::uvec4(_frameIndex, _temporalReuse ? 1u : 0u, _spatialReuse ? 1u : 0u, writeHistory ? 1u : 0u),
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    const std::array<VkDescriptorSet, 1> descriptorSets{
        context.GetDevice().GetBindless().GetSet(),
    };
    vkCmdBindDescriptorSets(commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        _pipelineLayout,
        0,
        static_cast<uint32_t>(descriptorSets.size()),
        descriptorSets.data(),
        0,
        nullptr);
    vkCmdPushConstants(commandBuffer,
        _pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(RestirDiPushConstants),
        &pushConstants);

    const uint64_t totalReservoirs = static_cast<uint64_t>(_width) * _height * _reservoirCount;
    vkCmdDispatch(commandBuffer, static_cast<uint32_t>((totalReservoirs + 255u) / 256u), 1, 1);

    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void RestirDiPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }
    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
    _backendAvailable = false;
}
} // namespace vesta::render

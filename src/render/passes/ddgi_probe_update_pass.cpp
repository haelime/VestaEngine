#include <vesta/render/passes/ddgi_probe_update_pass.h>

#include <algorithm>

#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct DdgiProbeUpdatePushConstants {
    glm::uvec4 bufferIndices{ kInvalidResourceIndex, kInvalidResourceIndex, 0u, 0u };
    glm::uvec4 gridAndFrame{ 8u, 4u, 8u, 0u };
    glm::uvec4 rayParams{ 128u, 0u, 0u, 0u };
    glm::vec4 probeParams{ 2.0f, 0.95f, 0.0f, 0.0f };
    glm::vec4 lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::vec4 directionalLightColor{ 1.0f };
    glm::vec4 environmentParams{ 1.0f, 0.0f, 0.0f, 0.22f };
};

static_assert(sizeof(DdgiProbeUpdatePushConstants) <= 256, "DDGI push constants must fit common Vulkan limits.");
} // namespace

void DdgiProbeUpdatePass::SetProbeBuffers(BufferHandle irradiance, BufferHandle visibility)
{
    _irradianceBuffer = irradiance;
    _visibilityBuffer = visibility;
}

void DdgiProbeUpdatePass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void DdgiProbeUpdatePass::SetFrameSlot(uint32_t frameSlot)
{
    _frameSlot = frameSlot % static_cast<uint32_t>(_descriptorSets.size());
}

void DdgiProbeUpdatePass::SetFrameIndex(uint32_t frameIndex)
{
    _frameIndex = frameIndex;
}

void DdgiProbeUpdatePass::SetControls(uint32_t probeCountX,
    uint32_t probeCountY,
    uint32_t probeCountZ,
    uint32_t raysPerProbe,
    float probeSpacing,
    float hysteresis,
    glm::vec4 lightDirectionAndIntensity,
    glm::vec4 directionalLightColor,
    glm::vec4 environmentParams)
{
    _probeCountX = std::clamp(probeCountX, 1u, 32u);
    _probeCountY = std::clamp(probeCountY, 1u, 16u);
    _probeCountZ = std::clamp(probeCountZ, 1u, 32u);
    _raysPerProbe = std::clamp(raysPerProbe, 16u, 1024u);
    _probeSpacing = std::clamp(probeSpacing, 0.25f, 10.0f);
    _hysteresis = std::clamp(hysteresis, 0.0f, 1.0f);
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
    _directionalLightColor = directionalLightColor;
    _environmentParams = environmentParams;
}

void DdgiProbeUpdatePass::Initialize(RenderDevice& device)
{
    _backendAvailable = false;
    if (device.GetDevice() == VK_NULL_HANDLE || device.GetRayTracingSupport().rayQueryFeatures.rayQuery != VK_TRUE) {
        return;
    }
    if (_pipeline != VK_NULL_HANDLE) {
        _backendAvailable = true;
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/ddgi_probe_update.comp.spv"));

    const std::array<VkDescriptorPoolSize, 1> poolSizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, static_cast<uint32_t>(_descriptorSets.size()) },
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = static_cast<uint32_t>(_descriptorSets.size());
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &_descriptorPool));

    std::array<VkDescriptorSetLayoutBinding, 1> bindings{
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT, 0),
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo =
        vkinit::descriptorset_layout_create_info(bindings.data(), static_cast<uint32_t>(bindings.size()));
    VK_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &_descriptorSetLayout));

    const std::array<VkDescriptorSetLayout, 2> allocationLayouts{ _descriptorSetLayout, _descriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(_descriptorSets.size());
    allocInfo.pSetLayouts = allocationLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(vkDevice, &allocInfo, _descriptorSets.data()));

    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{ device.GetBindless().GetLayout(), _descriptorSetLayout };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(DdgiProbeUpdatePushConstants) },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
    _backendAvailable = _pipeline != VK_NULL_HANDLE;
}

void DdgiProbeUpdatePass::Setup(RenderGraphBuilder&)
{
}

void DdgiProbeUpdatePass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _scene == nullptr || !_scene->HasRayTracingScene() || !_irradianceBuffer || !_visibilityBuffer) {
        return;
    }

    const uint32_t irradianceIndex = context.GetDevice().GetBufferResource(_irradianceBuffer).bindless.storageBuffer;
    const uint32_t visibilityIndex = context.GetDevice().GetBufferResource(_visibilityBuffer).bindless.storageBuffer;
    const DdgiProbeUpdatePushConstants pushConstants{
        .bufferIndices = glm::uvec4(irradianceIndex, visibilityIndex, 0u, 0u),
        .gridAndFrame = glm::uvec4(_probeCountX, _probeCountY, _probeCountZ, _frameIndex),
        .rayParams = glm::uvec4(_raysPerProbe, 0u, 0u, 0u),
        .probeParams = glm::vec4(_probeSpacing, _hysteresis, 0.0f, 0.0f),
        .lightDirectionAndIntensity = _lightDirectionAndIntensity,
        .directionalLightColor = _directionalLightColor,
        .environmentParams = _environmentParams,
    };

    const VkAccelerationStructureKHR topLevelAccelerationStructure = _scene->GetTopLevelAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR accelerationStructureWrite{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR
    };
    accelerationStructureWrite.accelerationStructureCount = 1;
    accelerationStructureWrite.pAccelerationStructures = &topLevelAccelerationStructure;

    VkWriteDescriptorSet accelerationWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    const VkDescriptorSet descriptorSet = _descriptorSets[_frameSlot];
    accelerationWrite.pNext = &accelerationStructureWrite;
    accelerationWrite.dstSet = descriptorSet;
    accelerationWrite.dstBinding = 0;
    accelerationWrite.descriptorCount = 1;
    accelerationWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(context.GetDevice().GetDevice(), 1, &accelerationWrite, 0, nullptr);

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    const std::array<VkDescriptorSet, 2> descriptorSets{ context.GetDevice().GetBindless().GetSet(), descriptorSet };
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
        sizeof(DdgiProbeUpdatePushConstants),
        &pushConstants);

    const uint32_t totalRays = _probeCountX * _probeCountY * _probeCountZ * _raysPerProbe;
    vkCmdDispatch(commandBuffer, (totalRays + 127u) / 128u, 1, 1);

    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void DdgiProbeUpdatePass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }
    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
    if (_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vkDevice, _descriptorPool, nullptr);
        _descriptorPool = VK_NULL_HANDLE;
    }
    if (_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vkDevice, _descriptorSetLayout, nullptr);
        _descriptorSetLayout = VK_NULL_HANDLE;
    }
    _descriptorSets = {};
    _backendAvailable = false;
}
} // namespace vesta::render

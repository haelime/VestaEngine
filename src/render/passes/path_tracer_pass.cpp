#include <vesta/render/passes/path_tracer_pass.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include <glm/glm.hpp>

#include <vesta/render/renderer.h>
#include <vesta/render/rhi/render_device.h>
#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct ComputePathTracePushConstants {
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::vec4 cameraPositionAndFrame{ 0.0f };
    uint32_t outputImageIndex{ 0 };
    uint32_t triangleBufferIndex{ 0 };
    uint32_t triangleCount{ 0 };
    uint32_t frameIndex{ 0 };
};

struct HardwarePathTracePushConstants {
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::vec4 cameraPositionAndFrame{ 0.0f };
    uint32_t triangleBufferIndex{ 0 };
    uint32_t triangleCount{ 0 };
    uint32_t frameIndex{ 0 };
    uint32_t reserved{ 0 };
};

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void ClearOutput(const RenderGraphContext& context, GraphTextureHandle output)
{
    VkClearColorValue clearValue{};
    clearValue.float32[3] = 1.0f;
    const VkImageSubresourceRange clearRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(context.GetCommandBuffer(),
        context.GetDevice().GetImage(context.GetTextureHandle(output)),
        VK_IMAGE_LAYOUT_GENERAL,
        &clearValue,
        1,
        &clearRange);
}
} // namespace

void PathTracerPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void PathTracerPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void PathTracerPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void PathTracerPass::SetFrameIndex(uint32_t frameIndex)
{
    _frameIndex = frameIndex;
}

void PathTracerPass::SetFrameSlot(uint32_t frameSlot)
{
    _frameSlot = frameSlot % static_cast<uint32_t>(_rtDescriptorSets.size());
}

void PathTracerPass::SetEnabled(bool enabled)
{
    _enabled = enabled;
}

void PathTracerPass::SetBackendPreference(PathTraceBackend backend)
{
    _backendPreference = backend;
}

void PathTracerPass::Initialize(RenderDevice& device)
{
    if (_pipeline == VK_NULL_HANDLE && device.GetDevice() != VK_NULL_HANDLE) {
        VkDevice vkDevice = device.GetDevice();
        _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/pathtrace.comp.spv"));

        const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
        const std::array<VkPushConstantRange, 1> pushConstants{
            VkPushConstantRange{
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = sizeof(ComputePathTracePushConstants),
            },
        };
        _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

        vkutil::ComputePipelineDesc pipelineDesc{};
        pipelineDesc.layout = _pipelineLayout;
        pipelineDesc.computeShader = _computeShader;
        _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
    }

    if (!device.IsRayTracingSupported() || _rtPipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    const RayTracingFunctions& rt = device.GetRayTracingFunctions();
    _raygenShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/pathtrace.rgen.spv"));
    _missShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/pathtrace.rmiss.spv"));
    _closestHitShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/pathtrace.rchit.spv"));

    const std::array<VkDescriptorPoolSize, 2> poolSizes{
        VkDescriptorPoolSize{
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            static_cast<uint32_t>(_rtDescriptorSets.size()),
        },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, static_cast<uint32_t>(_rtDescriptorSets.size()) },
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = static_cast<uint32_t>(_rtDescriptorSets.size());
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &_rtDescriptorPool));

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{
        vkinit::descriptorset_layout_binding(
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0),
        vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1),
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo =
        vkinit::descriptorset_layout_create_info(bindings.data(), static_cast<uint32_t>(bindings.size()));
    VK_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &_rtDescriptorSetLayout));

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _rtDescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(_rtDescriptorSets.size());
    const std::array<VkDescriptorSetLayout, 2> rtSetLayouts{ _rtDescriptorSetLayout, _rtDescriptorSetLayout };
    allocInfo.pSetLayouts = rtSetLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(vkDevice, &allocInfo, _rtDescriptorSets.data()));

    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{
        device.GetBindless().GetLayout(),
        _rtDescriptorSetLayout,
    };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            .offset = 0,
            .size = sizeof(HardwarePathTracePushConstants),
        },
    };
    _rtPipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    std::array<VkPipelineShaderStageCreateInfo, 3> stages{
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_RAYGEN_BIT_KHR, _raygenShader),
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_MISS_BIT_KHR, _missShader),
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, _closestHitShader),
    };

    std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout = _rtPipelineLayout;
    VK_CHECK(rt.vkCreateRayTracingPipelinesKHR(
        vkDevice, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_rtPipeline));

    const RayTracingSupport& rtSupport = device.GetRayTracingSupport();
    const uint32_t handleSize = rtSupport.rayTracingPipelineProperties.shaderGroupHandleSize;
    const uint32_t handleAlignment = rtSupport.rayTracingPipelineProperties.shaderGroupHandleAlignment;
    const uint32_t baseAlignment = rtSupport.rayTracingPipelineProperties.shaderGroupBaseAlignment;
    const uint32_t handleSizeAligned = AlignUp(handleSize, handleAlignment);
    const uint32_t groupStride = AlignUp(handleSizeAligned, baseAlignment);
    const uint32_t groupCount = static_cast<uint32_t>(groups.size());

    std::vector<uint8_t> shaderHandles(handleSize * groupCount);
    VK_CHECK(rt.vkGetRayTracingShaderGroupHandlesKHR(
        vkDevice, _rtPipeline, 0, groupCount, shaderHandles.size(), shaderHandles.data()));

    constexpr VmaAllocationCreateFlags kMappedHostFlags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    _shaderBindingTable = device.CreateBuffer(BufferDesc{
        .size = static_cast<VkDeviceSize>(groupStride) * groupCount,
        .usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .debugName = "PathTraceSBT",
    });

    AllocatedBuffer sbtBuffer = device.GetBufferResource(_shaderBindingTable);
    auto* mappedSbt = static_cast<std::byte*>(sbtBuffer.allocationInfo.pMappedData);
    for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
        std::memcpy(mappedSbt + static_cast<size_t>(groupStride) * groupIndex,
            shaderHandles.data() + static_cast<size_t>(handleSize) * groupIndex,
            handleSize);
    }

    const VkDeviceAddress sbtAddress = device.GetBufferDeviceAddress(_shaderBindingTable);
    _raygenSbt = VkStridedDeviceAddressRegionKHR{ sbtAddress, groupStride, groupStride };
    _missSbt = VkStridedDeviceAddressRegionKHR{ sbtAddress + groupStride, groupStride, groupStride };
    _hitSbt = VkStridedDeviceAddressRegionKHR{ sbtAddress + groupStride * 2u, groupStride, groupStride };
    _callableSbt = {};
}

void PathTracerPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void PathTracerPass::Execute(const RenderGraphContext& context)
{
    _activeBackend = PathTraceBackend::Compute;

    if (!_enabled || _scene == nullptr || _camera == nullptr || !_scene->IsLoaded() || !_scene->GetTriangleBuffer()) {
        ClearOutput(context, _output);
        return;
    }

    const bool wantsHardwareRt = _backendPreference != PathTraceBackend::Compute;
    const bool canUseHardwareRt = wantsHardwareRt && context.GetDevice().IsRayTracingSupported()
        && _scene->HasRayTracingScene() && _rtPipeline != VK_NULL_HANDLE;

    if (canUseHardwareRt) {
        _activeBackend = PathTraceBackend::HardwareRT;
        const RayTracingFunctions& rt = context.GetDevice().GetRayTracingFunctions();

        const HardwarePathTracePushConstants pushConstants{
            .inverseViewProjection = _camera->GetInverseViewProjection(),
            .cameraPositionAndFrame = glm::vec4(_camera->GetPosition(), static_cast<float>(_frameIndex)),
            .triangleBufferIndex = context.GetDevice().GetBufferResource(_scene->GetTriangleBuffer()).bindless.storageBuffer,
            .triangleCount = static_cast<uint32_t>(_scene->GetTriangles().size()),
            .frameIndex = _frameIndex,
        };

        const VkAccelerationStructureKHR topLevelAccelerationStructure = _scene->GetTopLevelAccelerationStructure();
        VkWriteDescriptorSetAccelerationStructureKHR accelerationStructureWrite{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR
        };
        accelerationStructureWrite.accelerationStructureCount = 1;
        accelerationStructureWrite.pAccelerationStructures = &topLevelAccelerationStructure;

        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageView = context.GetTextureView(_output);
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].pNext = &accelerationStructureWrite;
        const VkDescriptorSet rtDescriptorSet = _rtDescriptorSets[_frameSlot];
        writes[0].dstSet = rtDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1] =
            vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, rtDescriptorSet, &outputImageInfo, 1);
        vkUpdateDescriptorSets(context.GetDevice().GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkCommandBuffer commandBuffer = context.GetCommandBuffer();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _rtPipeline);

        const std::array<VkDescriptorSet, 2> descriptorSets{
            context.GetDevice().GetBindless().GetSet(),
            rtDescriptorSet,
        };
        vkCmdBindDescriptorSets(commandBuffer,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            _rtPipelineLayout,
            0,
            static_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            0,
            nullptr);
        vkCmdPushConstants(commandBuffer,
            _rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            0,
            sizeof(HardwarePathTracePushConstants),
            &pushConstants);

        const VkExtent3D outputExtent = context.GetTextureExtent(_output);
        rt.vkCmdTraceRaysKHR(
            commandBuffer, &_raygenSbt, &_missSbt, &_hitSbt, &_callableSbt, outputExtent.width, outputExtent.height, 1);
        return;
    }

    if (_pipeline == VK_NULL_HANDLE) {
        ClearOutput(context, _output);
        return;
    }

    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    const uint32_t outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage;
    const uint32_t triangleBufferIndex = context.GetDevice().GetBufferResource(_scene->GetTriangleBuffer()).bindless.storageBuffer;

    ComputePathTracePushConstants pushConstants{
        .inverseViewProjection = _camera->GetInverseViewProjection(),
        .cameraPositionAndFrame = glm::vec4(_camera->GetPosition(), static_cast<float>(_frameIndex)),
        .outputImageIndex = outputImageIndex,
        .triangleBufferIndex = triangleBufferIndex,
        .triangleCount = static_cast<uint32_t>(_scene->GetTriangles().size()),
        .frameIndex = _frameIndex,
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePathTracePushConstants), &pushConstants);

    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    vkCmdDispatch(commandBuffer, (outputExtent.width + 7) / 8, (outputExtent.height + 7) / 8, 1);
}

void PathTracerPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    if (_shaderBindingTable) {
        device.DestroyBuffer(_shaderBindingTable);
        _shaderBindingTable = {};
    }

    vkutil::destroy_pipeline(vkDevice, _rtPipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _rtPipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _raygenShader);
    vkutil::destroy_shader_module(vkDevice, _missShader);
    vkutil::destroy_shader_module(vkDevice, _closestHitShader);

    if (_rtDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vkDevice, _rtDescriptorSetLayout, nullptr);
        _rtDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (_rtDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vkDevice, _rtDescriptorPool, nullptr);
        _rtDescriptorPool = VK_NULL_HANDLE;
    }
    _rtDescriptorSets.fill(VK_NULL_HANDLE);

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
}
} // namespace vesta::render

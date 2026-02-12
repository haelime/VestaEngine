#include <vesta/render/passes/ray_effects_pass.h>

#include <array>
#include <algorithm>

#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct RayEffectsPushConstants {
    uint32_t normalImageIndex{ kInvalidResourceIndex };
    uint32_t depthImageIndex{ kInvalidResourceIndex };
    uint32_t outputImageIndex{ kInvalidResourceIndex };
    uint32_t frameIndex{ 0 };
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::vec4 cameraPosition{ 0.0f };
    glm::vec4 lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::uvec4 sampleCounts{ 1u, 1u, 1u, 1u };
    glm::vec4 rayParams{ 100.0f, 2.0f, 0.0f, 0.0f };
    glm::uvec4 flags{ 0u, 0u, 0u, 0u };
};

static_assert(sizeof(RayEffectsPushConstants) <= 256, "Ray effects push constants must fit common Vulkan limits.");

void ClearRayEffectsOutput(const RenderGraphContext& context, GraphTextureHandle output)
{
    VkClearColorValue clearValue{};
    clearValue.float32[0] = 1.0f;
    clearValue.float32[1] = 1.0f;
    clearValue.float32[2] = 0.0f;
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

void RayEffectsPass::SetInputs(GraphTextureHandle normal, GraphTextureHandle depth)
{
    _normal = normal;
    _depth = depth;
}

void RayEffectsPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void RayEffectsPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void RayEffectsPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void RayEffectsPass::SetFrameSlot(uint32_t frameSlot)
{
    _frameSlot = frameSlot % static_cast<uint32_t>(_descriptorSets.size());
}

void RayEffectsPass::SetFrameIndex(uint32_t frameIndex)
{
    _frameIndex = frameIndex;
}

void RayEffectsPass::SetLight(glm::vec4 lightDirectionAndIntensity)
{
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
}

void RayEffectsPass::SetControls(bool shadowsEnabled,
    bool ambientOcclusionEnabled,
    bool reflectionsEnabled,
    bool globalIlluminationEnabled,
    uint32_t shadowSamples,
    uint32_t aoSamples,
    uint32_t reflectionSamples,
    uint32_t giSamples,
    float maxRayDistance,
    float aoRadius,
    float reflectionRoughnessCutoff)
{
    _shadowsEnabled = shadowsEnabled;
    _ambientOcclusionEnabled = ambientOcclusionEnabled;
    _reflectionsEnabled = reflectionsEnabled;
    _globalIlluminationEnabled = globalIlluminationEnabled;
    _shadowSamples = std::clamp(shadowSamples, 1u, 8u);
    _aoSamples = std::clamp(aoSamples, 1u, 8u);
    _reflectionSamples = std::clamp(reflectionSamples, 1u, 8u);
    _giSamples = std::clamp(giSamples, 1u, 8u);
    _maxRayDistance = std::clamp(maxRayDistance, 0.1f, 10000.0f);
    _aoRadius = std::clamp(aoRadius, 0.05f, 32.0f);
    _reflectionRoughnessCutoff = std::clamp(reflectionRoughnessCutoff, 0.0f, 1.0f);
}

void RayEffectsPass::Initialize(RenderDevice& device)
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
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/ray_effects.comp.spv"));

    const std::array<VkDescriptorPoolSize, 1> poolSizes{
        VkDescriptorPoolSize{
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            static_cast<uint32_t>(_descriptorSets.size()),
        },
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = static_cast<uint32_t>(_descriptorSets.size());
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &_descriptorPool));

    std::array<VkDescriptorSetLayoutBinding, 1> bindings{
        vkinit::descriptorset_layout_binding(
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_COMPUTE_BIT, 0),
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

    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{
        device.GetBindless().GetLayout(),
        _descriptorSetLayout,
    };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(RayEffectsPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
    _backendAvailable = _pipeline != VK_NULL_HANDLE;
}

void RayEffectsPass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_normal, ResourceUsage::StorageRead);
    builder.Read(_depth, ResourceUsage::SampledRead);
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void RayEffectsPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _scene == nullptr || _camera == nullptr || !_scene->HasRayTracingScene()) {
        ClearRayEffectsOutput(context, _output);
        return;
    }

    const ImageHandle normalHandle = context.GetTextureHandle(_normal);
    const ImageHandle depthHandle = context.GetTextureHandle(_depth);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    const RayEffectsPushConstants pushConstants{
        .normalImageIndex = context.GetDevice().GetImageResource(normalHandle).bindless.storageImage,
        .depthImageIndex = context.GetDevice().GetImageResource(depthHandle).bindless.sampledImage,
        .outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
        .frameIndex = _frameIndex,
        .inverseViewProjection = _camera->GetInverseViewProjection(),
        .cameraPosition = glm::vec4(_camera->GetPosition(), 0.0f),
        .lightDirectionAndIntensity = _lightDirectionAndIntensity,
        .sampleCounts = glm::uvec4(_shadowSamples, _aoSamples, _reflectionSamples, _giSamples),
        .rayParams = glm::vec4(_maxRayDistance, _aoRadius, _reflectionRoughnessCutoff, 0.0f),
        .flags = glm::uvec4(
            _shadowsEnabled ? 1u : 0u,
            _ambientOcclusionEnabled ? 1u : 0u,
            _reflectionsEnabled ? 1u : 0u,
            _globalIlluminationEnabled ? 1u : 0u),
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
    const std::array<VkDescriptorSet, 2> descriptorSets{
        context.GetDevice().GetBindless().GetSet(),
        descriptorSet,
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
        sizeof(RayEffectsPushConstants),
        &pushConstants);

    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    vkCmdDispatch(commandBuffer, (outputExtent.width + 7u) / 8u, (outputExtent.height + 7u) / 8u, 1);
}

void RayEffectsPass::Shutdown(RenderDevice& device)
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

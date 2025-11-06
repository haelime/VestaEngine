#include <vesta/render/passes/deferred_lighting_pass.h>

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
struct DeferredLightingPushConstants {
    uint32_t albedoImageIndex{ 0 };
    uint32_t normalImageIndex{ 0 };
    uint32_t outputImageIndex{ 0 };
    uint32_t padding{ 0 };
    glm::vec4 lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
};
} // namespace

void DeferredLightingPass::SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal)
{
    _albedo = albedo;
    _normal = normal;
}

void DeferredLightingPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void DeferredLightingPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/deferred_lighting.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(DeferredLightingPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
}

void DeferredLightingPass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_albedo, ResourceUsage::StorageRead);
    builder.Read(_normal, ResourceUsage::StorageRead);
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void DeferredLightingPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE) {
        return;
    }

    const ImageHandle albedoHandle = context.GetTextureHandle(_albedo);
    const ImageHandle normalHandle = context.GetTextureHandle(_normal);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);

    DeferredLightingPushConstants pushConstants{
        .albedoImageIndex = context.GetDevice().GetImageResource(albedoHandle).bindless.storageImage,
        .normalImageIndex = context.GetDevice().GetImageResource(normalHandle).bindless.storageImage,
        .outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
        _pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(DeferredLightingPushConstants),
        &pushConstants);

    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    vkCmdDispatch(commandBuffer, (outputExtent.width + 7) / 8, (outputExtent.height + 7) / 8, 1);
}

void DeferredLightingPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
}
} // namespace vesta::render

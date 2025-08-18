#include <vesta/render/passes/deferred_lighting_pass.h>

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>

namespace vesta::render {
namespace {
// Bindless slots are pushed instead of descriptors so the compute shader can
// fetch the right GBuffer images directly from the global bindless set.
struct DeferredLightingPushConstants {
    uint32_t albedoImageIndex{ 0 };
    uint32_t normalImageIndex{ 0 };
    uint32_t materialImageIndex{ 0 };
    uint32_t depthImageIndex{ 0 };
    uint32_t outputImageIndex{ 0 };
    uint32_t padding0{ 0 };
    uint32_t padding1{ 0 };
    uint32_t padding2{ 0 };
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::vec4 cameraPosition{ 0.0f };
    glm::vec4 lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
};
} // namespace

void DeferredLightingPass::SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth)
{
    _albedo = albedo;
    _normal = normal;
    _material = material;
    _depth = depth;
}

void DeferredLightingPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void DeferredLightingPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void DeferredLightingPass::SetLight(glm::vec4 lightDirectionAndIntensity)
{
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
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
    builder.Read(_material, ResourceUsage::StorageRead);
    builder.Read(_depth, ResourceUsage::SampledRead);
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void DeferredLightingPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _camera == nullptr) {
        return;
    }

    const ImageHandle albedoHandle = context.GetTextureHandle(_albedo);
    const ImageHandle normalHandle = context.GetTextureHandle(_normal);
    const ImageHandle materialHandle = context.GetTextureHandle(_material);
    const ImageHandle depthHandle = context.GetTextureHandle(_depth);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);

    DeferredLightingPushConstants pushConstants{
        .albedoImageIndex = context.GetDevice().GetImageResource(albedoHandle).bindless.storageImage,
        .normalImageIndex = context.GetDevice().GetImageResource(normalHandle).bindless.storageImage,
        .materialImageIndex = context.GetDevice().GetImageResource(materialHandle).bindless.storageImage,
        .depthImageIndex = context.GetDevice().GetImageResource(depthHandle).bindless.sampledImage,
        .outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
        .inverseViewProjection = _camera->GetInverseViewProjection(),
        .cameraPosition = glm::vec4(_camera->GetPosition(), 1.0f),
        .lightDirectionAndIntensity = _lightDirectionAndIntensity,
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

    // One thread group shades an 8x8 tile.
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

#include <vesta/render/passes/path_tracer_pass.h>

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct PathTracePushConstants {
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::vec4 cameraPositionAndFrame{ 0.0f };
    uint32_t outputImageIndex{ 0 };
    uint32_t triangleBufferIndex{ 0 };
    uint32_t triangleCount{ 0 };
    uint32_t frameIndex{ 0 };
};
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

void PathTracerPass::SetEnabled(bool enabled)
{
    _enabled = enabled;
}

void PathTracerPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/pathtrace.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PathTracePushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
}

void PathTracerPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void PathTracerPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE) {
        return;
    }

    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    VkCommandBuffer commandBuffer = context.GetCommandBuffer();

    if (!_enabled || _scene == nullptr || _camera == nullptr || !_scene->IsLoaded() || !_scene->GetTriangleBuffer()) {
        VkClearColorValue clearValue{};
        clearValue.float32[3] = 1.0f;
        const VkImageSubresourceRange clearRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        vkCmdClearColorImage(commandBuffer,
            context.GetDevice().GetImage(outputHandle),
            VK_IMAGE_LAYOUT_GENERAL,
            &clearValue,
            1,
            &clearRange);
        return;
    }

    const uint32_t outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage;
    const uint32_t triangleBufferIndex = context.GetDevice().GetBufferResource(_scene->GetTriangleBuffer()).bindless.storageBuffer;

    PathTracePushConstants pushConstants{
        .inverseViewProjection = _camera->GetInverseViewProjection(),
        .cameraPositionAndFrame = glm::vec4(_camera->GetPosition(), static_cast<float>(_frameIndex)),
        .outputImageIndex = outputImageIndex,
        .triangleBufferIndex = triangleBufferIndex,
        .triangleCount = static_cast<uint32_t>(_scene->GetTriangles().size()),
        .frameIndex = _frameIndex,
    };

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PathTracePushConstants), &pushConstants);

    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    vkCmdDispatch(commandBuffer, (outputExtent.width + 7) / 8, (outputExtent.height + 7) / 8, 1);
}

void PathTracerPass::Shutdown(RenderDevice& device)
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

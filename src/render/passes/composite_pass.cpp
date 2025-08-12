#include <vesta/render/passes/composite_pass.h>

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
// mode selects which intermediate result to visualize. params.x is currently
// used for gaussian blending strength in composite mode.
struct CompositePushConstants {
    uint32_t deferredImageIndex{ 0 };
    uint32_t pathTraceImageIndex{ 0 };
    uint32_t gaussianImageIndex{ 0 };
    uint32_t mode{ 0 };
    glm::vec4 params{ 0.25f, 0.0f, 0.0f, 0.0f };
};
} // namespace

void CompositePass::SetInputs(GraphTextureHandle deferredLighting, GraphTextureHandle pathTrace, GraphTextureHandle gaussian)
{
    _deferredLighting = deferredLighting;
    _pathTrace = pathTrace;
    _gaussian = gaussian;
}

void CompositePass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void CompositePass::SetMode(uint32_t mode, float gaussianMix)
{
    _mode = mode;
    _gaussianMix = gaussianMix;
}

void CompositePass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/composite.vert.spv"));
    _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/composite.frag.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(CompositePushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.colorFormats = { device.GetSwapchainFormat() };
    pipelineDesc.vertexShader = _vertexShader;
    pipelineDesc.fragmentShader = _fragmentShader;
    pipelineDesc.cullMode = VK_CULL_MODE_NONE;

    _pipeline = vkutil::create_graphics_pipeline(vkDevice, pipelineDesc);
}

void CompositePass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_deferredLighting, ResourceUsage::StorageRead);
    builder.Read(_pathTrace, ResourceUsage::StorageRead);
    builder.Read(_gaussian, ResourceUsage::StorageRead);
    builder.Write(_output, ResourceUsage::ColorAttachmentWrite);
}

void CompositePass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE) {
        return;
    }

    const ImageHandle deferredHandle = context.GetTextureHandle(_deferredLighting);
    const ImageHandle pathTraceHandle = context.GetTextureHandle(_pathTrace);
    const ImageHandle gaussianHandle = context.GetTextureHandle(_gaussian);

    CompositePushConstants pushConstants{
        .deferredImageIndex = context.GetDevice().GetImageResource(deferredHandle).bindless.storageImage,
        .pathTraceImageIndex = context.GetDevice().GetImageResource(pathTraceHandle).bindless.storageImage,
        .gaussianImageIndex = context.GetDevice().GetImageResource(gaussianHandle).bindless.storageImage,
        .mode = _mode,
        .params = glm::vec4(_gaussianMix, 0.0f, 0.0f, 0.0f),
    };

    VkClearValue clearValue{};
    clearValue.color = { { 0.02f, 0.02f, 0.03f, 1.0f } };

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = context.GetTextureView(_output);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(context.GetRenderExtent().width);
    viewport.height = static_cast<float>(context.GetRenderExtent().height);
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = context.GetRenderExtent();

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, _pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CompositePushConstants), &pushConstants);
    // The full-screen triangle covers the entire frame without needing a vertex buffer.
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRendering(commandBuffer);
}

void CompositePass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _vertexShader);
    vkutil::destroy_shader_module(vkDevice, _fragmentShader);
}
} // namespace vesta::render

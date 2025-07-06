#include <vesta/render/passes/composite_pass.h>

#include <array>

#include <vesta/render/vulkan/vk_images.h>

namespace vesta::render {
namespace {
[[maybe_unused]] void example_blit_helper_usage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImage dstImage, VkExtent2D extent)
{
    const VkImageSubresourceRange colorRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkutil::transition_image(commandBuffer,
        srcImage,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT,
        colorRange);
    vkutil::transition_image(commandBuffer,
        dstImage,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        colorRange);
    vkutil::copy_image_to_image(commandBuffer, srcImage, dstImage, extent, extent);
}
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

void CompositePass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_deferredLighting, ResourceUsage::SampledRead);
    builder.Read(_pathTrace, ResourceUsage::SampledRead);
    builder.Read(_gaussian, ResourceUsage::SampledRead);
    builder.Write(_output, ResourceUsage::ColorAttachmentWrite);
}

void CompositePass::Execute(const RenderGraphContext& context)
{
    VkClearValue compositeClear{};
    compositeClear.color = { { 0.05f, 0.07f, 0.10f, 1.0f } };

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = context.GetTextureView(_output);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = compositeClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    // The graph has already synchronized all upstream outputs. Composition stays clear-only until real shaders land.
    vkCmdBeginRendering(context.GetCommandBuffer(), &renderingInfo);
    vkCmdEndRendering(context.GetCommandBuffer());
}
} // namespace vesta::render

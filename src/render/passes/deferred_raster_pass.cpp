#include <vesta/render/passes/deferred_raster_pass.h>

#include <array>

#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
[[maybe_unused]] void example_deferred_pipeline_boilerplate(VkDevice device, VkFormat colorFormat, VkFormat depthFormat)
{
    const std::array<VkDescriptorSetLayout, 0> descriptorSetLayouts{};
    const std::array<VkPushConstantRange, 0> pushConstants{};
    VkPipelineLayout layout = vkutil::create_pipeline_layout(device, descriptorSetLayouts, pushConstants);

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = layout;
    pipelineDesc.colorFormat = colorFormat;
    pipelineDesc.depthFormat = depthFormat;
    pipelineDesc.depthTestEnable = true;
    pipelineDesc.depthWriteEnable = true;

    // Real shader modules are intentionally not wired here yet; this is only the shared setup skeleton.
    (void)pipelineDesc;
    vkutil::destroy_pipeline_layout(device, layout);
}
} // namespace

void DeferredRasterPass::SetGBufferTargets(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle depth)
{
    _gbufferAlbedo = albedo;
    _gbufferNormal = normal;
    _sceneDepth = depth;
}

void DeferredRasterPass::SetLightingTarget(GraphTextureHandle lighting)
{
    _lightingTarget = lighting;
}

void DeferredRasterPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_gbufferAlbedo, ResourceUsage::ColorAttachmentWrite);
    builder.Write(_gbufferNormal, ResourceUsage::ColorAttachmentWrite);
    builder.Write(_sceneDepth, ResourceUsage::DepthAttachmentWrite);
    builder.Write(_lightingTarget, ResourceUsage::ColorAttachmentWrite);
}

void DeferredRasterPass::Execute(const RenderGraphContext& context)
{
    VkClearValue albedoClear{};
    albedoClear.color = { { 0.12f, 0.16f, 0.20f, 1.0f } };
    VkClearValue normalClear{};
    normalClear.color = { { 0.5f, 0.5f, 1.0f, 1.0f } };

    std::array<VkRenderingAttachmentInfo, 2> gbufferAttachments{};
    gbufferAttachments[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    gbufferAttachments[0].imageView = context.GetTextureView(_gbufferAlbedo);
    gbufferAttachments[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    gbufferAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    gbufferAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    gbufferAttachments[0].clearValue = albedoClear;

    gbufferAttachments[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    gbufferAttachments[1].imageView = context.GetTextureView(_gbufferNormal);
    gbufferAttachments[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    gbufferAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    gbufferAttachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    gbufferAttachments[1].clearValue = normalClear;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = context.GetTextureView(_sceneDepth);
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo gbufferRendering{};
    gbufferRendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    gbufferRendering.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    gbufferRendering.layerCount = 1;
    gbufferRendering.colorAttachmentCount = static_cast<uint32_t>(gbufferAttachments.size());
    gbufferRendering.pColorAttachments = gbufferAttachments.data();
    gbufferRendering.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(context.GetCommandBuffer(), &gbufferRendering);
    vkCmdEndRendering(context.GetCommandBuffer());

    VkClearValue lightingClear{};
    lightingClear.color = { { 0.16f, 0.22f, 0.30f, 1.0f } };

    VkRenderingAttachmentInfo lightingAttachment{};
    lightingAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    lightingAttachment.imageView = context.GetTextureView(_lightingTarget);
    lightingAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    lightingAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    lightingAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    lightingAttachment.clearValue = lightingClear;

    VkRenderingInfo lightingRendering{};
    lightingRendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    lightingRendering.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    lightingRendering.layerCount = 1;
    lightingRendering.colorAttachmentCount = 1;
    lightingRendering.pColorAttachments = &lightingAttachment;
    vkCmdBeginRendering(context.GetCommandBuffer(), &lightingRendering);
    vkCmdEndRendering(context.GetCommandBuffer());
}
} // namespace vesta::render

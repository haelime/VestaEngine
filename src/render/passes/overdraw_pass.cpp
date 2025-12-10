#include <vesta/render/passes/overdraw_pass.h>

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct OverdrawPushConstants {
    glm::mat4 viewProjection{ 1.0f };
};
} // namespace

void OverdrawPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void OverdrawPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void OverdrawPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void OverdrawPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/overdraw.vert.spv"));
    _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/overdraw.frag.spv"));

    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(OverdrawPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, {}, pushConstants);

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(vesta::scene::SceneVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 1> attributes{};
    attributes[0] = VkVertexInputAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, position),
    };

    VkPipelineColorBlendAttachmentState additiveBlend{};
    additiveBlend.blendEnable = VK_TRUE;
    additiveBlend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    additiveBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    additiveBlend.colorBlendOp = VK_BLEND_OP_ADD;
    additiveBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    additiveBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    additiveBlend.alphaBlendOp = VK_BLEND_OP_ADD;
    additiveBlend.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
    pipelineDesc.vertexShader = _vertexShader;
    pipelineDesc.fragmentShader = _fragmentShader;
    pipelineDesc.cullMode = VK_CULL_MODE_BACK_BIT;
    pipelineDesc.depthTestEnable = false;
    pipelineDesc.depthWriteEnable = false;
    pipelineDesc.vertexBindings = { binding };
    pipelineDesc.vertexAttributes = { attributes.begin(), attributes.end() };
    pipelineDesc.colorBlendAttachments = { additiveBlend };

    _pipeline = vkutil::create_graphics_pipeline(vkDevice, pipelineDesc);
}

void OverdrawPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_output, ResourceUsage::ColorAttachmentWrite);
}

void OverdrawPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _scene == nullptr || _camera == nullptr || !_scene->HasRasterGeometry()) {
        return;
    }

    VkClearValue clearValue{};
    clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

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

    OverdrawPushConstants pushConstants{
        .viewProjection = _camera->GetViewProjection(),
    };

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
    vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(OverdrawPushConstants), &pushConstants);

    const VkBuffer vertexBuffer = context.GetDevice().GetBuffer(_scene->GetVertexBuffer());
    const VkBuffer indexBuffer = context.GetDevice().GetBuffer(_scene->GetIndexBuffer());
    constexpr VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    for (const vesta::scene::SceneSurface& surface : _scene->GetSurfaces()) {
        vkCmdDrawIndexed(commandBuffer, surface.indexCount, 1, surface.firstIndex, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);
}

void OverdrawPass::Shutdown(RenderDevice& device)
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

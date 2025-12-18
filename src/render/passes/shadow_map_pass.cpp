#include <vesta/render/passes/shadow_map_pass.h>

#include <array>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct ShadowPushConstants {
    glm::mat4 lightViewProjection{ 1.0f };
};
} // namespace

glm::mat4 BuildDirectionalShadowViewProjection(
    const vesta::scene::SceneBounds& bounds,
    glm::vec4 lightDirectionAndIntensity)
{
    const float radius = glm::max(bounds.radius, 1.0f);
    glm::vec3 lightDirection = glm::vec3(lightDirectionAndIntensity);
    if (glm::dot(lightDirection, lightDirection) <= 1.0e-6f) {
        lightDirection = glm::vec3(-0.4f, -1.0f, -0.3f);
    }
    lightDirection = glm::normalize(lightDirection);
    const glm::vec3 lightPosition = bounds.center - lightDirection * radius * 2.5f;
    const glm::vec3 up = std::abs(glm::dot(lightDirection, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.92f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 lightView = glm::lookAt(lightPosition, bounds.center, up);
    glm::mat4 lightProjection = glm::ortho(-radius, radius, -radius, radius, 0.05f, radius * 5.5f);
    lightProjection[1][1] *= -1.0f;
    return lightProjection * lightView;
}

void ShadowMapPass::SetOutput(GraphTextureHandle shadowMap)
{
    _shadowMap = shadowMap;
}

void ShadowMapPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void ShadowMapPass::SetLight(glm::vec4 lightDirectionAndIntensity)
{
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
}

void ShadowMapPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/shadow_depth.vert.spv"));
    _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/shadow_depth.frag.spv"));

    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(ShadowPushConstants),
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

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.depthFormat = VK_FORMAT_D32_SFLOAT;
    pipelineDesc.vertexShader = _vertexShader;
    pipelineDesc.fragmentShader = _fragmentShader;
    pipelineDesc.cullMode = VK_CULL_MODE_BACK_BIT;
    pipelineDesc.depthTestEnable = true;
    pipelineDesc.depthWriteEnable = true;
    pipelineDesc.vertexBindings = { binding };
    pipelineDesc.vertexAttributes = { attributes.begin(), attributes.end() };

    _pipeline = vkutil::create_graphics_pipeline(vkDevice, pipelineDesc);
}

void ShadowMapPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_shadowMap, ResourceUsage::DepthAttachmentWrite);
}

void ShadowMapPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _scene == nullptr || !_scene->HasRasterGeometry()) {
        return;
    }

    ShadowPushConstants pushConstants{
        .lightViewProjection = BuildDirectionalShadowViewProjection(_scene->GetBounds(), _lightDirectionAndIntensity),
    };

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = context.GetTextureView(_shadowMap);
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue = depthClear;

    const VkExtent3D shadowExtent = context.GetTextureExtent(_shadowMap);
    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, VkExtent2D{ shadowExtent.width, shadowExtent.height } };
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = &depthAttachment;

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(shadowExtent.width);
    viewport.height = static_cast<float>(shadowExtent.height);
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = VkExtent2D{ shadowExtent.width, shadowExtent.height };

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);
    vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants), &pushConstants);

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

void ShadowMapPass::Shutdown(RenderDevice& device)
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

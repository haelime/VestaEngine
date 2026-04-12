#include <vesta/render/passes/gaussian_splat_pass.h>

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
// params.x = point size, params.y = opacity. Keeping UI-driven tuning values in
// push constants makes iteration cheap without rebuilding descriptor sets.
struct GaussianPushConstants {
    glm::mat4 viewProjection{ 1.0f };
    glm::vec4 params{ 6.0f, 0.35f, 0.0f, 0.0f };
};
} // namespace

void GaussianSplatPass::SetDepthInput(GraphTextureHandle depth)
{
    _depthInput = depth;
}

void GaussianSplatPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void GaussianSplatPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void GaussianSplatPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void GaussianSplatPass::SetParams(float pointSize, float opacity, bool enabled)
{
    _pointSize = pointSize;
    _opacity = opacity;
    _enabled = enabled;
}

void GaussianSplatPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/gaussian.vert.spv"));
    _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/gaussian.frag.spv"));

    const std::array<VkDescriptorSetLayout, 0> descriptorSetLayouts{};
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(GaussianPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(vesta::scene::SceneVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Gaussian data reuses SceneVertex but reads only position, color, and
    // splat parameters. Mesh scenes leave splatParams at sane defaults.
    std::array<VkVertexInputAttributeDescription, 3> attributes{};
    attributes[0] = VkVertexInputAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, position),
    };
    attributes[1] = VkVertexInputAttributeDescription{
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, color),
    };
    attributes[2] = VkVertexInputAttributeDescription{
        .location = 3,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, splatParams),
    };

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
    pipelineDesc.depthFormat = VK_FORMAT_D32_SFLOAT;
    pipelineDesc.vertexShader = _vertexShader;
    pipelineDesc.fragmentShader = _fragmentShader;
    pipelineDesc.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    pipelineDesc.cullMode = VK_CULL_MODE_NONE;
    pipelineDesc.depthTestEnable = true;
    pipelineDesc.depthWriteEnable = false;
    pipelineDesc.blendingEnable = true;
    pipelineDesc.vertexBindings = { binding };
    pipelineDesc.vertexAttributes = { attributes.begin(), attributes.end() };

    _pipeline = vkutil::create_graphics_pipeline(vkDevice, pipelineDesc);
}

void GaussianSplatPass::Setup(RenderGraphBuilder& builder)
{
    if (_depthInput) {
        builder.Read(_depthInput, ResourceUsage::DepthRead);
    }
    builder.Write(_output, ResourceUsage::ColorAttachmentWrite);
}

void GaussianSplatPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE) {
        return;
    }

    VkClearValue colorClear{};
    colorClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = context.GetTextureView(_output);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = colorClear;

    VkRenderingAttachmentInfo depthAttachment{};
    if (_depthInput) {
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = context.GetTextureView(_depthInput);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = _depthInput ? &depthAttachment : nullptr;

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
    if (_enabled && _scene != nullptr && _camera != nullptr && _scene->HasGaussianSplats()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);

        GaussianPushConstants pushConstants{
            .viewProjection = _camera->GetViewProjection(),
            .params = glm::vec4(_pointSize, _opacity, 0.0f, 0.0f),
        };
        vkCmdPushConstants(commandBuffer,
            _pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(GaussianPushConstants),
            &pushConstants);

        const VkBuffer vertexBuffer = context.GetDevice().GetBuffer(_scene->GetVertexBuffer());
        constexpr VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(_scene->GetVertices().size()), 1, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);
}

void GaussianSplatPass::Shutdown(RenderDevice& device)
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

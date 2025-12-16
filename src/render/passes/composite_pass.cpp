#include <vesta/render/passes/composite_pass.h>

#include <array>
#include <algorithm>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
constexpr uint32_t kInvalidImageIndex = kInvalidResourceIndex;

// mode selects which intermediate result to visualize. params holds gaussian
// blend strength, exposure EV, and near/far planes for linear depth debug.
struct CompositePushConstants {
    glm::uvec4 imageIndices0{ kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex };
    glm::uvec4 imageIndices1{ 0u, 0u, 0u, 0u };
    glm::uvec4 imageIndices2{ kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex };
    glm::uvec4 imageIndices3{ kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex };
    glm::uvec4 imageIndices4{ kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex };
    glm::uvec4 gaussianDebug{ kInvalidImageIndex, 0u, 0u, 8u };
    glm::vec4 params{ 0.25f, 0.0f, 0.0f, 0.0f };
    glm::vec4 compareParams{ 0.0f, 0.5f, 4.0f, 0.0f };
    glm::vec4 postParams{ 1.0f, 1.0f, 0.0f, 0.0f };
    glm::vec4 ssaoParams{ 1.0f, 0.75f, 1.35f, 0.0f };
    glm::mat4 inverseViewProjection{ 1.0f };
};
} // namespace

void CompositePass::SetInputs(
    GraphTextureHandle deferredLighting,
    GraphTextureHandle pathTrace,
    GraphTextureHandle gaussianAccum,
    GraphTextureHandle gaussianReveal,
    GraphTextureHandle gaussianDebug)
{
    _deferredLighting = deferredLighting;
    _pathTrace = pathTrace;
    _gaussianAccum = gaussianAccum;
    _gaussianReveal = gaussianReveal;
    _gaussianDebug = gaussianDebug;
}

void CompositePass::SetGBufferInputs(
    GraphTextureHandle albedo,
    GraphTextureHandle normalRoughness,
    GraphTextureHandle material,
    GraphTextureHandle debug,
    GraphTextureHandle motion,
    GraphTextureHandle lightingDebug,
    GraphTextureHandle depth)
{
    _gbufferAlbedo = albedo;
    _gbufferNormalRoughness = normalRoughness;
    _gbufferMaterial = material;
    _gbufferDebug = debug;
    _gbufferMotion = motion;
    _lightingDebug = lightingDebug;
    _gbufferDepth = depth;
}

void CompositePass::SetGaussianDebugResources(uint32_t tileRangeBufferIndex, uint32_t tileCountX, uint32_t tileCountY)
{
    _gaussianTileRangeBufferIndex = tileRangeBufferIndex;
    _gaussianTileCountX = tileCountX;
    _gaussianTileCountY = tileCountY;
}

void CompositePass::SetShadowMap(GraphTextureHandle shadowMap)
{
    _shadowMap = shadowMap;
}

void CompositePass::SetOverdraw(GraphTextureHandle overdraw)
{
    _overdraw = overdraw;
}

void CompositePass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void CompositePass::SetMode(uint32_t mode, float gaussianMix, uint32_t debugView, uint32_t gaussianDebugView)
{
    _mode = mode;
    _gaussianMix = gaussianMix;
    _debugView = debugView;
    _gaussianDebugView = gaussianDebugView;
}

void CompositePass::SetCompare(uint32_t compareMode, float splitPosition, float differenceScale)
{
    _compareMode = compareMode;
    _compareSplitPosition = glm::clamp(splitPosition, 0.02f, 0.98f);
    _compareDifferenceScale = glm::max(differenceScale, 0.1f);
}

void CompositePass::SetExposure(float exposureEv)
{
    _exposureEv = exposureEv;
}

void CompositePass::SetToneMapping(uint32_t toneMappingMode)
{
    _toneMappingMode = toneMappingMode;
}

void CompositePass::SetPostProcess(float saturation, float contrast, bool vignetteEnabled, float vignetteStrength)
{
    _saturation = std::clamp(saturation, 0.0f, 2.0f);
    _contrast = std::clamp(contrast, 0.25f, 2.0f);
    _vignetteEnabled = vignetteEnabled;
    _vignetteStrength = std::clamp(vignetteStrength, 0.0f, 1.0f);
}

void CompositePass::SetAmbientOcclusion(bool enabled, float radius, float intensity)
{
    _ssaoParams = glm::vec4(enabled ? 1.0f : 0.0f, std::max(radius, 0.01f), std::clamp(intensity, 0.0f, 4.0f), 0.0f);
}

void CompositePass::SetCameraMatrices(const glm::mat4& viewProjection, const glm::mat4& inverseViewProjection)
{
    _viewProjection = viewProjection;
    _inverseViewProjection = inverseViewProjection;
}

void CompositePass::SetDepthRange(float nearPlane, float farPlane)
{
    _nearPlane = nearPlane;
    _farPlane = farPlane;
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
    if (_deferredLighting) {
        builder.Read(_deferredLighting, ResourceUsage::StorageRead);
    }
    if (_pathTrace) {
        builder.Read(_pathTrace, ResourceUsage::StorageRead);
    }
    if (_gaussianAccum) {
        builder.Read(_gaussianAccum, ResourceUsage::StorageRead);
    }
    if (_gaussianReveal) {
        builder.Read(_gaussianReveal, ResourceUsage::StorageRead);
    }
    if (_gaussianDebug) {
        builder.Read(_gaussianDebug, ResourceUsage::StorageRead);
    }
    if (_gbufferAlbedo) {
        builder.Read(_gbufferAlbedo, ResourceUsage::StorageRead);
    }
    if (_gbufferNormalRoughness) {
        builder.Read(_gbufferNormalRoughness, ResourceUsage::StorageRead);
    }
    if (_gbufferMaterial) {
        builder.Read(_gbufferMaterial, ResourceUsage::StorageRead);
    }
    if (_gbufferDebug) {
        builder.Read(_gbufferDebug, ResourceUsage::StorageRead);
    }
    if (_gbufferMotion) {
        builder.Read(_gbufferMotion, ResourceUsage::StorageRead);
    }
    if (_lightingDebug) {
        builder.Read(_lightingDebug, ResourceUsage::StorageRead);
    }
    if (_gbufferDepth) {
        builder.Read(_gbufferDepth, ResourceUsage::SampledRead);
    }
    if (_shadowMap) {
        builder.Read(_shadowMap, ResourceUsage::SampledRead);
    }
    if (_overdraw) {
        builder.Read(_overdraw, ResourceUsage::StorageRead);
    }
    builder.Write(_output, ResourceUsage::ColorAttachmentWrite);
}

void CompositePass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE) {
        return;
    }

    CompositePushConstants pushConstants{
        .imageIndices0 = glm::uvec4(kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex),
        .imageIndices1 = glm::uvec4(_mode, _debugView, _gaussianDebugView, kInvalidImageIndex),
        .imageIndices2 = glm::uvec4(kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex),
        .imageIndices3 = glm::uvec4(kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex),
        .imageIndices4 = glm::uvec4(kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex, kInvalidImageIndex),
        .gaussianDebug = glm::uvec4(_gaussianTileRangeBufferIndex, _gaussianTileCountX, _gaussianTileCountY, 8u),
        .params = glm::vec4(_gaussianMix, _exposureEv, _nearPlane, _farPlane),
        .compareParams = glm::vec4(static_cast<float>(_compareMode), _compareSplitPosition, _compareDifferenceScale, static_cast<float>(_toneMappingMode)),
        .postParams = glm::vec4(_saturation, _contrast, _vignetteEnabled ? 1.0f : 0.0f, _vignetteStrength),
        .ssaoParams = _ssaoParams,
        .inverseViewProjection = _inverseViewProjection,
    };

    if (_deferredLighting) {
        const ImageHandle deferredHandle = context.GetTextureHandle(_deferredLighting);
        pushConstants.imageIndices0.x = context.GetDevice().GetImageResource(deferredHandle).bindless.storageImage;
    }
    if (_pathTrace) {
        const ImageHandle pathTraceHandle = context.GetTextureHandle(_pathTrace);
        pushConstants.imageIndices0.y = context.GetDevice().GetImageResource(pathTraceHandle).bindless.storageImage;
    }
    if (_gaussianAccum) {
        const ImageHandle gaussianAccumHandle = context.GetTextureHandle(_gaussianAccum);
        pushConstants.imageIndices0.z = context.GetDevice().GetImageResource(gaussianAccumHandle).bindless.storageImage;
    }
    if (_gaussianReveal) {
        const ImageHandle gaussianRevealHandle = context.GetTextureHandle(_gaussianReveal);
        pushConstants.imageIndices0.w = context.GetDevice().GetImageResource(gaussianRevealHandle).bindless.storageImage;
    }
    if (_gaussianDebug) {
        const ImageHandle gaussianDebugHandle = context.GetTextureHandle(_gaussianDebug);
        pushConstants.imageIndices1.w = context.GetDevice().GetImageResource(gaussianDebugHandle).bindless.storageImage;
    }
    if (_gbufferAlbedo) {
        const ImageHandle handle = context.GetTextureHandle(_gbufferAlbedo);
        pushConstants.imageIndices2.x = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }
    if (_gbufferNormalRoughness) {
        const ImageHandle handle = context.GetTextureHandle(_gbufferNormalRoughness);
        pushConstants.imageIndices2.y = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }
    if (_gbufferMaterial) {
        const ImageHandle handle = context.GetTextureHandle(_gbufferMaterial);
        pushConstants.imageIndices2.z = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }
    if (_gbufferDebug) {
        const ImageHandle handle = context.GetTextureHandle(_gbufferDebug);
        pushConstants.imageIndices3.x = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }
    if (_gbufferMotion) {
        const ImageHandle handle = context.GetTextureHandle(_gbufferMotion);
        pushConstants.imageIndices3.y = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }
    if (_lightingDebug) {
        const ImageHandle handle = context.GetTextureHandle(_lightingDebug);
        pushConstants.imageIndices3.z = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }
    if (_gbufferDepth) {
        const ImageHandle handle = context.GetTextureHandle(_gbufferDepth);
        pushConstants.imageIndices2.w = context.GetDevice().GetImageResource(handle).bindless.sampledImage;
    }
    if (_shadowMap) {
        const ImageHandle handle = context.GetTextureHandle(_shadowMap);
        pushConstants.imageIndices4.x = context.GetDevice().GetImageResource(handle).bindless.sampledImage;
    }
    if (_overdraw) {
        const ImageHandle handle = context.GetTextureHandle(_overdraw);
        pushConstants.imageIndices4.y = context.GetDevice().GetImageResource(handle).bindless.storageImage;
    }

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

#include <vesta/render/passes/temporal_aa_pass.h>

#include <algorithm>
#include <array>

#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/render/renderer.h>

namespace vesta::render {
namespace {
struct TemporalAAPushConstants {
    uint32_t inputImageIndex{ 0 };
    uint32_t outputImageIndex{ 0 };
    uint32_t normalImageIndex{ 0 };
    uint32_t materialImageIndex{ 0 };
    uint32_t depthImageIndex{ 0 };
    uint32_t motionImageIndex{ 0 };
    uint32_t reactiveImageIndex{ kInvalidResourceIndex };
    uint32_t historyImageIndex{ 0 };
    uint32_t frameIndex{ 0 };
    float feedback{ 0.88f };
    uint32_t enabled{ 1 };
    uint32_t debugView{ 0 };
    float sharpness{ 0.0f };
    uint32_t materialReactiveMask{ 1 };
    float reactiveMaskStrength{ 0.65f };
    float reactiveMetallicThreshold{ 0.55f };
    float reactiveEmissiveThreshold{ 0.08f };
    uint32_t reserved2{ 0 };
    uint32_t reserved3{ 0 };
    uint32_t reserved4{ 0 };
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::mat4 previousViewProjection{ 1.0f };
};
} // namespace

void TemporalAAPass::SetInputs(GraphTextureHandle input,
    GraphTextureHandle normalRoughness,
    GraphTextureHandle material,
    GraphTextureHandle motion,
    GraphTextureHandle reactive,
    GraphTextureHandle depth)
{
    _input = input;
    _normalRoughness = normalRoughness;
    _material = material;
    _motion = motion;
    _reactive = reactive;
    _depth = depth;
}

void TemporalAAPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void TemporalAAPass::SetEnabled(bool enabled)
{
    _enabled = enabled;
}

void TemporalAAPass::SetFeedback(float feedback)
{
    _feedback = std::clamp(feedback, 0.0f, 0.98f);
}

void TemporalAAPass::SetUpscalerSharpness(float sharpness)
{
    _upscalerSharpness = std::clamp(sharpness, 0.0f, 1.0f);
}

void TemporalAAPass::SetReactiveMask(bool materialReactiveMask, float strength)
{
    _materialReactiveMask = materialReactiveMask;
    _reactiveMaskStrength = std::clamp(strength, 0.0f, 1.0f);
}

void TemporalAAPass::SetFrameIndex(uint32_t frameIndex)
{
    _frameIndex = frameIndex;
}

void TemporalAAPass::SetCameraMatrices(const glm::mat4& viewProjection, const glm::mat4& inverseViewProjection)
{
    _viewProjection = viewProjection;
    _inverseViewProjection = inverseViewProjection;
}

void TemporalAAPass::SetDebugView(RendererDebugView debugView)
{
    _debugView = debugView;
}

void TemporalAAPass::EnsureHistoryImage(RenderDevice& device, VkExtent3D extent)
{
    if (_historyImage && _historyExtent.width == extent.width && _historyExtent.height == extent.height
        && _historyExtent.depth == extent.depth) {
        return;
    }

    DestroyHistoryImage(device);
    _historyImage = device.CreateImage(ImageDesc{
        .extent = extent,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT,
        .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        .registerBindlessStorage = true,
        .debugName = "TemporalAA.History",
    });
    _historyExtent = extent;
    _historyInitialized = false;
}

void TemporalAAPass::DestroyHistoryImage(RenderDevice& device)
{
    if (_historyImage) {
        device.WaitIdle();
        device.DestroyImage(_historyImage);
        _historyImage = {};
    }
    _historyExtent = {};
    _historyInitialized = false;
}

void TemporalAAPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/temporal_aa.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(TemporalAAPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
}

void TemporalAAPass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_input, ResourceUsage::StorageRead);
    builder.Read(_normalRoughness, ResourceUsage::StorageRead);
    builder.Read(_material, ResourceUsage::StorageRead);
    builder.Read(_motion, ResourceUsage::StorageRead);
    if (_reactive) {
        builder.Read(_reactive, ResourceUsage::StorageRead);
    }
    builder.Read(_depth, ResourceUsage::SampledRead);
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void TemporalAAPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || !_input || !_output || !_normalRoughness || !_material || !_motion || !_depth) {
        return;
    }

    const ImageHandle inputHandle = context.GetTextureHandle(_input);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    const ImageHandle normalHandle = context.GetTextureHandle(_normalRoughness);
    const ImageHandle materialHandle = context.GetTextureHandle(_material);
    const ImageHandle motionHandle = context.GetTextureHandle(_motion);
    const ImageHandle reactiveHandle = _reactive ? context.GetTextureHandle(_reactive) : ImageHandle{};
    const ImageHandle depthHandle = context.GetTextureHandle(_depth);
    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    EnsureHistoryImage(context.GetDevice(), outputExtent);

    const VkImageSubresourceRange colorRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkutil::transition_image(context.GetCommandBuffer(),
        context.GetDevice().GetImage(_historyImage),
        _historyInitialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        _historyInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        _historyInitialized ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        colorRange);
    _historyInitialized = true;

    const TemporalAAPushConstants pushConstants{
        .inputImageIndex = context.GetDevice().GetImageResource(inputHandle).bindless.storageImage,
        .outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
        .normalImageIndex = context.GetDevice().GetImageResource(normalHandle).bindless.storageImage,
        .materialImageIndex = context.GetDevice().GetImageResource(materialHandle).bindless.storageImage,
        .depthImageIndex = context.GetDevice().GetImageResource(depthHandle).bindless.sampledImage,
        .motionImageIndex = context.GetDevice().GetImageResource(motionHandle).bindless.storageImage,
        .reactiveImageIndex = _reactive
            ? context.GetDevice().GetImageResource(reactiveHandle).bindless.storageImage
            : kInvalidResourceIndex,
        .historyImageIndex = context.GetDevice().GetImageResource(_historyImage).bindless.storageImage,
        .frameIndex = _frameIndex,
        .feedback = _feedback,
        .enabled = _enabled ? 1u : 0u,
        .debugView = static_cast<uint32_t>(_debugView),
        .sharpness = _upscalerSharpness,
        .materialReactiveMask = _materialReactiveMask ? 1u : 0u,
        .reactiveMaskStrength = _reactiveMaskStrength,
        .inverseViewProjection = _inverseViewProjection,
        .previousViewProjection = _hasPreviousViewProjection ? _previousViewProjection : _viewProjection,
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalAAPushConstants), &pushConstants);

    vkCmdDispatch(commandBuffer, (outputExtent.width + 7) / 8, (outputExtent.height + 7) / 8, 1);

    _previousViewProjection = _viewProjection;
    _hasPreviousViewProjection = true;
}

void TemporalAAPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
    DestroyHistoryImage(device);
}
} // namespace vesta::render

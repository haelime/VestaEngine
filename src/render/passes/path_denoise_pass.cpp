#include <vesta/render/passes/path_denoise_pass.h>

#include <array>
#include <algorithm>

#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
struct PathDenoisePushConstants {
    uint32_t inputImageIndex{ 0 };
    uint32_t outputImageIndex{ 0 };
    uint32_t normalGuideImageIndex{ 0 };
    uint32_t depthGuideImageIndex{ 0 };
    uint32_t historyImageIndex{ 0 };
    uint32_t frameIndex{ 0 };
    float strength{ 0.65f };
    float temporalBlend{ 0.88f };
    float normalSigma{ 28.0f };
    float depthSigma{ 220.0f };
    uint32_t iterations{ 3 };
    uint32_t reserved1{ 0 };
};
} // namespace

void PathDenoisePass::SetInput(GraphTextureHandle input)
{
    _input = input;
}

void PathDenoisePass::SetGuides(GraphTextureHandle normalGuide, GraphTextureHandle depthGuide)
{
    _normalGuide = normalGuide;
    _depthGuide = depthGuide;
}

void PathDenoisePass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void PathDenoisePass::SetStrength(float strength)
{
    _strength = std::clamp(strength, 0.0f, 1.0f);
}

void PathDenoisePass::SetTemporalBlend(float blend)
{
    _temporalBlend = std::clamp(blend, 0.0f, 0.98f);
}

void PathDenoisePass::SetIterations(uint32_t iterations)
{
    _iterations = std::clamp(iterations, 1u, 5u);
}

void PathDenoisePass::SetFrameIndex(uint32_t frameIndex)
{
    _frameIndex = frameIndex;
}

void PathDenoisePass::EnsureHistoryImage(RenderDevice& device, VkExtent3D extent)
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
        .debugName = "PathDenoise.History",
    });
    _historyExtent = extent;
    _historyInitialized = false;
}

void PathDenoisePass::DestroyHistoryImage(RenderDevice& device)
{
    if (_historyImage) {
        device.WaitIdle();
        device.DestroyImage(_historyImage);
        _historyImage = {};
    }
    _historyExtent = {};
    _historyInitialized = false;
}

void PathDenoisePass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/path_denoise.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PathDenoisePushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
}

void PathDenoisePass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_input, ResourceUsage::StorageRead);
    if (_normalGuide) {
        builder.Read(_normalGuide, ResourceUsage::StorageRead);
    }
    if (_depthGuide) {
        builder.Read(_depthGuide, ResourceUsage::StorageRead);
    }
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void PathDenoisePass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || !_input || !_output) {
        return;
    }

    const ImageHandle inputHandle = context.GetTextureHandle(_input);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    const ImageHandle normalGuideHandle = context.GetTextureHandle(_normalGuide);
    const ImageHandle depthGuideHandle = context.GetTextureHandle(_depthGuide);
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

    const PathDenoisePushConstants pushConstants{
        .inputImageIndex = context.GetDevice().GetImageResource(inputHandle).bindless.storageImage,
        .outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
        .normalGuideImageIndex = context.GetDevice().GetImageResource(normalGuideHandle).bindless.storageImage,
        .depthGuideImageIndex = context.GetDevice().GetImageResource(depthGuideHandle).bindless.storageImage,
        .historyImageIndex = context.GetDevice().GetImageResource(_historyImage).bindless.storageImage,
        .frameIndex = _frameIndex,
        .strength = _strength,
        .temporalBlend = _temporalBlend,
        .iterations = _iterations,
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, _pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PathDenoisePushConstants), &pushConstants);

    vkCmdDispatch(commandBuffer, (outputExtent.width + 7) / 8, (outputExtent.height + 7) / 8, 1);
}

void PathDenoisePass::Shutdown(RenderDevice& device)
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

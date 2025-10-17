#include <vesta/render/passes/gaussian_splat_pass.h>

#include <array>

#include <vesta/render/vulkan/vk_pipelines.h>

namespace vesta::render {
namespace {
[[maybe_unused]] void example_gaussian_compute_boilerplate(VkDevice device)
{
    const std::array<VkDescriptorSetLayout, 0> descriptorSetLayouts{};
    const std::array<VkPushConstantRange, 0> pushConstants{};
    VkPipelineLayout layout = vkutil::create_pipeline_layout(device, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = layout;

    // The real pass will pair this with a loaded compute shader and dispatch sequence.
    (void)pipelineDesc;
    vkutil::destroy_pipeline_layout(device, layout);
}
} // namespace

void GaussianSplatPass::SetDepthInput(GraphTextureHandle depth)
{
    _depthInput = depth;
}

void GaussianSplatPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void GaussianSplatPass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_depthInput, ResourceUsage::DepthRead);
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void GaussianSplatPass::Execute(const RenderGraphContext& context)
{
    (void)context;
    // Skeleton only: radix sort, tile binning, and splat rasterization dispatches fan out from this pass chain.
}
} // namespace vesta::render

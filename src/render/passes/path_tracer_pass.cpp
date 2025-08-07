#include <vesta/render/passes/path_tracer_pass.h>

#include <array>

#include <vesta/render/vulkan/vk_descriptors.h>

namespace vesta::render {
namespace {
[[maybe_unused]] void example_descriptor_boilerplate(VkDevice device)
{
    const std::array<VkDescriptorSetLayoutBinding, 1> bindings{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const std::array<VkDescriptorBindingFlags, 1> bindingFlags{ 0 };
    VkDescriptorSetLayout layout = vkutil::create_descriptor_set_layout(device, bindings, bindingFlags);

    const std::array<VkDescriptorPoolSize, 1> poolSizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
    };
    VkDescriptorPool pool = vkutil::create_descriptor_pool(device, poolSizes, 1);
    VkDescriptorSet descriptorSet = vkutil::allocate_descriptor_set(device, pool, layout);

    // The real pass will write actual TLAS/output descriptors here.
    (void)descriptorSet;
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
}
} // namespace

void PathTracerPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void PathTracerPass::SetDepthInput(GraphTextureHandle depth)
{
    _depthInput = depth;
}

void PathTracerPass::Setup(RenderGraphBuilder& builder)
{
    if (_depthInput) {
        builder.Read(_depthInput, ResourceUsage::DepthRead);
    }
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void PathTracerPass::Execute(const RenderGraphContext& context)
{
    (void)context;
    // Skeleton only: a real implementation would build BLAS/TLAS, bind an SBT, and call vkCmdTraceRaysKHR here.
}
} // namespace vesta::render

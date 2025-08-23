#pragma once

#include <vector>

#include <vesta/render/vulkan/vk_types.h>

namespace vkutil {
struct PipelineShaderStages {
    VkShaderModule vertexShader{ VK_NULL_HANDLE };
    VkShaderModule fragmentShader{ VK_NULL_HANDLE };
    VkShaderModule computeShader{ VK_NULL_HANDLE };
};

// GraphicsPipelineDesc mirrors the major decisions that shape a graphics
// pipeline: render target formats, fixed function state, and shader stages.
struct GraphicsPipelineDesc {
    VkPipelineLayout layout{ VK_NULL_HANDLE };
    std::vector<VkFormat> colorFormats;
    std::optional<VkFormat> depthFormat{};
    VkShaderModule vertexShader{ VK_NULL_HANDLE };
    VkShaderModule fragmentShader{ VK_NULL_HANDLE };
    VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPolygonMode polygonMode{ VK_POLYGON_MODE_FILL };
    VkCullModeFlags cullMode{ VK_CULL_MODE_BACK_BIT };
    VkFrontFace frontFace{ VK_FRONT_FACE_COUNTER_CLOCKWISE };
    bool depthTestEnable{ false };
    bool depthWriteEnable{ false };
    bool blendingEnable{ false };
    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments;
    std::vector<VkVertexInputBindingDescription> vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;
};

struct ComputePipelineDesc {
    VkPipelineLayout layout{ VK_NULL_HANDLE };
    VkShaderModule computeShader{ VK_NULL_HANDLE };
};

[[nodiscard]] VkPipelineLayout create_pipeline_layout(
    VkDevice device,
    std::span<const VkDescriptorSetLayout> descriptorSetLayouts,
    std::span<const VkPushConstantRange> pushConstantRanges = {});

[[nodiscard]] VkPipeline create_graphics_pipeline(VkDevice device, const GraphicsPipelineDesc& desc);
[[nodiscard]] VkPipeline create_compute_pipeline(VkDevice device, const ComputePipelineDesc& desc);

void destroy_pipeline(VkDevice device, VkPipeline& pipeline);
void destroy_pipeline_layout(VkDevice device, VkPipelineLayout& layout);
} // namespace vkutil

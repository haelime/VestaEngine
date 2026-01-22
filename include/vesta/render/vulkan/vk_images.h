#pragma once

#include <vesta/render/vulkan/vk_types.h>

namespace vkutil {
// Image helpers centralize common layout transitions and view creation patterns.
[[nodiscard]] VkImageSubresourceRange make_image_subresource_range(
    VkImageAspectFlags aspectMask,
    uint32_t mipLevels = 1,
    uint32_t layers = 1);

void transition_image(VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask,
    VkImageSubresourceRange subresourceRange);

void copy_image_to_image(VkCommandBuffer commandBuffer,
    VkImage srcImage,
    VkImage dstImage,
    VkExtent2D srcExtent,
    VkExtent2D dstExtent);

[[nodiscard]] VkImageView create_image_view(
    VkDevice device,
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    uint32_t mipLevels = 1,
    uint32_t layers = 1);
} // namespace vkutil

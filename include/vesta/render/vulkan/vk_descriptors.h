#pragma once

#include <vesta/render/vulkan/vk_types.h>

namespace vkutil {
// Thin wrappers around Vulkan descriptor boilerplate. They keep the sample code
// readable without hiding the descriptor concepts themselves.
[[nodiscard]] VkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    std::span<const VkDescriptorSetLayoutBinding> bindings,
    std::span<const VkDescriptorBindingFlags> bindingFlags = {});

[[nodiscard]] VkDescriptorPool create_descriptor_pool(
    VkDevice device,
    std::span<const VkDescriptorPoolSize> poolSizes,
    uint32_t maxSets,
    VkDescriptorPoolCreateFlags flags = 0);

[[nodiscard]] VkDescriptorSet allocate_descriptor_set(
    VkDevice device,
    VkDescriptorPool descriptorPool,
    VkDescriptorSetLayout layout);

void update_descriptor_set(VkDevice device, std::span<const VkWriteDescriptorSet> writes);

[[nodiscard]] VkWriteDescriptorSet write_sampled_image(
    VkDescriptorSet dstSet,
    VkDescriptorImageInfo* imageInfo,
    uint32_t binding,
    uint32_t arrayElement = 0);

[[nodiscard]] VkWriteDescriptorSet write_storage_image(
    VkDescriptorSet dstSet,
    VkDescriptorImageInfo* imageInfo,
    uint32_t binding,
    uint32_t arrayElement = 0);

[[nodiscard]] VkWriteDescriptorSet write_storage_buffer(
    VkDescriptorSet dstSet,
    VkDescriptorBufferInfo* bufferInfo,
    uint32_t binding,
    uint32_t arrayElement = 0);

[[nodiscard]] VkWriteDescriptorSet write_uniform_buffer(
    VkDescriptorSet dstSet,
    VkDescriptorBufferInfo* bufferInfo,
    uint32_t binding,
    uint32_t arrayElement = 0);
} // namespace vkutil

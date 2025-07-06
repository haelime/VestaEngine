#include <vesta/render/vulkan/vk_descriptors.h>

VkDescriptorSetLayout vkutil::create_descriptor_set_layout(VkDevice device,
    std::span<const VkDescriptorSetLayoutBinding> bindings,
    std::span<const VkDescriptorBindingFlags> bindingFlags)
{
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    if (!bindingFlags.empty()) {
        bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = bindingFlags.empty() ? nullptr : &bindingFlagsInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout));
    return layout;
}

VkDescriptorPool vkutil::create_descriptor_pool(
    VkDevice device, std::span<const VkDescriptorPoolSize> poolSizes, uint32_t maxSets, VkDescriptorPoolCreateFlags flags)
{
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = flags;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));
    return pool;
}

VkDescriptorSet vkutil::allocate_descriptor_set(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    return descriptorSet;
}

void vkutil::update_descriptor_set(VkDevice device, std::span<const VkWriteDescriptorSet> writes)
{
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

VkWriteDescriptorSet vkutil::write_sampled_image(
    VkDescriptorSet dstSet, VkDescriptorImageInfo* imageInfo, uint32_t binding, uint32_t arrayElement)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = dstSet;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = imageInfo;
    return write;
}

VkWriteDescriptorSet vkutil::write_storage_image(
    VkDescriptorSet dstSet, VkDescriptorImageInfo* imageInfo, uint32_t binding, uint32_t arrayElement)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = dstSet;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = imageInfo;
    return write;
}

VkWriteDescriptorSet vkutil::write_storage_buffer(
    VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo, uint32_t binding, uint32_t arrayElement)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = dstSet;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = bufferInfo;
    return write;
}

VkWriteDescriptorSet vkutil::write_uniform_buffer(
    VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo, uint32_t binding, uint32_t arrayElement)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = dstSet;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = bufferInfo;
    return write;
}

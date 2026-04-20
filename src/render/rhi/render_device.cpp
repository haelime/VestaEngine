#include <vesta/render/rhi/render_device.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>

#include <SDL_vulkan.h>

#include <fmt/format.h>

#include <vesta/core/debug.h>
#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_images.h>

#include "VkBootstrap.h"

namespace vesta::render {
namespace {
constexpr uint32_t kRequiredVulkanMajor = 1;
constexpr uint32_t kRequiredVulkanMinor = 3;
constexpr uint32_t kRequiredVulkanPatch = 0;
constexpr VkDeviceSize kDefaultUploadStagingCapacity = 32ull * 1024ull * 1024ull;
constexpr VmaAllocationCreateFlags kHostUploadFlags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

uint32_t CalculateDedicatedVideoMemoryMiB(VkPhysicalDevice physicalDevice)
{
    // On most discrete GPUs the largest DEVICE_LOCAL heap is a good proxy for
    // dedicated VRAM. It is not perfect, but it is enough for preset selection.
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    VkDeviceSize dedicatedBytes = 0;
    for (uint32_t heapIndex = 0; heapIndex < memoryProperties.memoryHeapCount; ++heapIndex) {
        const VkMemoryHeap& heap = memoryProperties.memoryHeaps[heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            dedicatedBytes = std::max(dedicatedBytes, heap.size);
        }
    }

    return static_cast<uint32_t>(dedicatedBytes / (1024ull * 1024ull));
}

VkDescriptorSetLayoutBinding make_bindless_binding(uint32_t binding, VkDescriptorType type, uint32_t descriptorCount)
{
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = type;
    layoutBinding.descriptorCount = descriptorCount;
    layoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
    return layoutBinding;
}
} // namespace

void BindlessDescriptorManager::Initialize(VkDevice device, VkSampler defaultSampler)
{
    // This project uses one large descriptor set instead of binding a new set
    // for every pass/resource pair. That keeps sample shaders short.
    _defaultSampler = defaultSampler;

    const std::array<VkDescriptorPoolSize, 3> poolSizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxSampledImages },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxStorageImages },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxStorageBuffers },
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &_pool));

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{
        make_bindless_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxSampledImages),
        make_bindless_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxStorageImages),
        make_bindless_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxStorageBuffers),
    };
    std::array<VkDescriptorBindingFlags, 3> bindingFlags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &_layout));

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_layout;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &_set));
}

void BindlessDescriptorManager::Shutdown(VkDevice device)
{
    if (_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, _layout, nullptr);
        _layout = VK_NULL_HANDLE;
    }
    if (_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, _pool, nullptr);
        _pool = VK_NULL_HANDLE;
    }

    _set = VK_NULL_HANDLE;
    _defaultSampler = VK_NULL_HANDLE;
    _nextSampledImage = 0;
    _nextStorageImage = 0;
    _nextStorageBuffer = 0;
}

uint32_t BindlessDescriptorManager::RegisterSampledImage(VkDevice device, VkImageView view, VkImageLayout layout)
{
    if (_nextSampledImage >= kMaxSampledImages) {
        fmt::println("Bindless sampled image heap overflow: {} / {}", _nextSampledImage, kMaxSampledImages);
        std::abort();
    }

    const uint32_t slot = _nextSampledImage++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = _defaultSampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = _set;
    write.dstBinding = 0;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return slot;
}

uint32_t BindlessDescriptorManager::RegisterStorageImage(VkDevice device, VkImageView view, VkImageLayout layout)
{
    if (_nextStorageImage >= kMaxStorageImages) {
        fmt::println("Bindless storage image heap overflow: {} / {}", _nextStorageImage, kMaxStorageImages);
        std::abort();
    }

    const uint32_t slot = _nextStorageImage++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = _set;
    write.dstBinding = 1;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return slot;
}

uint32_t BindlessDescriptorManager::RegisterStorageBuffer(VkDevice device, VkBuffer buffer, VkDeviceSize range)
{
    if (_nextStorageBuffer >= kMaxStorageBuffers) {
        fmt::println("Bindless storage buffer heap overflow: {} / {}", _nextStorageBuffer, kMaxStorageBuffers);
        std::abort();
    }

    const uint32_t slot = _nextStorageBuffer++;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = _set;
    write.dstBinding = 2;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return slot;
}

bool RenderDevice::Initialize(SDL_Window* window, const RenderDeviceDesc& desc)
{
    _window = window;

    // Vulkan instance/device creation happens before the allocator because VMA
    // needs both handles to manage memory allocations for buffers and images.
    CreateInstanceAndDevice(desc);
    CreateAllocator();
    InitializeImmediateContext();
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSampler));
    _bindless.Initialize(_device, _defaultSampler);
    CreateSwapchain(desc.swapchainExtent);

    return true;
}

void RenderDevice::Shutdown()
{
    WaitIdle();
    DestroySwapchain();
    FlushUploadBatch();
    CleanupResourceStorage();
    ShutdownImmediateContext();
    _bindless.Shutdown(_device);

    if (_defaultSampler != VK_NULL_HANDLE) {
        vkDestroySampler(_device, _defaultSampler, nullptr);
        _defaultSampler = VK_NULL_HANDLE;
    }

    if (_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(_allocator);
        _allocator = VK_NULL_HANDLE;
    }

    if (_device != VK_NULL_HANDLE) {
        vkDestroyDevice(_device, nullptr);
        _device = VK_NULL_HANDLE;
    }

    if (_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        _surface = VK_NULL_HANDLE;
    }

    if (_debugMessenger != VK_NULL_HANDLE) {
        vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
        _debugMessenger = VK_NULL_HANDLE;
    }

    if (_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(_instance, nullptr);
        _instance = VK_NULL_HANDLE;
    }

    _window = nullptr;
}

void RenderDevice::WaitIdle() const
{
    if (_device != VK_NULL_HANDLE) {
        VK_CHECK(vkDeviceWaitIdle(_device));
    }
}

void RenderDevice::RecreateSwapchain(VkExtent2D extent)
{
    WaitIdle();
    DestroySwapchain();
    CreateSwapchain(extent);
}

BufferHandle RenderDevice::CreateBuffer(const BufferDesc& desc)
{
    BufferHandle handle = AllocateBufferSlot();
    AllocatedBuffer& buffer = _buffers[handle.index];
    buffer.desc = desc;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = desc.size;
    bufferInfo.usage = desc.usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = desc.memoryUsage;
    allocInfo.flags = desc.allocationFlags;

    if (_transferQueue != VK_NULL_HANDLE && _transferQueueFamily != _graphicsQueueFamily) {
        const std::array<uint32_t, 2> queueFamilies{ _graphicsQueueFamily, _transferQueueFamily };
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size());
        bufferInfo.pQueueFamilyIndices = queueFamilies.data();
        VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation, &buffer.allocationInfo));
    } else {
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation, &buffer.allocationInfo));
    }

    if (desc.registerBindlessStorage) {
        buffer.bindless.storageBuffer = _bindless.RegisterStorageBuffer(_device, buffer.buffer, desc.size);
    }

    return handle;
}

ImageHandle RenderDevice::CreateImage(const ImageDesc& desc)
{
    ImageHandle handle = AllocateImageSlot();
    AllocatedImage& image = _images[handle.index];
    image.desc = desc;

    VkImageCreateInfo imageInfo = vkinit::image_create_info(desc.format, desc.usage, desc.extent);
    imageInfo.mipLevels = desc.mipLevels;
    imageInfo.arrayLayers = desc.arrayLayers;
    imageInfo.initialLayout = desc.initialLayout;
    std::array<uint32_t, 2> queueFamilies{ _graphicsQueueFamily, _transferQueueFamily };
    if (_transferQueue != VK_NULL_HANDLE && _transferQueueFamily != _graphicsQueueFamily) {
        imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        imageInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size());
        imageInfo.pQueueFamilyIndices = queueFamilies.data();
    }

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = desc.memoryUsage;

    VK_CHECK(vmaCreateImage(_allocator, &imageInfo, &allocInfo, &image.image, &image.allocation, &image.allocationInfo));

    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(desc.format, image.image, desc.aspectFlags);
    viewInfo.subresourceRange.levelCount = desc.mipLevels;
    viewInfo.subresourceRange.layerCount = desc.arrayLayers;
    VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &image.defaultView));

    // Registering in the bindless heap stores the slot index once so passes can
    // push only integers instead of full descriptor updates every frame.
    if (desc.registerBindlessSampled) {
        image.bindless.sampledImage =
            _bindless.RegisterSampledImage(_device, image.defaultView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (desc.registerBindlessStorage) {
        image.bindless.storageImage = _bindless.RegisterStorageImage(_device, image.defaultView, VK_IMAGE_LAYOUT_GENERAL);
    }

    return handle;
}

void RenderDevice::DestroyBuffer(BufferHandle handle)
{
    if (!handle) {
        return;
    }

    AllocatedBuffer& buffer = _buffers[handle.index];
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
        buffer = {};
        _freeBufferSlots.push_back(handle.index);
    }
}

void RenderDevice::DestroyImage(ImageHandle handle)
{
    if (!handle) {
        return;
    }

    AllocatedImage& image = _images[handle.index];
    if (image.defaultView != VK_NULL_HANDLE) {
        vkDestroyImageView(_device, image.defaultView, nullptr);
        image.defaultView = VK_NULL_HANDLE;
    }

    if (!image.ownedBySwapchain && image.image != VK_NULL_HANDLE) {
        vmaDestroyImage(_allocator, image.image, image.allocation);
    }

    if (image.image != VK_NULL_HANDLE || image.ownedBySwapchain) {
        image = {};
        _freeImageSlots.push_back(handle.index);
    }
}

void RenderDevice::InitializeImmediateContext()
{
    if (_device == VK_NULL_HANDLE) {
        return;
    }

    VkCommandPoolCreateInfo poolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
    VK_CHECK(vkCreateCommandPool(_device, &poolInfo, nullptr, &_immediateContext.commandPool));

    VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(_immediateContext.commandPool);
    VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo, &_immediateContext.commandBuffer));

    VkFenceCreateInfo fenceInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(_device, &fenceInfo, nullptr, &_immediateContext.fence));

    VkCommandPoolCreateInfo uploadPoolInfo = vkinit::command_pool_create_info(GetTransferQueueFamily());
    VK_CHECK(vkCreateCommandPool(_device, &uploadPoolInfo, nullptr, &_uploadContext.commandPool));

    VkCommandBufferAllocateInfo uploadAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext.commandPool);
    VK_CHECK(vkAllocateCommandBuffers(_device, &uploadAllocInfo, &_uploadContext.commandBuffer));

    VK_CHECK(vkCreateFence(_device, &fenceInfo, nullptr, &_uploadContext.fence));
    EnsureUploadCapacity(kDefaultUploadStagingCapacity);
}

void RenderDevice::ShutdownImmediateContext()
{
    if (_device == VK_NULL_HANDLE) {
        return;
    }

    FlushUploadBatch();
    if (_uploadContext.stagingBuffer) {
        DestroyBuffer(_uploadContext.stagingBuffer);
        _uploadContext.stagingBuffer = {};
    }
    _uploadContext.mappedData = nullptr;
    _uploadContext.capacity = 0;
    _uploadContext.offset = 0;
    _uploadContext.recording = false;
    _uploadContext.pendingCopies = 0;
    _uploadBatchStats = {};

    if (_uploadContext.fence != VK_NULL_HANDLE) {
        vkDestroyFence(_device, _uploadContext.fence, nullptr);
        _uploadContext.fence = VK_NULL_HANDLE;
    }
    if (_uploadContext.commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_device, _uploadContext.commandPool, nullptr);
        _uploadContext.commandPool = VK_NULL_HANDLE;
    }
    _uploadContext.commandBuffer = VK_NULL_HANDLE;

    if (_immediateContext.fence != VK_NULL_HANDLE) {
        vkDestroyFence(_device, _immediateContext.fence, nullptr);
        _immediateContext.fence = VK_NULL_HANDLE;
    }
    if (_immediateContext.commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_device, _immediateContext.commandPool, nullptr);
        _immediateContext.commandPool = VK_NULL_HANDLE;
    }
    _immediateContext.commandBuffer = VK_NULL_HANDLE;
}

void RenderDevice::EnsureUploadCapacity(VkDeviceSize requiredBytes)
{
    const VkDeviceSize targetCapacity = std::max(requiredBytes, kDefaultUploadStagingCapacity);
    if (_uploadContext.capacity >= targetCapacity && _uploadContext.stagingBuffer) {
        return;
    }

    FlushUploadBatch();
    if (_uploadContext.stagingBuffer) {
        DestroyBuffer(_uploadContext.stagingBuffer);
        _uploadContext.stagingBuffer = {};
    }

    _uploadContext.stagingBuffer = CreateBuffer(BufferDesc{
        .size = targetCapacity,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kHostUploadFlags,
        .debugName = "UploadSchedulerStaging",
    });
    const AllocatedBuffer& stagingBuffer = GetBufferResource(_uploadContext.stagingBuffer);
    _uploadContext.mappedData = stagingBuffer.allocationInfo.pMappedData;
    _uploadContext.capacity = targetCapacity;
    _uploadContext.offset = 0;
    _uploadBatchStats.stagingCapacity = targetCapacity;
}

void RenderDevice::BeginUploadBatchRecording()
{
    if (_uploadContext.recording || _device == VK_NULL_HANDLE) {
        return;
    }

    WaitForFenceOrAssert(_uploadContext.fence, "BeginUploadBatchRecording");
    VK_CHECK(vkResetFences(_device, 1, &_uploadContext.fence));
    VK_CHECK(vkResetCommandPool(_device, _uploadContext.commandPool, 0));

    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(_uploadContext.commandBuffer, &beginInfo));
    _uploadContext.recording = true;
}

void RenderDevice::SetDebugWaitContext(std::string_view context)
{
    _debugWaitContext.assign(context.begin(), context.end());
}

void RenderDevice::WaitForFenceOrAssert(VkFence fence, std::string_view waitLabel)
{
    if (_device == VK_NULL_HANDLE || fence == VK_NULL_HANDLE) {
        return;
    }

#if defined(NDEBUG)
    VK_CHECK(vkWaitForFences(_device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
#else
    constexpr uint64_t kDebugWaitTimeoutNs = 5ull * 1000ull * 1000ull * 1000ull;
    const VkResult waitResult = vkWaitForFences(_device, 1, &fence, VK_TRUE, kDebugWaitTimeoutNs);
    if (waitResult != VK_SUCCESS) {
        const std::string context = _debugWaitContext.empty() ? std::string("unspecified") : _debugWaitContext;
        VESTA_ASSERT(false,
            fmt::format(
                "Fence wait failed in {} (result={}) | context='{}' pendingBytes={} pendingCopies={} stagingCapacity={} queue={}",
                waitLabel,
                string_VkResult(waitResult),
                context,
                static_cast<unsigned long long>(_uploadBatchStats.pendingBytes),
                _uploadBatchStats.pendingCopies,
                static_cast<unsigned long long>(_uploadBatchStats.stagingCapacity),
                _transferQueue != VK_NULL_HANDLE ? "transfer" : "graphics"));
    }
#endif
}

void RenderDevice::ImmediateSubmit(const std::function<void(VkCommandBuffer)>& recorder)
{
    if (_device == VK_NULL_HANDLE || !recorder) {
        return;
    }

    FlushUploadBatch();
    WaitForFenceOrAssert(_immediateContext.fence, "ImmediateSubmit");
    VK_CHECK(vkResetFences(_device, 1, &_immediateContext.fence));
    VK_CHECK(vkResetCommandPool(_device, _immediateContext.commandPool, 0));

    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(_immediateContext.commandBuffer, &beginInfo));
    recorder(_immediateContext.commandBuffer);
    VK_CHECK(vkEndCommandBuffer(_immediateContext.commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo = vkinit::command_buffer_submit_info(_immediateContext.commandBuffer);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&commandBufferInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submitInfo, _immediateContext.fence));
    VK_CHECK(vkWaitForFences(_device, 1, &_immediateContext.fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
}

void RenderDevice::UploadBufferData(BufferHandle destination, VkDeviceSize destinationOffset, std::span<const std::byte> data)
{
    if (_device == VK_NULL_HANDLE || !destination || data.empty()) {
        return;
    }

    if (data.size_bytes() > _uploadContext.capacity) {
        EnsureUploadCapacity(static_cast<VkDeviceSize>(data.size_bytes()));
    }
    if (_uploadContext.offset + data.size_bytes() > _uploadContext.capacity) {
        FlushUploadBatch();
    }

    BeginUploadBatchRecording();

    std::byte* stagingBytes = static_cast<std::byte*>(_uploadContext.mappedData);
    std::memcpy(stagingBytes + _uploadContext.offset, data.data(), data.size_bytes());
    FlushBuffer(_uploadContext.stagingBuffer, _uploadContext.offset, static_cast<VkDeviceSize>(data.size_bytes()));

    const VkBufferCopy copyRegion{
        .srcOffset = _uploadContext.offset,
        .dstOffset = destinationOffset,
        .size = static_cast<VkDeviceSize>(data.size_bytes()),
    };
    vkCmdCopyBuffer(_uploadContext.commandBuffer, GetBuffer(_uploadContext.stagingBuffer), GetBuffer(destination), 1, &copyRegion);

    _uploadContext.offset += static_cast<VkDeviceSize>(data.size_bytes());
    ++_uploadContext.pendingCopies;
    _uploadBatchStats.pendingBytes = _uploadContext.offset;
    _uploadBatchStats.pendingCopies = _uploadContext.pendingCopies;
}

void RenderDevice::UploadImageData(ImageHandle destination, std::span<const std::byte> data)
{
    if (_device == VK_NULL_HANDLE || !destination || data.empty()) {
        return;
    }

    FlushUploadBatch();
    const AllocatedImage& image = GetImageResource(destination);
    const VkDeviceSize requiredBytes = data.size_bytes();
    EnsureUploadCapacity(requiredBytes);

    std::byte* stagingBytes = static_cast<std::byte*>(_uploadContext.mappedData);
    std::memcpy(stagingBytes, data.data(), data.size_bytes());
    FlushBuffer(_uploadContext.stagingBuffer, 0, requiredBytes);
    _uploadBatchStats.pendingBytes = requiredBytes;
    _uploadBatchStats.pendingCopies = 1;

    ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
        const VkImageSubresourceRange colorRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
        vkutil::transition_image(commandBuffer,
            image.image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            colorRange);

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = image.desc.extent;
        vkCmdCopyBufferToImage(
            commandBuffer, GetBuffer(_uploadContext.stagingBuffer), image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        vkutil::transition_image(commandBuffer,
            image.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            colorRange);
    });

    _uploadBatchStats.lastSubmittedBytes = requiredBytes;
    _uploadBatchStats.totalSubmittedBytes += requiredBytes;
    _uploadBatchStats.pendingBytes = 0;
    _uploadBatchStats.pendingCopies = 0;
}

void RenderDevice::FlushUploadBatch()
{
    if (_device == VK_NULL_HANDLE || !_uploadContext.recording) {
        return;
    }

    VK_CHECK(vkEndCommandBuffer(_uploadContext.commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo = vkinit::command_buffer_submit_info(_uploadContext.commandBuffer);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&commandBufferInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(GetTransferQueue(), 1, &submitInfo, _uploadContext.fence));
    WaitForFenceOrAssert(_uploadContext.fence, "FlushUploadBatch");

    _uploadBatchStats.lastSubmittedBytes = _uploadContext.offset;
    _uploadBatchStats.totalSubmittedBytes += _uploadContext.offset;
    _uploadBatchStats.pendingBytes = 0;
    _uploadBatchStats.pendingCopies = 0;

    _uploadContext.offset = 0;
    _uploadContext.pendingCopies = 0;
    _uploadContext.recording = false;
}

void RenderDevice::FlushBuffer(BufferHandle handle, VkDeviceSize offset, VkDeviceSize size)
{
    if (!handle || handle.index >= _buffers.size()) {
        return;
    }
    AllocatedBuffer& buffer = _buffers[handle.index];
    if (buffer.allocation == VK_NULL_HANDLE) {
        return;
    }
    VK_CHECK(vmaFlushAllocation(_allocator, buffer.allocation, offset, size));
}

void RenderDevice::InvalidateBuffer(BufferHandle handle, VkDeviceSize offset, VkDeviceSize size)
{
    if (!handle || handle.index >= _buffers.size()) {
        return;
    }
    AllocatedBuffer& buffer = _buffers[handle.index];
    if (buffer.allocation == VK_NULL_HANDLE) {
        return;
    }
    VK_CHECK(vmaInvalidateAllocation(_allocator, buffer.allocation, offset, size));
}

VkBuffer RenderDevice::GetBuffer(BufferHandle handle) const { return _buffers[handle.index].buffer; }
VkImage RenderDevice::GetImage(ImageHandle handle) const { return _images[handle.index].image; }
VkImageView RenderDevice::GetImageView(ImageHandle handle) const { return _images[handle.index].defaultView; }
VkFormat RenderDevice::GetImageFormat(ImageHandle handle) const { return _images[handle.index].desc.format; }
VkExtent3D RenderDevice::GetImageExtent(ImageHandle handle) const { return _images[handle.index].desc.extent; }
VkImageAspectFlags RenderDevice::GetImageAspectFlags(ImageHandle handle) const { return _images[handle.index].desc.aspectFlags; }

VkDeviceAddress RenderDevice::GetBufferDeviceAddress(BufferHandle handle) const
{
    VkBufferDeviceAddressInfo addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = _buffers[handle.index].buffer;
    return vkGetBufferDeviceAddress(_device, &addressInfo);
}

const AllocatedImage& RenderDevice::GetImageResource(ImageHandle handle) const { return _images[handle.index]; }
const AllocatedBuffer& RenderDevice::GetBufferResource(BufferHandle handle) const { return _buffers[handle.index]; }
ImageHandle RenderDevice::GetSwapchainImageHandle(uint32_t imageIndex) const { return _swapchainImageHandles.at(imageIndex); }

void RenderDevice::CreateInstanceAndDevice(const RenderDeviceDesc& desc)
{
    vkb::InstanceBuilder builder;
    auto instanceResult = builder.set_app_name(desc.appName)
        .set_engine_name(desc.engineName)
        .request_validation_layers(desc.enableValidation)
        .use_default_debug_messenger()
        .require_api_version(kRequiredVulkanMajor, kRequiredVulkanMinor, kRequiredVulkanPatch)
        .build();

    vkb::Instance instance = instanceResult.value();
    _instance = instance.instance;
    _debugMessenger = instance.debug_messenger;

    SDL_bool surfaceCreated = SDL_Vulkan_CreateSurface(_window, _instance, &_surface);
    if (surfaceCreated != SDL_TRUE) {
        fmt::println("Failed to create SDL Vulkan surface: {}", SDL_GetError());
        std::abort();
    }

    VkPhysicalDeviceVulkan13Features features13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;

    VkPhysicalDeviceFeatures features10{};
    features10.independentBlend = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{ instance };
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(kRequiredVulkanMajor, kRequiredVulkanMinor)
        .set_required_features(features10)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    _rayTracingSupport = {};

    // Hardware RT is optional. We probe it and keep a compute fallback so the
    // rest of the renderer does not depend on RT support.
    const bool hasRequiredRayTracingExtensions = physicalDevice.is_extension_present(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        && physicalDevice.is_extension_present(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
        && physicalDevice.is_extension_present(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    if (hasRequiredRayTracingExtensions) {
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
        };
        accelerationStructureFeatures.accelerationStructure = VK_TRUE;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
        };
        rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;

        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
        rayQueryFeatures.rayQuery = VK_TRUE;

        const bool enabledAccelerationStructure =
            physicalDevice.enable_extension_features_if_present(accelerationStructureFeatures);
        const bool enabledRayTracingPipeline =
            physicalDevice.enable_extension_features_if_present(rayTracingPipelineFeatures);
        const bool enabledRayQuery = physicalDevice.enable_extension_features_if_present(rayQueryFeatures);

        if (enabledAccelerationStructure && enabledRayTracingPipeline) {
            physicalDevice.enable_extensions_if_present(std::vector<const char*>{
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            });
            physicalDevice.enable_extension_if_present(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            physicalDevice.enable_extension_if_present(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);

            _rayTracingSupport.supported = true;
            _rayTracingSupport.accelerationStructureFeatures = accelerationStructureFeatures;
            _rayTracingSupport.rayTracingPipelineFeatures = rayTracingPipelineFeatures;
            if (enabledRayQuery) {
                _rayTracingSupport.rayQueryFeatures = rayQueryFeatures;
            }
        }
    }

    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    vkb::Device device = deviceBuilder.build().value();

    _device = device.device;
    _physicalDevice = physicalDevice.physical_device;
    _gpuName = physicalDevice.name;
    _dedicatedVideoMemoryMiB = CalculateDedicatedVideoMemoryMiB(_physicalDevice);
    _graphicsQueue = device.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = device.get_queue_index(vkb::QueueType::graphics).value();
    _presentQueue = device.get_queue(vkb::QueueType::present).value();
    _presentQueueFamily = device.get_queue_index(vkb::QueueType::present).value();
    const auto transferQueue = device.get_queue(vkb::QueueType::transfer);
    const auto transferQueueFamily = device.get_queue_index(vkb::QueueType::transfer);
    if (transferQueue && transferQueueFamily) {
        _transferQueue = transferQueue.value();
        _transferQueueFamily = transferQueueFamily.value();
    } else {
        _transferQueue = VK_NULL_HANDLE;
        _transferQueueFamily = _graphicsQueueFamily;
    }

    if (_rayTracingSupport.supported) {
        // KHR ray tracing entry points are device-level function pointers, so
        // they must be queried after logical device creation.
        _rayTracingFunctions.vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(_device, "vkCreateAccelerationStructureKHR"));
        _rayTracingFunctions.vkDestroyAccelerationStructureKHR =
            reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
                vkGetDeviceProcAddr(_device, "vkDestroyAccelerationStructureKHR"));
        _rayTracingFunctions.vkGetAccelerationStructureBuildSizesKHR =
            reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
                vkGetDeviceProcAddr(_device, "vkGetAccelerationStructureBuildSizesKHR"));
        _rayTracingFunctions.vkCmdBuildAccelerationStructuresKHR =
            reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
                vkGetDeviceProcAddr(_device, "vkCmdBuildAccelerationStructuresKHR"));
        _rayTracingFunctions.vkGetAccelerationStructureDeviceAddressKHR =
            reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
                vkGetDeviceProcAddr(_device, "vkGetAccelerationStructureDeviceAddressKHR"));
        _rayTracingFunctions.vkCreateRayTracingPipelinesKHR =
            reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
                vkGetDeviceProcAddr(_device, "vkCreateRayTracingPipelinesKHR"));
        _rayTracingFunctions.vkGetRayTracingShaderGroupHandlesKHR =
            reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
                vkGetDeviceProcAddr(_device, "vkGetRayTracingShaderGroupHandlesKHR"));
        _rayTracingFunctions.vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(_device, "vkCmdTraceRaysKHR"));

        _rayTracingSupport.supported = _rayTracingFunctions.vkCreateAccelerationStructureKHR != nullptr
            && _rayTracingFunctions.vkDestroyAccelerationStructureKHR != nullptr
            && _rayTracingFunctions.vkGetAccelerationStructureBuildSizesKHR != nullptr
            && _rayTracingFunctions.vkCmdBuildAccelerationStructuresKHR != nullptr
            && _rayTracingFunctions.vkGetAccelerationStructureDeviceAddressKHR != nullptr
            && _rayTracingFunctions.vkCreateRayTracingPipelinesKHR != nullptr
            && _rayTracingFunctions.vkGetRayTracingShaderGroupHandlesKHR != nullptr
            && _rayTracingFunctions.vkCmdTraceRaysKHR != nullptr;

        VkPhysicalDeviceProperties2 properties2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        properties2.pNext = &_rayTracingSupport.accelerationStructureProperties;
        _rayTracingSupport.accelerationStructureProperties.pNext = &_rayTracingSupport.rayTracingPipelineProperties;
        vkGetPhysicalDeviceProperties2(_physicalDevice, &properties2);
        _rayTracingSupport.accelerationStructureProperties.pNext = nullptr;
    }

}

void RenderDevice::CreateAllocator()
{
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.physicalDevice = _physicalDevice;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&allocatorInfo, &_allocator));
}

void RenderDevice::CreateSwapchain(VkExtent2D extent)
{
    vkb::SwapchainBuilder swapchainBuilder{ _physicalDevice, _device, _surface };
    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    // FIFO is chosen as the safe default because it is universally supported.
    vkb::Swapchain swapchain = swapchainBuilder.set_desired_format(
            VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(extent.width, extent.height)
        .add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .build()
        .value();

    _swapchain = swapchain.swapchain;
    _swapchainExtent = swapchain.extent;

    const std::vector<VkImage> swapchainImages = swapchain.get_images().value();
    const std::vector<VkImageView> swapchainImageViews = swapchain.get_image_views().value();

    _swapchainImageHandles.clear();
    _swapchainImageHandles.reserve(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        ImageHandle handle = AllocateImageSlot();
        AllocatedImage& image = _images[handle.index];
        image.image = swapchainImages[i];
        image.defaultView = swapchainImageViews[i];
        image.ownedBySwapchain = true;
        image.desc.extent = VkExtent3D{ _swapchainExtent.width, _swapchainExtent.height, 1 };
        image.desc.format = _swapchainImageFormat;
        image.desc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
        image.desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        _swapchainImageHandles.push_back(handle);
    }
}

void RenderDevice::DestroySwapchain()
{
    for (ImageHandle handle : _swapchainImageHandles) {
        DestroyImage(handle);
    }
    _swapchainImageHandles.clear();

    if (_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
        _swapchain = VK_NULL_HANDLE;
    }
}

void RenderDevice::CleanupResourceStorage()
{
    for (uint32_t index = 0; index < _images.size(); ++index) {
        if (_images[index].image != VK_NULL_HANDLE || _images[index].ownedBySwapchain) {
            DestroyImage(ImageHandle{ index });
        }
    }
    for (uint32_t index = 0; index < _buffers.size(); ++index) {
        if (_buffers[index].buffer != VK_NULL_HANDLE) {
            DestroyBuffer(BufferHandle{ index });
        }
    }
}

BufferHandle RenderDevice::AllocateBufferSlot()
{
    if (!_freeBufferSlots.empty()) {
        const uint32_t slot = _freeBufferSlots.back();
        _freeBufferSlots.pop_back();
        return BufferHandle{ slot };
    }

    _buffers.emplace_back();
    return BufferHandle{ static_cast<uint32_t>(_buffers.size() - 1) };
}

ImageHandle RenderDevice::AllocateImageSlot()
{
    if (!_freeImageSlots.empty()) {
        const uint32_t slot = _freeImageSlots.back();
        _freeImageSlots.pop_back();
        return ImageHandle{ slot };
    }

    _images.emplace_back();
    return ImageHandle{ static_cast<uint32_t>(_images.size() - 1) };
}
} // namespace vesta::render

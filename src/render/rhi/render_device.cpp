#include <vesta/render/rhi/render_device.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>

#include <SDL_vulkan.h>

#include <fmt/format.h>

#include <vesta/render/vulkan/vk_initializers.h>

#include "VkBootstrap.h"

namespace vesta::render {
namespace {
constexpr uint32_t kRequiredVulkanMajor = 1;
constexpr uint32_t kRequiredVulkanMinor = 3;
constexpr uint32_t kRequiredVulkanPatch = 0;

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

void BindlessDescriptorManager::Initialize(VkDevice device)
{
    // This project uses one large descriptor set instead of binding a new set
    // for every pass/resource pair. That keeps sample shaders short.
    const std::array<VkDescriptorPoolSize, 3> poolSizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxSampledImages },
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
        make_bindless_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxSampledImages),
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
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = _set;
    write.dstBinding = 0;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
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
    _bindless.Initialize(_device);
    CreateSwapchain(desc.swapchainExtent);

    return true;
}

void RenderDevice::Shutdown()
{
    WaitIdle();
    DestroySwapchain();
    CleanupResourceStorage();
    _bindless.Shutdown(_device);

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
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = desc.memoryUsage;
    allocInfo.flags = desc.allocationFlags;

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation, &buffer.allocationInfo));

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

void RenderDevice::ImmediateSubmit(const std::function<void(VkCommandBuffer)>& recorder) const
{
    if (_device == VK_NULL_HANDLE || !recorder) {
        return;
    }

    // ImmediateSubmit is useful for setup work such as uploads or AS builds.
    // It records a short-lived command buffer, submits it, and waits right away.
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
    VK_CHECK(vkCreateCommandPool(_device, &poolInfo, nullptr, &commandPool));

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(commandPool);
    VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer));

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo = vkinit::fence_create_info();
    VK_CHECK(vkCreateFence(_device, &fenceInfo, nullptr, &fence));

    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    recorder(commandBuffer);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo = vkinit::command_buffer_submit_info(commandBuffer);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&commandBufferInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(_device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));

    vkDestroyFence(_device, fence, nullptr);
    vkDestroyCommandPool(_device, commandPool, nullptr);
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

    vkb::PhysicalDeviceSelector selector{ instance };
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(kRequiredVulkanMajor, kRequiredVulkanMinor)
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

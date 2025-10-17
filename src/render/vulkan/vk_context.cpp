#include <vesta/render/vulkan/vk_context.h>

#include <cstdlib>
#include <SDL.h>
#include <SDL_vulkan.h>

#include <fmt/format.h>

#include "VkBootstrap.h"

namespace {
// Raise/lower this if you want to target a different Vulkan baseline.
constexpr uint32_t kRequiredVulkanMajor = 1;
constexpr uint32_t kRequiredVulkanMinor = 3;
constexpr uint32_t kRequiredVulkanPatch = 0;
}

void vkctx::init_instance_and_device(VulkanContext& context, SDL_Window* window, bool enableValidationLayers)
{
    vkb::InstanceBuilder builder;

    // App/engine names are visible in Vulkan tools and validation output.
    auto instanceResult = builder.set_app_name("Vesta Engine")
        .set_engine_name("Vesta Engine")
        // Toggle this from the engine side if you want a quieter release path.
        .request_validation_layers(enableValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(kRequiredVulkanMajor, kRequiredVulkanMinor, kRequiredVulkanPatch)
        .build();

    vkb::Instance instance = instanceResult.value();
    context.instance = instance.instance;
    context.debugMessenger = instance.debug_messenger;

    SDL_bool surfaceCreated = SDL_Vulkan_CreateSurface(window, context.instance, &context.surface);
    if (surfaceCreated != SDL_TRUE) {
        fmt::println("Failed to create SDL Vulkan surface: {}", SDL_GetError());
        abort();
    }

    VkPhysicalDeviceVulkan13Features features13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    // Keep only the features your renderer really uses. More required features means fewer compatible GPUs.
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{ instance };
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(kRequiredVulkanMajor, kRequiredVulkanMinor)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(context.surface)
        .select()
        .value();

    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    vkb::Device device = deviceBuilder.build().value();

    context.device = device.device;
    context.physicalDevice = physicalDevice.physical_device;
    context.graphicsQueue = device.get_queue(vkb::QueueType::graphics).value();
    context.graphicsQueueFamily = device.get_queue_index(vkb::QueueType::graphics).value();
    context.presentQueue = device.get_queue(vkb::QueueType::present).value();
    context.presentQueueFamily = device.get_queue_index(vkb::QueueType::present).value();
}

void vkctx::create_swapchain(VulkanContext& context, VkExtent2D extent)
{
    vkb::SwapchainBuilder swapchainBuilder{ context.physicalDevice, context.device, context.surface };

    // Common Windows format. If you want HDR, linear workflows, or different post-process paths, revisit this.
    context.swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain swapchain = swapchainBuilder.set_desired_format(
            VkSurfaceFormatKHR{ .format = context.swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        // FIFO is the safest default. MAILBOX favors lower latency; IMMEDIATE favors tearing/lowest latency.
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        // Right now the engine matches the swapchain to the window size directly.
        .set_desired_extent(extent.width, extent.height)
        // Current renderer only clears via transfer. Add COLOR_ATTACHMENT if you move to direct raster rendering.
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    context.swapchain = swapchain.swapchain;
    context.swapchainExtent = swapchain.extent;
    context.swapchainImages = swapchain.get_images().value();
    context.swapchainImageViews = swapchain.get_image_views().value();
}

void vkctx::destroy_swapchain(VulkanContext& context)
{
    for (VkImageView imageView : context.swapchainImageViews) {
        vkDestroyImageView(context.device, imageView, nullptr);
    }
    context.swapchainImageViews.clear();
    context.swapchainImages.clear();

    if (context.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(context.device, context.swapchain, nullptr);
        context.swapchain = VK_NULL_HANDLE;
    }
}

void vkctx::destroy_context(VulkanContext& context)
{
    if (context.device != VK_NULL_HANDLE) {
        vkDestroyDevice(context.device, nullptr);
        context.device = VK_NULL_HANDLE;
    }

    if (context.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(context.instance, context.surface, nullptr);
        context.surface = VK_NULL_HANDLE;
    }

    if (context.debugMessenger != VK_NULL_HANDLE) {
        vkb::destroy_debug_utils_messenger(context.instance, context.debugMessenger);
        context.debugMessenger = VK_NULL_HANDLE;
    }

    if (context.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context.instance, nullptr);
        context.instance = VK_NULL_HANDLE;
    }
}

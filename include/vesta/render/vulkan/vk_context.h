#pragma once

#include <vesta/render/vulkan/vk_types.h>

struct SDL_Window;

struct VulkanContext {
    VkInstance instance{ VK_NULL_HANDLE };
    VkDebugUtilsMessengerEXT debugMessenger{ VK_NULL_HANDLE };
    VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
    VkDevice device{ VK_NULL_HANDLE };
    VkSurfaceKHR surface{ VK_NULL_HANDLE };

    VkQueue graphicsQueue{ VK_NULL_HANDLE };
    uint32_t graphicsQueueFamily{ 0 };
    VkQueue presentQueue{ VK_NULL_HANDLE };
    uint32_t presentQueueFamily{ 0 };

    VkSwapchainKHR swapchain{ VK_NULL_HANDLE };
    VkFormat swapchainImageFormat{ VK_FORMAT_UNDEFINED };
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkExtent2D swapchainExtent{};
};

namespace vkctx {
void init_instance_and_device(VulkanContext& context, SDL_Window* window, bool enableValidationLayers);
void create_swapchain(VulkanContext& context, VkExtent2D extent);
void destroy_swapchain(VulkanContext& context);
void destroy_context(VulkanContext& context);
}

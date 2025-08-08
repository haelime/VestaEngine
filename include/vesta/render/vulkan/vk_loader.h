#pragma once

#include <filesystem>

#include <vesta/render/vulkan/vk_types.h>

namespace vkutil {
[[nodiscard]] std::vector<uint32_t> load_spirv_file(const std::filesystem::path& path);
[[nodiscard]] VkShaderModule create_shader_module(VkDevice device, std::span<const uint32_t> spirvCode);
[[nodiscard]] VkShaderModule load_shader_module(VkDevice device, const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path resolve_runtime_path(const std::filesystem::path& relativePath);
void destroy_shader_module(VkDevice device, VkShaderModule& shaderModule);
} // namespace vkutil

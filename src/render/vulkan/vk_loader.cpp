#include <vesta/render/vulkan/vk_loader.h>

#include <fstream>
#include <stdexcept>

std::vector<uint32_t> vkutil::load_spirv_file(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + path.string());
    }

    const std::streamsize byteSize = file.tellg();
    if (byteSize <= 0 || (byteSize % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) {
        throw std::runtime_error("Invalid SPIR-V byte size: " + path.string());
    }

    std::vector<uint32_t> spirv(static_cast<size_t>(byteSize) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(spirv.data()), byteSize);

    if (!file) {
        throw std::runtime_error("Failed to read SPIR-V file: " + path.string());
    }

    return spirv;
}

VkShaderModule vkutil::create_shader_module(VkDevice device, std::span<const uint32_t> spirvCode)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size_bytes();
    createInfo.pCode = spirvCode.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
    return shaderModule;
}

VkShaderModule vkutil::load_shader_module(VkDevice device, const std::filesystem::path& path)
{
    const std::vector<uint32_t> spirv = load_spirv_file(path);
    return create_shader_module(device, spirv);
}

void vkutil::destroy_shader_module(VkDevice device, VkShaderModule& shaderModule)
{
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }
}

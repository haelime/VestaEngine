#include <vesta/render/vulkan/vk_loader.h>

#include <array>
#include <fstream>
#include <stdexcept>

#include <Windows.h>

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

std::filesystem::path vkutil::resolve_runtime_path(const std::filesystem::path& relativePath)
{
    std::array<wchar_t, MAX_PATH> modulePath{};
    const DWORD length = GetModuleFileNameW(nullptr, modulePath.data(), static_cast<DWORD>(modulePath.size()));

    const std::array searchRoots{
        length > 0 ? std::filesystem::path(modulePath.data()).parent_path() : std::filesystem::path{},
        std::filesystem::current_path(),
    };

    for (const std::filesystem::path& root : searchRoots) {
        if (root.empty()) {
            continue;
        }

        for (std::filesystem::path cursor = root; !cursor.empty();) {
            const std::filesystem::path candidate = cursor / relativePath;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }

            const std::filesystem::path parent = cursor.parent_path();
            if (parent == cursor) {
                break;
            }
            cursor = parent;
        }
    }

    throw std::runtime_error("Failed to locate runtime path: " + relativePath.string());
}

void vkutil::destroy_shader_module(VkDevice device, VkShaderModule& shaderModule)
{
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }
}

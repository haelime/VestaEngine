#include <vesta/scene/scene.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <string_view>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fmt/format.h>

#include <glm/common.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <vesta/core/debug.h>
#include <vesta/render/renderer.h>
#include <vesta/render/rhi/render_device.h>
#include <vesta/render/vulkan/vk_images.h>
#include <vesta/scene/camera.h>

namespace vesta::scene {
namespace {
constexpr uint32_t kRealtimeGaussianSortLimit = 200000;

constexpr auto kLoadOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::LoadGLBBuffers
    | fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages | fastgltf::Options::GenerateMeshIndices;
constexpr VmaAllocationCreateFlags kMappedHostFlags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
constexpr float kShC0 = 0.28209479177387814f;
const glm::quat kGaussianImportRotation = glm::angleAxis(glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));

bool ShouldAutoLayoutDemoScene(const std::filesystem::path& path, const fastgltf::Scene& scene)
{
    return path.filename() == "basicmesh.glb" && scene.nodeIndices.size() > 1;
}

glm::mat4 MakeDemoRootLayoutTransform(size_t rootIndex, size_t rootCount)
{
    constexpr float kRootSpacing = 2.75f;
    const float centeredIndex = static_cast<float>(rootIndex) - (static_cast<float>(rootCount) - 1.0f) * 0.5f;
    return glm::translate(glm::mat4(1.0f), glm::vec3(centeredIndex * kRootSpacing, 0.0f, 0.0f));
}

glm::mat4 NodeToMatrix(const fastgltf::Node& node)
{
    if (const auto* matrix = std::get_if<fastgltf::Node::TransformMatrix>(&node.transform)) {
        glm::mat4 result(1.0f);
        std::memcpy(&result[0][0], matrix->data(), sizeof(float) * 16);
        return result;
    }

    const auto* trs = std::get_if<fastgltf::Node::TRS>(&node.transform);
    if (trs == nullptr) {
        return glm::mat4(1.0f);
    }

    const glm::vec3 translation(trs->translation[0], trs->translation[1], trs->translation[2]);
    const glm::quat rotation(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]);
    const glm::vec3 scale(trs->scale[0], trs->scale[1], trs->scale[2]);

    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

glm::vec3 ApplyGaussianImportTransform(glm::vec3 position)
{
    return kGaussianImportRotation * position;
}

glm::vec4 ApplyGaussianImportTransform(glm::vec4 rotation)
{
    const glm::quat gaussianRotation(rotation.w, rotation.x, rotation.y, rotation.z);
    const glm::quat transformed = glm::normalize(kGaussianImportRotation * gaussianRotation);
    return glm::vec4(transformed.x, transformed.y, transformed.z, transformed.w);
}

glm::vec3 NormalizeGaussianScaleForScene(glm::vec3 scale, float sceneRadius)
{
    constexpr float kAbsoluteMinScale = 1.0e-6f;
    constexpr float kRelativeMinScale = 1.0e-7f;

    const float minScale = std::max(kAbsoluteMinScale, sceneRadius * kRelativeMinScale);
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(scale[axis]) || scale[axis] <= 0.0f) {
            scale[axis] = minScale;
        } else {
            scale[axis] = std::max(scale[axis], minScale);
        }
    }
    return scale;
}

std::vector<glm::vec3> ReadVec3Accessor(const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
{
    std::vector<glm::vec3> data(accessor.count);
    fastgltf::copyFromAccessor<glm::vec3>(asset, accessor, data.data());
    return data;
}

std::vector<glm::vec2> ReadVec2Accessor(const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
{
    std::vector<glm::vec2> data(accessor.count);
    fastgltf::copyFromAccessor<glm::vec2>(asset, accessor, data.data());
    return data;
}

std::vector<glm::vec4> ReadVec4Accessor(const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
{
    std::vector<glm::vec4> data(accessor.count);
    fastgltf::copyFromAccessor<glm::vec4>(asset, accessor, data.data());
    return data;
}

std::vector<uint32_t> ReadIndexAccessor(const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
{
    std::vector<uint32_t> data(accessor.count);
    fastgltf::copyFromAccessor<uint32_t>(asset, accessor, data.data());
    return data;
}

std::span<const std::byte> GetBufferSourceBytes(const fastgltf::DataSource& dataSource)
{
    return std::visit(
        [](const auto& source) -> std::span<const std::byte> {
            using T = std::decay_t<decltype(source)>;
            if constexpr (std::is_same_v<T, fastgltf::sources::Vector>) {
                return {
                    reinterpret_cast<const std::byte*>(source.bytes.data()),
                    source.bytes.size(),
                };
            } else if constexpr (std::is_same_v<T, fastgltf::sources::ByteView>) {
                return source.bytes;
            } else {
                return {};
            }
        },
        dataSource);
}

std::span<const std::byte> GetImageSourceBytes(const fastgltf::Asset& asset, const fastgltf::Image& image)
{
    return std::visit(
        [&](const auto& source) -> std::span<const std::byte> {
            using T = std::decay_t<decltype(source)>;
            if constexpr (std::is_same_v<T, fastgltf::sources::Vector>) {
                return {
                    reinterpret_cast<const std::byte*>(source.bytes.data()),
                    source.bytes.size(),
                };
            } else if constexpr (std::is_same_v<T, fastgltf::sources::ByteView>) {
                return source.bytes;
            } else if constexpr (std::is_same_v<T, fastgltf::sources::BufferView>) {
                const fastgltf::BufferView& bufferView = asset.bufferViews.at(source.bufferViewIndex);
                const fastgltf::Buffer& buffer = asset.buffers.at(bufferView.bufferIndex);
                const std::span<const std::byte> bufferBytes = GetBufferSourceBytes(buffer.data);
                if (bufferBytes.empty()) {
                    return {};
                }
                const size_t byteOffset = bufferView.byteOffset;
                const size_t byteLength = bufferView.byteLength;
                if (byteOffset >= bufferBytes.size()) {
                    return {};
                }
                return bufferBytes.subspan(byteOffset, std::min(byteLength, bufferBytes.size() - byteOffset));
            } else {
                return {};
            }
        },
        image.data);
}

std::optional<SceneTextureAsset> DecodeTextureAsset(
    const fastgltf::Asset& asset, const fastgltf::Image& image, const std::filesystem::path& sourcePath, bool srgb)
{
    SceneTextureAsset textureAsset;
    textureAsset.name = image.name;
    textureAsset.srgb = srgb;

    int width = 0;
    int height = 0;
    int channelCount = 0;
    stbi_uc* decodedPixels = nullptr;

    if (const auto* uriSource = std::get_if<fastgltf::sources::URI>(&image.data); uriSource != nullptr) {
        const std::filesystem::path imagePath = sourcePath.parent_path() / uriSource->uri.fspath();
        decodedPixels = stbi_load(imagePath.string().c_str(), &width, &height, &channelCount, STBI_rgb_alpha);
        if (textureAsset.name.empty()) {
            textureAsset.name = imagePath.filename().string();
        }
    } else {
        const std::span<const std::byte> imageBytes = GetImageSourceBytes(asset, image);
        if (!imageBytes.empty()) {
            decodedPixels = stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(imageBytes.data()), static_cast<int>(imageBytes.size()), &width, &height, &channelCount, STBI_rgb_alpha);
        }
    }

    if (decodedPixels == nullptr || width <= 0 || height <= 0) {
        if (decodedPixels != nullptr) {
            stbi_image_free(decodedPixels);
        }
        return std::nullopt;
    }

    textureAsset.width = static_cast<uint32_t>(width);
    textureAsset.height = static_cast<uint32_t>(height);
    textureAsset.rgba8Pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    std::memcpy(textureAsset.rgba8Pixels.data(), decodedPixels, textureAsset.rgba8Pixels.size());
    stbi_image_free(decodedPixels);

    if (textureAsset.name.empty()) {
        textureAsset.name = "SceneTexture";
    }
    return textureAsset;
}

SceneMaterial MakeDefaultMaterial()
{
    return SceneMaterial{};
}

std::vector<glm::vec4> GenerateTangents(std::span<const glm::vec3> positions,
    std::span<const glm::vec3> normals,
    std::span<const glm::vec2> texCoords,
    std::span<const uint32_t> indices)
{
    std::vector<glm::vec4> tangents(positions.size(), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    if (positions.empty() || texCoords.size() < positions.size() || indices.size() < 3) {
        return tangents;
    }

    std::vector<glm::vec3> tan1(positions.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> tan2(positions.size(), glm::vec3(0.0f));

    for (size_t triangle = 0; triangle + 2 < indices.size(); triangle += 3) {
        const uint32_t i0 = indices[triangle + 0];
        const uint32_t i1 = indices[triangle + 1];
        const uint32_t i2 = indices[triangle + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
            continue;
        }

        const glm::vec3 edge1 = positions[i1] - positions[i0];
        const glm::vec3 edge2 = positions[i2] - positions[i0];
        const glm::vec2 uv1 = texCoords[i1] - texCoords[i0];
        const glm::vec2 uv2 = texCoords[i2] - texCoords[i0];
        const float denominator = uv1.x * uv2.y - uv1.y * uv2.x;
        if (std::abs(denominator) < 1.0e-6f) {
            continue;
        }

        const float inverse = 1.0f / denominator;
        const glm::vec3 tangent = (edge1 * uv2.y - edge2 * uv1.y) * inverse;
        const glm::vec3 bitangent = (edge2 * uv1.x - edge1 * uv2.x) * inverse;

        tan1[i0] += tangent;
        tan1[i1] += tangent;
        tan1[i2] += tangent;
        tan2[i0] += bitangent;
        tan2[i1] += bitangent;
        tan2[i2] += bitangent;
    }

    for (size_t vertexIndex = 0; vertexIndex < positions.size(); ++vertexIndex) {
        glm::vec3 normal = vertexIndex < normals.size() ? normals[vertexIndex] : glm::vec3(0.0f, 1.0f, 0.0f);
        if (!std::isfinite(normal.x) || !std::isfinite(normal.y) || !std::isfinite(normal.z) || glm::length(normal) < 1.0e-4f) {
            normal = glm::vec3(0.0f, 1.0f, 0.0f);
        } else {
            normal = glm::normalize(normal);
        }

        glm::vec3 tangent = tan1[vertexIndex] - normal * glm::dot(normal, tan1[vertexIndex]);
        if (!std::isfinite(tangent.x) || !std::isfinite(tangent.y) || !std::isfinite(tangent.z) || glm::length(tangent) < 1.0e-4f) {
            tangent = std::abs(normal.y) > 0.99f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), normal));
        } else {
            tangent = glm::normalize(tangent);
        }

        const float handedness = glm::dot(glm::cross(normal, tangent), tan2[vertexIndex]) < 0.0f ? -1.0f : 1.0f;
        tangents[vertexIndex] = glm::vec4(tangent, handedness);
    }

    return tangents;
}

void FinalizeBounds(SceneBounds& bounds, const std::vector<SceneVertex>& vertices)
{
    if (vertices.empty()) {
        bounds = {};
        return;
    }

    bounds.minimum = vertices.front().position;
    bounds.maximum = vertices.front().position;
    for (const SceneVertex& vertex : vertices) {
        bounds.minimum = glm::min(bounds.minimum, vertex.position);
        bounds.maximum = glm::max(bounds.maximum, vertex.position);
    }

    bounds.center = (bounds.minimum + bounds.maximum) * 0.5f;
    bounds.radius = 0.0f;
    for (const SceneVertex& vertex : vertices) {
        bounds.radius = std::max(bounds.radius, glm::distance(bounds.center, vertex.position));
    }
    bounds.radius = std::max(bounds.radius, 1.0f);
}

SceneSurfaceBounds ComputeSurfaceBounds(const std::vector<SceneVertex>& vertices, uint32_t baseVertex, std::span<const uint32_t> primitiveIndices)
{
    if (primitiveIndices.empty()) {
        return {};
    }

    glm::vec3 minimum = vertices[baseVertex + primitiveIndices.front()].position;
    glm::vec3 maximum = minimum;
    for (uint32_t index : primitiveIndices) {
        const glm::vec3 position = vertices[baseVertex + index].position;
        minimum = glm::min(minimum, position);
        maximum = glm::max(maximum, position);
    }

    SceneSurfaceBounds bounds{};
    bounds.center = (minimum + maximum) * 0.5f;
    for (uint32_t index : primitiveIndices) {
        bounds.radius = std::max(bounds.radius, glm::distance(bounds.center, vertices[baseVertex + index].position));
    }
    return bounds;
}

SceneBounds ComputeVertexRangeBounds(const std::vector<SceneVertex>& vertices, uint32_t firstVertex, uint32_t vertexCount)
{
    SceneBounds bounds{};
    if (vertexCount == 0 || firstVertex >= vertices.size()) {
        return bounds;
    }

    const uint32_t endVertex = std::min<uint32_t>(firstVertex + vertexCount, static_cast<uint32_t>(vertices.size()));
    bounds.minimum = vertices[firstVertex].position;
    bounds.maximum = vertices[firstVertex].position;
    for (uint32_t vertexIndex = firstVertex; vertexIndex < endVertex; ++vertexIndex) {
        bounds.minimum = glm::min(bounds.minimum, vertices[vertexIndex].position);
        bounds.maximum = glm::max(bounds.maximum, vertices[vertexIndex].position);
    }

    bounds.center = (bounds.minimum + bounds.maximum) * 0.5f;
    bounds.radius = 0.0f;
    for (uint32_t vertexIndex = firstVertex; vertexIndex < endVertex; ++vertexIndex) {
        bounds.radius = std::max(bounds.radius, glm::distance(bounds.center, vertices[vertexIndex].position));
    }
    return bounds;
}

void DestroyRayTracingResources(vesta::render::RenderDevice& device, GpuScene& gpu)
{
    if (gpu.topLevelAccelerationStructure != VK_NULL_HANDLE) {
        device.GetRayTracingFunctions().vkDestroyAccelerationStructureKHR(device.GetDevice(), gpu.topLevelAccelerationStructure, nullptr);
        gpu.topLevelAccelerationStructure = VK_NULL_HANDLE;
    }
    if (gpu.bottomLevelAccelerationStructure != VK_NULL_HANDLE) {
        device.GetRayTracingFunctions().vkDestroyAccelerationStructureKHR(
            device.GetDevice(), gpu.bottomLevelAccelerationStructure, nullptr);
        gpu.bottomLevelAccelerationStructure = VK_NULL_HANDLE;
    }
    if (gpu.topLevelBuffer) {
        device.DestroyBuffer(gpu.topLevelBuffer);
        gpu.topLevelBuffer = {};
    }
    if (gpu.bottomLevelBuffer) {
        device.DestroyBuffer(gpu.bottomLevelBuffer);
        gpu.bottomLevelBuffer = {};
    }
    gpu.bottomLevelBuildMs = 0.0f;
    gpu.topLevelBuildMs = 0.0f;
}

bool IntersectRaySphere(
    const glm::vec3& rayOrigin, const glm::vec3& rayDirection, const glm::vec3& center, float radius, float& hitDistance)
{
    const glm::vec3 offset = rayOrigin - center;
    const float b = glm::dot(offset, rayDirection);
    const float c = glm::dot(offset, offset) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0f) {
        return false;
    }

    float t = -b - std::sqrt(discriminant);
    if (t < 0.0f) {
        t = -b + std::sqrt(discriminant);
    }
    if (t < 0.0f) {
        return false;
    }

    hitDistance = t;
    return true;
}

template <typename T>
void CopyToMappedBuffer(vesta::render::RenderDevice& device, vesta::render::BufferHandle handle, std::span<const T> data)
{
    if (data.empty()) {
        return;
    }

    const vesta::render::AllocatedBuffer& buffer = device.GetBufferResource(handle);
    std::memcpy(buffer.allocationInfo.pMappedData, data.data(), data.size_bytes());
    device.FlushBuffer(handle, 0, static_cast<VkDeviceSize>(data.size_bytes()));
}

VkTransformMatrixKHR MakeIdentityTransformMatrix()
{
    VkTransformMatrixKHR matrix{};
    matrix.matrix[0][0] = 1.0f;
    matrix.matrix[1][1] = 1.0f;
    matrix.matrix[2][2] = 1.0f;
    return matrix;
}

enum class PlyFormat {
    Unknown = 0,
    Ascii,
    BinaryLittleEndian,
};

enum class PlyScalarType {
    Invalid = 0,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
};

struct PlyProperty {
    std::string name;
    PlyScalarType type{ PlyScalarType::Invalid };
};

struct PlyVertexLayout {
    PlyFormat format{ PlyFormat::Unknown };
    size_t vertexCount{ 0 };
    std::vector<PlyProperty> properties;
    size_t headerBytes{ 0 };
};

PlyScalarType ParsePlyScalarType(std::string_view type)
{
    if (type == "char" || type == "int8") {
        return PlyScalarType::Int8;
    }
    if (type == "uchar" || type == "uint8") {
        return PlyScalarType::Uint8;
    }
    if (type == "short" || type == "int16") {
        return PlyScalarType::Int16;
    }
    if (type == "ushort" || type == "uint16") {
        return PlyScalarType::Uint16;
    }
    if (type == "int" || type == "int32") {
        return PlyScalarType::Int32;
    }
    if (type == "uint" || type == "uint32") {
        return PlyScalarType::Uint32;
    }
    if (type == "float" || type == "float32") {
        return PlyScalarType::Float32;
    }
    if (type == "double" || type == "float64") {
        return PlyScalarType::Float64;
    }
    return PlyScalarType::Invalid;
}

size_t PlyScalarTypeSize(PlyScalarType type)
{
    switch (type) {
    case PlyScalarType::Int8:
    case PlyScalarType::Uint8:
        return 1;
    case PlyScalarType::Int16:
    case PlyScalarType::Uint16:
        return 2;
    case PlyScalarType::Int32:
    case PlyScalarType::Uint32:
    case PlyScalarType::Float32:
        return 4;
    case PlyScalarType::Float64:
        return 8;
    case PlyScalarType::Invalid:
    default:
        return 0;
    }
}

float Sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

float ReadPlyScalarAsFloat(const std::byte* source, PlyScalarType type)
{
    switch (type) {
    case PlyScalarType::Int8:
        return static_cast<float>(*reinterpret_cast<const int8_t*>(source));
    case PlyScalarType::Uint8:
        return static_cast<float>(*reinterpret_cast<const uint8_t*>(source));
    case PlyScalarType::Int16:
        return static_cast<float>(*reinterpret_cast<const int16_t*>(source));
    case PlyScalarType::Uint16:
        return static_cast<float>(*reinterpret_cast<const uint16_t*>(source));
    case PlyScalarType::Int32:
        return static_cast<float>(*reinterpret_cast<const int32_t*>(source));
    case PlyScalarType::Uint32:
        return static_cast<float>(*reinterpret_cast<const uint32_t*>(source));
    case PlyScalarType::Float32:
        return *reinterpret_cast<const float*>(source);
    case PlyScalarType::Float64:
        return static_cast<float>(*reinterpret_cast<const double*>(source));
    case PlyScalarType::Invalid:
    default:
        return 0.0f;
    }
}

bool ParsePlyHeader(const std::filesystem::path& path, PlyVertexLayout& layout)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return false;
    }

    std::string line;
    if (!std::getline(input, line) || line != "ply") {
        return false;
    }
    bool inVertexElement = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::istringstream stream(line);
        std::string token;
        stream >> token;
        if (token == "format") {
            std::string formatName;
            stream >> formatName;
            if (formatName == "ascii") {
                layout.format = PlyFormat::Ascii;
            } else if (formatName == "binary_little_endian") {
                layout.format = PlyFormat::BinaryLittleEndian;
            } else {
                layout.format = PlyFormat::Unknown;
            }
        } else if (token == "element") {
            std::string elementName;
            size_t count = 0;
            stream >> elementName >> count;
            inVertexElement = elementName == "vertex";
            if (inVertexElement) {
                layout.vertexCount = count;
            }
        } else if (token == "property" && inVertexElement) {
            std::string typeName;
            std::string propertyName;
            stream >> typeName;
            if (typeName == "list") {
                return false;
            }
            stream >> propertyName;
            layout.properties.push_back(PlyProperty{
                .name = propertyName,
                .type = ParsePlyScalarType(typeName),
            });
        } else if (token == "end_header") {
            layout.headerBytes = static_cast<size_t>(input.tellg());
            return layout.format != PlyFormat::Unknown && layout.vertexCount > 0 && !layout.properties.empty();
        }
    }

    return false;
}

std::optional<std::filesystem::path> ResolveGaussianSourcePath(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    if (std::filesystem::is_regular_file(path)) {
        return path;
    }

    if (!std::filesystem::is_directory(path)) {
        return std::nullopt;
    }

    const std::filesystem::path directPly = path / "point_cloud.ply";
    if (std::filesystem::exists(directPly)) {
        return directPly;
    }

    const std::filesystem::path pointCloudDirectory = std::filesystem::exists(path / "point_cloud") ? path / "point_cloud" : path;
    std::optional<std::filesystem::path> bestPath;
    int bestIteration = -1;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(pointCloudDirectory)) {
        if (!entry.is_directory()) {
            continue;
        }

        const std::string directoryName = entry.path().filename().string();
        if (!directoryName.starts_with("iteration_")) {
            continue;
        }

        int iteration = -1;
        try {
            iteration = std::stoi(directoryName.substr(std::string("iteration_").size()));
        } catch (...) {
            iteration = -1;
        }

        const std::filesystem::path candidate = entry.path() / "point_cloud.ply";
        if (iteration >= 0 && std::filesystem::exists(candidate) && iteration > bestIteration) {
            bestIteration = iteration;
            bestPath = candidate;
        }
    }

    return bestPath;
}

uint32_t DetectGaussianShDegree(const std::unordered_map<std::string, size_t>& propertyIndex)
{
    size_t restCount = 0;
    while (propertyIndex.contains(fmt::format("f_rest_{}", restCount))) {
        ++restCount;
    }

    if (restCount >= 45) {
        return 3;
    }
    if (restCount >= 24) {
        return 2;
    }
    if (restCount >= 9) {
        return 1;
    }
    return 0;
}

bool ParseGaussianPly(const std::filesystem::path& path, ParsedScene& parsedScene)
{
    const std::optional<std::filesystem::path> gaussianPath = ResolveGaussianSourcePath(path);
    if (!gaussianPath.has_value()) {
        return false;
    }

    PlyVertexLayout layout;
    if (!ParsePlyHeader(*gaussianPath, layout)) {
        return false;
    }

    std::unordered_map<std::string, size_t> propertyIndex;
    for (size_t i = 0; i < layout.properties.size(); ++i) {
        propertyIndex.emplace(layout.properties[i].name, i);
        if (layout.properties[i].type == PlyScalarType::Invalid) {
            return false;
        }
    }

    parsedScene.gaussianShDegree = DetectGaussianShDegree(propertyIndex);

    const auto findProperty = [&](std::string_view name) -> int {
        const auto it = propertyIndex.find(std::string(name));
        return it != propertyIndex.end() ? static_cast<int>(it->second) : -1;
    };

    struct GaussianPropertyIndices {
        int x{ -1 };
        int y{ -1 };
        int z{ -1 };
        int red{ -1 };
        int green{ -1 };
        int blue{ -1 };
        int opacity{ -1 };
        int scale0{ -1 };
        int scale1{ -1 };
        int scale2{ -1 };
        int rot0{ -1 };
        int rot1{ -1 };
        int rot2{ -1 };
        int rot3{ -1 };
        int fdc0{ -1 };
        int fdc1{ -1 };
        int fdc2{ -1 };
        std::array<std::array<int, 3>, kGaussianMaxShCoefficients> fRest{};
    } indices;
    for (auto& triplet : indices.fRest) {
        triplet = { -1, -1, -1 };
    }

    indices.x = findProperty("x");
    indices.y = findProperty("y");
    indices.z = findProperty("z");
    indices.red = findProperty("red");
    indices.green = findProperty("green");
    indices.blue = findProperty("blue");
    indices.opacity = findProperty("opacity");
    indices.scale0 = findProperty("scale_0");
    indices.scale1 = findProperty("scale_1");
    indices.scale2 = findProperty("scale_2");
    indices.rot0 = findProperty("rot_0");
    indices.rot1 = findProperty("rot_1");
    indices.rot2 = findProperty("rot_2");
    indices.rot3 = findProperty("rot_3");
    indices.fdc0 = findProperty("f_dc_0");
    indices.fdc1 = findProperty("f_dc_1");
    indices.fdc2 = findProperty("f_dc_2");

    const uint32_t activeCoefficientCount =
        std::min<uint32_t>((parsedScene.gaussianShDegree + 1u) * (parsedScene.gaussianShDegree + 1u), kGaussianMaxShCoefficients);
    for (uint32_t coefficientIndex = 1; coefficientIndex < activeCoefficientCount; ++coefficientIndex) {
        const uint32_t restBaseIndex = (coefficientIndex - 1u) * 3u;
        indices.fRest[coefficientIndex][0] = findProperty(fmt::format("f_rest_{}", restBaseIndex + 0u));
        indices.fRest[coefficientIndex][1] = findProperty(fmt::format("f_rest_{}", restBaseIndex + 1u));
        indices.fRest[coefficientIndex][2] = findProperty(fmt::format("f_rest_{}", restBaseIndex + 2u));
    }

    const auto readVertexFromValues = [&](const std::vector<float>& values) {
        auto getValue = [&](std::string_view name, float fallback = 0.0f) {
            const auto it = propertyIndex.find(std::string(name));
            return it != propertyIndex.end() && it->second < values.size() ? values[it->second] : fallback;
        };

        SceneVertex vertex{};
        vertex.position = glm::vec3(getValue("x"), getValue("y"), getValue("z"));

        const bool hasShColor = propertyIndex.contains("f_dc_0") && propertyIndex.contains("f_dc_1") && propertyIndex.contains("f_dc_2");
        if (hasShColor) {
            vertex.color.r = glm::clamp(kShC0 * getValue("f_dc_0") + 0.5f, 0.0f, 1.0f);
            vertex.color.g = glm::clamp(kShC0 * getValue("f_dc_1") + 0.5f, 0.0f, 1.0f);
            vertex.color.b = glm::clamp(kShC0 * getValue("f_dc_2") + 0.5f, 0.0f, 1.0f);
        } else if (propertyIndex.contains("red") && propertyIndex.contains("green") && propertyIndex.contains("blue")) {
            vertex.color.r = getValue("red") > 1.0f ? getValue("red") / 255.0f : getValue("red");
            vertex.color.g = getValue("green") > 1.0f ? getValue("green") / 255.0f : getValue("green");
            vertex.color.b = getValue("blue") > 1.0f ? getValue("blue") / 255.0f : getValue("blue");
        }

        const bool hasScales =
            propertyIndex.contains("scale_0") && propertyIndex.contains("scale_1") && propertyIndex.contains("scale_2");
        if (hasScales) {
            const float sx = std::exp(getValue("scale_0"));
            const float sy = std::exp(getValue("scale_1"));
            const float sz = std::exp(getValue("scale_2"));
            vertex.normal = glm::vec3(sx, sy, sz);
            vertex.splatParams.x = (sx + sy + sz) / 3.0f;
        } else {
            // Plain point clouds do not carry learned Gaussian scale, so use a
            // tiny screen-space baseline that stays readable instead of a huge blur.
            vertex.normal = glm::vec3(1.0f);
            vertex.splatParams.x = 0.28f;
        }

        if (propertyIndex.contains("rot_0") && propertyIndex.contains("rot_1") && propertyIndex.contains("rot_2")
            && propertyIndex.contains("rot_3")) {
            glm::vec4 rotation(getValue("rot_1"), getValue("rot_2"), getValue("rot_3"), getValue("rot_0", 1.0f));
            const float rotationLength = glm::length(rotation);
            vertex.tangent = rotationLength > 1.0e-4f ? rotation / rotationLength : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        } else {
            vertex.tangent = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        vertex.splatParams.y = propertyIndex.contains("opacity") ? Sigmoid(getValue("opacity")) : vertex.color.a;
        vertex.color.a = 1.0f;
        return vertex;
    };

    const auto readGaussianFromValues = [&](const std::vector<float>& values) {
        auto getValue = [&](std::string_view name, float fallback = 0.0f) {
            const auto it = propertyIndex.find(std::string(name));
            return it != propertyIndex.end() && it->second < values.size() ? values[it->second] : fallback;
        };

        GaussianPrimitive gaussian{};
        gaussian.positionOpacity = glm::vec4(getValue("x"), getValue("y"), getValue("z"), 1.0f);

        if (propertyIndex.contains("opacity")) {
            gaussian.positionOpacity.w = Sigmoid(getValue("opacity"));
        }

        if (propertyIndex.contains("scale_0") && propertyIndex.contains("scale_1") && propertyIndex.contains("scale_2")) {
            gaussian.scale = glm::vec4(
                std::exp(getValue("scale_0")),
                std::exp(getValue("scale_1")),
                std::exp(getValue("scale_2")),
                1.0f);
        } else {
            gaussian.scale = glm::vec4(1.0f);
        }

        if (propertyIndex.contains("rot_0") && propertyIndex.contains("rot_1") && propertyIndex.contains("rot_2")
            && propertyIndex.contains("rot_3")) {
            gaussian.rotation = glm::vec4(getValue("rot_1"), getValue("rot_2"), getValue("rot_3"), getValue("rot_0", 1.0f));
            const float length = glm::length(gaussian.rotation);
            gaussian.rotation = length > 1.0e-4f ? gaussian.rotation / length : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        if (propertyIndex.contains("f_dc_0") && propertyIndex.contains("f_dc_1") && propertyIndex.contains("f_dc_2")) {
            gaussian.shCoefficients[0] = glm::vec4(getValue("f_dc_0"), getValue("f_dc_1"), getValue("f_dc_2"), 0.0f);
        } else if (propertyIndex.contains("red") && propertyIndex.contains("green") && propertyIndex.contains("blue")) {
            const float red = getValue("red") > 1.0f ? getValue("red") / 255.0f : getValue("red");
            const float green = getValue("green") > 1.0f ? getValue("green") / 255.0f : getValue("green");
            const float blue = getValue("blue") > 1.0f ? getValue("blue") / 255.0f : getValue("blue");
            gaussian.shCoefficients[0] = glm::vec4(
                (red - 0.5f) / kShC0,
                (green - 0.5f) / kShC0,
                (blue - 0.5f) / kShC0,
                0.0f);
        }

        const uint32_t activeCoefficientCount = std::min<uint32_t>((parsedScene.gaussianShDegree + 1u) * (parsedScene.gaussianShDegree + 1u),
            kGaussianMaxShCoefficients);
        for (uint32_t coefficientIndex = 1; coefficientIndex < activeCoefficientCount; ++coefficientIndex) {
            const uint32_t restBaseIndex = (coefficientIndex - 1u) * 3u;
            gaussian.shCoefficients[coefficientIndex] = glm::vec4(
                getValue(fmt::format("f_rest_{}", restBaseIndex + 0u), 0.0f),
                getValue(fmt::format("f_rest_{}", restBaseIndex + 1u), 0.0f),
                getValue(fmt::format("f_rest_{}", restBaseIndex + 2u), 0.0f),
                0.0f);
        }

        return gaussian;
    };

    parsedScene.sceneKind = propertyIndex.contains("scale_0") && propertyIndex.contains("scale_1") && propertyIndex.contains("scale_2")
        ? SceneKind::Gaussian
        : SceneKind::PointCloud;
    parsedScene.gaussianUsesNativeScale =
        propertyIndex.contains("scale_0") && propertyIndex.contains("scale_1") && propertyIndex.contains("scale_2");
    parsedScene.gaussianVertices.clear();
    parsedScene.gaussianPrimitives.clear();
        parsedScene.gaussianVertices.reserve(layout.vertexCount);
        parsedScene.gaussianPrimitives.reserve(layout.vertexCount);

    if (layout.format == PlyFormat::Ascii) {
        std::ifstream input(*gaussianPath);
        if (!input.is_open()) {
            return false;
        }

        std::string line;
        bool headerEnded = false;
        while (std::getline(input, line)) {
            if (!headerEnded) {
                if (line == "end_header" || line == "end_header\r") {
                    headerEnded = true;
                }
                continue;
            }

            if (line.empty()) {
                continue;
            }
            std::istringstream stream(line);
            std::vector<float> values;
            values.reserve(layout.properties.size());
            for (size_t property = 0; property < layout.properties.size(); ++property) {
                float value = 0.0f;
                stream >> value;
                values.push_back(value);
            }
            parsedScene.gaussianVertices.push_back(readVertexFromValues(values));
            parsedScene.gaussianPrimitives.push_back(readGaussianFromValues(values));
        }
    } else {
        std::ifstream input(*gaussianPath, std::ios::binary);
        if (!input.is_open()) {
            return false;
        }
        input.seekg(static_cast<std::streamoff>(layout.headerBytes), std::ios::beg);

        size_t stride = 0;
        for (const PlyProperty& property : layout.properties) {
            stride += PlyScalarTypeSize(property.type);
        }
        if (stride == 0) {
            return false;
        }

        std::vector<size_t> propertyOffsets(layout.properties.size(), 0);
        size_t runningOffset = 0;
        for (size_t propertyIndexValue = 0; propertyIndexValue < layout.properties.size(); ++propertyIndexValue) {
            propertyOffsets[propertyIndexValue] = runningOffset;
            runningOffset += PlyScalarTypeSize(layout.properties[propertyIndexValue].type);
        }

        std::vector<std::byte> rowBytes(stride);
        const auto readValueFast = [&](int propertySlot, float fallback = 0.0f) {
            if (propertySlot < 0) {
                return fallback;
            }
            const size_t propertySlotIndex = static_cast<size_t>(propertySlot);
            return ReadPlyScalarAsFloat(rowBytes.data() + propertyOffsets[propertySlotIndex], layout.properties[propertySlotIndex].type);
        };

        const bool hasShColor = indices.fdc0 >= 0 && indices.fdc1 >= 0 && indices.fdc2 >= 0;
        const bool hasRgbColor = indices.red >= 0 && indices.green >= 0 && indices.blue >= 0;
        const bool hasScales = indices.scale0 >= 0 && indices.scale1 >= 0 && indices.scale2 >= 0;
        const bool hasRotation = indices.rot0 >= 0 && indices.rot1 >= 0 && indices.rot2 >= 0 && indices.rot3 >= 0;

        for (size_t vertexIndex = 0; vertexIndex < layout.vertexCount; ++vertexIndex) {
            input.read(reinterpret_cast<char*>(rowBytes.data()), static_cast<std::streamsize>(rowBytes.size()));
            if (!input) {
                return false;
            }

            SceneVertex vertex{};
            vertex.position = glm::vec3(readValueFast(indices.x), readValueFast(indices.y), readValueFast(indices.z));

            GaussianPrimitive gaussian{};
            gaussian.positionOpacity = glm::vec4(vertex.position, 1.0f);

            if (hasShColor) {
                const float fdc0 = readValueFast(indices.fdc0);
                const float fdc1 = readValueFast(indices.fdc1);
                const float fdc2 = readValueFast(indices.fdc2);
                vertex.color.r = glm::clamp(kShC0 * fdc0 + 0.5f, 0.0f, 1.0f);
                vertex.color.g = glm::clamp(kShC0 * fdc1 + 0.5f, 0.0f, 1.0f);
                vertex.color.b = glm::clamp(kShC0 * fdc2 + 0.5f, 0.0f, 1.0f);
                gaussian.shCoefficients[0] = glm::vec4(fdc0, fdc1, fdc2, 0.0f);
            } else if (hasRgbColor) {
                const float red = readValueFast(indices.red) > 1.0f ? readValueFast(indices.red) / 255.0f : readValueFast(indices.red);
                const float green =
                    readValueFast(indices.green) > 1.0f ? readValueFast(indices.green) / 255.0f : readValueFast(indices.green);
                const float blue = readValueFast(indices.blue) > 1.0f ? readValueFast(indices.blue) / 255.0f : readValueFast(indices.blue);
                vertex.color = glm::vec4(red, green, blue, 1.0f);
                gaussian.shCoefficients[0] = glm::vec4((red - 0.5f) / kShC0, (green - 0.5f) / kShC0, (blue - 0.5f) / kShC0, 0.0f);
            }

            if (indices.opacity >= 0) {
                const float opacity = Sigmoid(readValueFast(indices.opacity));
                vertex.splatParams.y = opacity;
                gaussian.positionOpacity.w = opacity;
            } else {
                vertex.splatParams.y = 1.0f;
            }

            if (hasScales) {
                const float sx = std::exp(readValueFast(indices.scale0));
                const float sy = std::exp(readValueFast(indices.scale1));
                const float sz = std::exp(readValueFast(indices.scale2));
                vertex.normal = glm::vec3(sx, sy, sz);
                vertex.splatParams.x = (sx + sy + sz) / 3.0f;
                gaussian.scale = glm::vec4(sx, sy, sz, 1.0f);
            } else {
                vertex.normal = glm::vec3(1.0f);
                vertex.splatParams.x = 0.28f;
                gaussian.scale = glm::vec4(1.0f);
            }

            if (hasRotation) {
                glm::vec4 rotation(
                    readValueFast(indices.rot1), readValueFast(indices.rot2), readValueFast(indices.rot3), readValueFast(indices.rot0, 1.0f));
                const float rotationLength = glm::length(rotation);
                rotation = rotationLength > 1.0e-4f ? rotation / rotationLength : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                vertex.tangent = rotation;
                gaussian.rotation = rotation;
            } else {
                vertex.tangent = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            }

            for (uint32_t coefficientIndex = 1; coefficientIndex < activeCoefficientCount; ++coefficientIndex) {
                gaussian.shCoefficients[coefficientIndex] = glm::vec4(
                    readValueFast(indices.fRest[coefficientIndex][0], 0.0f),
                    readValueFast(indices.fRest[coefficientIndex][1], 0.0f),
                    readValueFast(indices.fRest[coefficientIndex][2], 0.0f),
                    0.0f);
            }

            vertex.color.a = 1.0f;
            parsedScene.gaussianVertices.push_back(vertex);
            parsedScene.gaussianPrimitives.push_back(gaussian);
        }
    }

    if (!parsedScene.gaussianVertices.empty()) {
        parsedScene.objects.push_back(ParsedSceneObject{
            .name = path.stem().string(),
            .initialWorldTransform = glm::mat4(1.0f),
            .worldTransform = glm::mat4(1.0f),
            .firstPrimitive = 0,
            .primitiveCount = 0,
        });
    }

    return !parsedScene.gaussianVertices.empty();
}

template <typename T>
vesta::render::BufferHandle CreateHostBufferAndCopy(vesta::render::RenderDevice& device,
    std::span<const T> data,
    VkBufferUsageFlags usage,
    std::string debugName,
    bool registerBindlessStorage)
{
    vesta::render::BufferHandle buffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = sizeof(T) * data.size(),
        .usage = usage,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .registerBindlessStorage = registerBindlessStorage,
        .debugName = std::move(debugName),
    });
    CopyToMappedBuffer(device, buffer, data);
    return buffer;
}

template <typename T>
std::span<const std::byte> AsBytes(std::span<const T> data)
{
    return std::as_bytes(data);
}

uint32_t RemapTextureIndex(uint32_t textureIndex, std::span<const GpuSceneTexture> textures)
{
    if (textureIndex == render::kInvalidResourceIndex || textureIndex >= textures.size()) {
        return render::kInvalidResourceIndex;
    }

    return textures[textureIndex].bindlessSampledImage;
}

void RemapMaterialTextures(SceneMaterial& material, std::span<const GpuSceneTexture> textures)
{
    material.textureIndices0.x = RemapTextureIndex(material.textureIndices0.x, textures);
    material.textureIndices0.y = RemapTextureIndex(material.textureIndices0.y, textures);
    material.textureIndices0.z = RemapTextureIndex(material.textureIndices0.z, textures);
    material.textureIndices0.w = RemapTextureIndex(material.textureIndices0.w, textures);
    material.textureIndices1.x = RemapTextureIndex(material.textureIndices1.x, textures);
}

void RemapTriangleTextures(SceneTriangle& triangle, std::span<const GpuSceneTexture> textures)
{
    triangle.textureIndices0.x = RemapTextureIndex(triangle.textureIndices0.x, textures);
    triangle.textureIndices0.y = RemapTextureIndex(triangle.textureIndices0.y, textures);
    triangle.textureIndices0.z = RemapTextureIndex(triangle.textureIndices0.z, textures);
    triangle.textureIndices0.w = RemapTextureIndex(triangle.textureIndices0.w, textures);
    triangle.textureIndices1.x = RemapTextureIndex(triangle.textureIndices1.x, textures);
}
} // namespace

const PreparedScene& Scene::EmptyPreparedScene()
{
    static const PreparedScene emptyScene;
    return emptyScene;
}

const GpuScene& Scene::EmptyGpuScene()
{
    static const GpuScene emptyScene;
    return emptyScene;
}

const PreparedScene& Scene::GetPreparedOrEmpty() const
{
    return _prepared != nullptr ? *_prepared : EmptyPreparedScene();
}

const GpuScene& Scene::GetGpuOrEmpty() const
{
    return _gpu != nullptr ? *_gpu : EmptyGpuScene();
}

bool Scene::LoadFromFile(const std::filesystem::path& path)
{
    return ParseFromFile(path) && PrepareParsedScene();
}

bool Scene::ParseFromFile(const std::filesystem::path& path)
{
    _parsed.reset();
    _prepared.reset();

    if (!std::filesystem::exists(path)) {
        return false;
    }

    auto parsed = std::make_shared<ParsedScene>();
    ParsedScene& parsedScene = *parsed;
    parsedScene.sourcePath = path;
    parsedScene.sceneKind = SceneKind::Empty;

    if (std::filesystem::is_directory(path) || path.extension() == ".ply" || path.extension() == ".PLY") {
        if (!ParseGaussianPly(path, parsedScene)) {
            return false;
        }
        _parsed = std::move(parsed);
        return _parsed->IsLoaded();
    }

    fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization);
    fastgltf::GltfDataBuffer data;
    if (!data.loadFromFile(path)) {
        return false;
    }

    const fastgltf::GltfType type = fastgltf::determineGltfFileType(&data);
    std::optional<fastgltf::Expected<fastgltf::Asset>> asset;
    if (type == fastgltf::GltfType::GLB) {
        asset.emplace(parser.loadBinaryGLTF(&data, path.parent_path(), kLoadOptions));
    } else if (type == fastgltf::GltfType::glTF) {
        asset.emplace(parser.loadGLTF(&data, path.parent_path(), kLoadOptions));
    } else {
        return false;
    }

    if (!asset.has_value() || asset->error() != fastgltf::Error::None) {
        return false;
    }

    const fastgltf::Asset& gltf = asset->get();
    if (gltf.scenes.empty()) {
        return false;
    }

    std::unordered_map<uint64_t, uint32_t> textureCache;
    auto resolveSceneTextureIndex = [&](size_t gltfTextureIndex, bool srgb) -> uint32_t {
        if (gltfTextureIndex >= gltf.textures.size()) {
            return render::kInvalidResourceIndex;
        }

        const fastgltf::Texture& texture = gltf.textures.at(gltfTextureIndex);
        if (!texture.imageIndex.has_value()) {
            return render::kInvalidResourceIndex;
        }

        const size_t imageIndex = texture.imageIndex.value();
        if (imageIndex >= gltf.images.size()) {
            return render::kInvalidResourceIndex;
        }

        const uint64_t cacheKey = (static_cast<uint64_t>(imageIndex) << 1u) | (srgb ? 1ull : 0ull);
        if (const auto it = textureCache.find(cacheKey); it != textureCache.end()) {
            return it->second;
        }

        const std::optional<SceneTextureAsset> decodedTexture = DecodeTextureAsset(gltf, gltf.images.at(imageIndex), path, srgb);
        if (!decodedTexture.has_value()) {
            return render::kInvalidResourceIndex;
        }

        const uint32_t mappedTextureIndex = static_cast<uint32_t>(parsedScene.textures.size());
        parsedScene.textures.push_back(*decodedTexture);
        textureCache.emplace(cacheKey, mappedTextureIndex);
        return mappedTextureIndex;
    };

    auto resolveTextureSlot = [&](const auto& textureInfo, bool srgb) -> uint32_t {
        if (!textureInfo.has_value()) {
            return render::kInvalidResourceIndex;
        }
        return resolveSceneTextureIndex(textureInfo->textureIndex, srgb);
    };

    parsedScene.materials.reserve(gltf.materials.size() + 1);
    for (const fastgltf::Material& material : gltf.materials) {
        SceneMaterial sceneMaterial = MakeDefaultMaterial();
        sceneMaterial.baseColorFactor = glm::vec4(material.pbrData.baseColorFactor[0],
            material.pbrData.baseColorFactor[1],
            material.pbrData.baseColorFactor[2],
            material.pbrData.baseColorFactor[3]);
        sceneMaterial.emissiveFactor =
            glm::vec4(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2], 0.0f);
        sceneMaterial.materialParams = glm::vec4(material.pbrData.metallicFactor,
            material.pbrData.roughnessFactor,
            material.occlusionTexture.has_value() ? material.occlusionTexture->strength : 1.0f,
            material.normalTexture.has_value() ? material.normalTexture->scale : 1.0f);
        sceneMaterial.textureIndices0 = glm::uvec4(resolveTextureSlot(material.pbrData.baseColorTexture, true),
            resolveTextureSlot(material.pbrData.metallicRoughnessTexture, false),
            resolveTextureSlot(material.normalTexture, false),
            resolveTextureSlot(material.occlusionTexture, false));
        sceneMaterial.textureIndices1 =
            glm::uvec4(resolveTextureSlot(material.emissiveTexture, true), render::kInvalidResourceIndex, render::kInvalidResourceIndex, render::kInvalidResourceIndex);
        parsedScene.materials.push_back(sceneMaterial);
    }
    const uint32_t defaultMaterialIndex = static_cast<uint32_t>(parsedScene.materials.size());
    parsedScene.materials.push_back(MakeDefaultMaterial());

    const size_t sceneIndex = gltf.defaultScene.value_or(0);
    const fastgltf::Scene& rootScene = gltf.scenes.at(sceneIndex);

    std::function<void(size_t, const glm::mat4&)> appendNode = [&](size_t nodeIndex, const glm::mat4& parentMatrix) {
        const fastgltf::Node& node = gltf.nodes.at(nodeIndex);
        const glm::mat4 worldMatrix = parentMatrix * NodeToMatrix(node);

        if (node.meshIndex.has_value()) {
            const uint32_t objectIndex = static_cast<uint32_t>(parsedScene.objects.size());
            parsedScene.objects.push_back(ParsedSceneObject{
                .name = node.name.empty() ? fmt::format("Object {}", objectIndex) : std::string(node.name),
                .initialWorldTransform = worldMatrix,
                .worldTransform = worldMatrix,
                .firstPrimitive = static_cast<uint32_t>(parsedScene.primitives.size()),
                .primitiveCount = 0,
            });

            const fastgltf::Mesh& mesh = gltf.meshes.at(node.meshIndex.value());
            for (const fastgltf::Primitive& primitive : mesh.primitives) {
                const auto positionAttribute = primitive.findAttribute("POSITION");
                if (positionAttribute == primitive.attributes.end()) {
                    continue;
                }

                const fastgltf::Accessor& positionAccessor = gltf.accessors.at(positionAttribute->second);
                std::vector<glm::vec3> positions = ReadVec3Accessor(gltf, positionAccessor);

                std::vector<glm::vec3> normals(positions.size(), glm::vec3(0.0f, 1.0f, 0.0f));
                const auto normalAttribute = primitive.findAttribute("NORMAL");
                const bool hasNormals = normalAttribute != primitive.attributes.end();
                if (hasNormals) {
                    normals = ReadVec3Accessor(gltf, gltf.accessors.at(normalAttribute->second));
                }

                std::vector<glm::vec2> texCoords(positions.size(), glm::vec2(0.0f));
                const auto texCoordAttribute = primitive.findAttribute("TEXCOORD_0");
                if (texCoordAttribute != primitive.attributes.end()) {
                    texCoords = ReadVec2Accessor(gltf, gltf.accessors.at(texCoordAttribute->second));
                }

                std::vector<glm::vec4> tangents(positions.size(), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
                const auto tangentAttribute = primitive.findAttribute("TANGENT");
                const bool hasTangents = tangentAttribute != primitive.attributes.end();
                if (hasTangents) {
                    tangents = ReadVec4Accessor(gltf, gltf.accessors.at(tangentAttribute->second));
                }

                std::vector<uint32_t> primitiveIndices;
                if (primitive.indicesAccessor.has_value()) {
                    primitiveIndices = ReadIndexAccessor(gltf, gltf.accessors.at(primitive.indicesAccessor.value()));
                } else {
                    primitiveIndices.resize(positions.size());
                    for (uint32_t index = 0; index < static_cast<uint32_t>(primitiveIndices.size()); ++index) {
                        primitiveIndices[index] = index;
                    }
                }

                if (!hasTangents) {
                    tangents = GenerateTangents(positions, normals, texCoords, primitiveIndices);
                }

                const uint32_t materialIndex = primitive.materialIndex.has_value()
                    ? static_cast<uint32_t>(primitive.materialIndex.value())
                    : defaultMaterialIndex;
                parsedScene.primitives.push_back(ParsedPrimitive{
                    .positions = std::move(positions),
                    .normals = std::move(normals),
                    .tangents = std::move(tangents),
                    .texCoords = std::move(texCoords),
                    .indices = std::move(primitiveIndices),
                    .worldTransform = worldMatrix,
                    .objectIndex = objectIndex,
                    .materialIndex = materialIndex,
                    .hasNormals = hasNormals,
                    .hasTangents = hasTangents,
                });
                parsedScene.objects.back().primitiveCount += 1;
            }
        }

        for (size_t child : node.children) {
            appendNode(child, worldMatrix);
        }
    };

    const bool autoLayoutDemoScene = ShouldAutoLayoutDemoScene(path, rootScene);
    for (size_t rootIndex = 0; rootIndex < rootScene.nodeIndices.size(); ++rootIndex) {
        glm::mat4 rootTransform(1.0f);
        if (autoLayoutDemoScene) {
            rootTransform = MakeDemoRootLayoutTransform(rootIndex, rootScene.nodeIndices.size());
        }

        appendNode(rootScene.nodeIndices[rootIndex], rootTransform);
    }
    if (!parsedScene.primitives.empty()) {
        parsedScene.sceneKind = SceneKind::Mesh;
    }
    _parsed = std::move(parsed);
    return _parsed->IsLoaded();
}

bool Scene::PrepareParsedScene()
{
    _prepared.reset();
    const std::shared_ptr<ParsedScene> parsed = _parsed;
    if (!parsed || !parsed->IsLoaded()) {
        return false;
    }

    auto prepared = std::make_shared<PreparedScene>();
    PreparedScene& sceneData = *prepared;
    sceneData.sceneKind = parsed->sceneKind;
    sceneData.materials = parsed->materials;
    sceneData.gaussianShDegree = parsed->gaussianShDegree;
    sceneData.objects.resize(parsed->objects.size());
    for (size_t objectIndex = 0; objectIndex < parsed->objects.size(); ++objectIndex) {
        const ParsedSceneObject& parsedObject = parsed->objects[objectIndex];
        sceneData.objects[objectIndex] = SceneObject{
            .name = parsedObject.name,
            .initialWorldTransform = parsedObject.initialWorldTransform,
            .worldTransform = parsedObject.worldTransform,
            .firstPrimitive = parsedObject.firstPrimitive,
            .primitiveCount = parsedObject.primitiveCount,
        };
    }

    if (parsed->sceneKind == SceneKind::Gaussian || parsed->sceneKind == SceneKind::PointCloud) {
        sceneData.vertices = parsed->gaussianVertices;
        sceneData.gaussians = parsed->gaussianPrimitives;
        if (!sceneData.objects.empty()) {
            sceneData.objects.front().firstVertex = 0;
            sceneData.objects.front().vertexCount = static_cast<uint32_t>(sceneData.vertices.size());
        }
    } else {
        for (const ParsedPrimitive& primitive : parsed->primitives) {
            const uint32_t baseVertex = static_cast<uint32_t>(sceneData.vertices.size());
            const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(primitive.worldTransform)));
            const glm::mat3 tangentMatrix = glm::mat3(primitive.worldTransform);
            const SceneMaterial& material = sceneData.materials.at(primitive.materialIndex);
            SceneObject& sceneObject = sceneData.objects.at(primitive.objectIndex);

            if (sceneObject.vertexCount == 0) {
                sceneObject.firstVertex = baseVertex;
                sceneObject.firstSurface = static_cast<uint32_t>(sceneData.surfaces.size());
                sceneObject.firstTriangle = static_cast<uint32_t>(sceneData.triangles.size());
            }

            sceneData.vertices.reserve(sceneData.vertices.size() + primitive.positions.size());
            for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
                const glm::vec3 worldPosition =
                    glm::vec3(primitive.worldTransform * glm::vec4(primitive.positions[vertexIndex], 1.0f));
                glm::vec3 worldNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                if (vertexIndex < primitive.normals.size()) {
                    worldNormal = glm::normalize(normalMatrix * primitive.normals[vertexIndex]);
                }

                const bool normalFinite =
                    std::isfinite(worldNormal.x) && std::isfinite(worldNormal.y) && std::isfinite(worldNormal.z);
                if (!normalFinite || glm::length(worldNormal) < 0.001f) {
                    worldNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                }

                glm::vec4 tangent = vertexIndex < primitive.tangents.size() ? primitive.tangents[vertexIndex] : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                glm::vec3 worldTangent = tangentMatrix * glm::vec3(tangent);
                worldTangent = worldTangent - worldNormal * glm::dot(worldNormal, worldTangent);
                const bool tangentFinite =
                    std::isfinite(worldTangent.x) && std::isfinite(worldTangent.y) && std::isfinite(worldTangent.z);
                if (!tangentFinite || glm::length(worldTangent) < 0.001f) {
                    worldTangent = std::abs(worldNormal.y) > 0.99f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                                                   : glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), worldNormal));
                    tangent.w = 1.0f;
                } else {
                    worldTangent = glm::normalize(worldTangent);
                }

                sceneData.vertices.push_back(SceneVertex{
                    .position = worldPosition,
                    .normal = worldNormal,
                    .tangent = glm::vec4(worldTangent, tangent.w),
                    .color = material.baseColorFactor,
                    .texCoord = vertexIndex < primitive.texCoords.size() ? primitive.texCoords[vertexIndex] : glm::vec2(0.0f),
                    .splatParams = glm::vec2(1.0f, material.baseColorFactor.a),
                    .materialIndex = primitive.materialIndex,
                });
            }
            sceneObject.vertexCount += static_cast<uint32_t>(primitive.positions.size());

            if (!primitive.hasNormals) {
                for (size_t triangle = 0; triangle + 2 < primitive.indices.size(); triangle += 3) {
                    const uint32_t i0 = baseVertex + primitive.indices[triangle + 0];
                    const uint32_t i1 = baseVertex + primitive.indices[triangle + 1];
                    const uint32_t i2 = baseVertex + primitive.indices[triangle + 2];
                    const glm::vec3 faceNormal = glm::normalize(
                        glm::cross(sceneData.vertices[i1].position - sceneData.vertices[i0].position,
                            sceneData.vertices[i2].position - sceneData.vertices[i0].position));
                    sceneData.vertices[i0].normal = faceNormal;
                    sceneData.vertices[i1].normal = faceNormal;
                    sceneData.vertices[i2].normal = faceNormal;
                }
            }

            const uint32_t firstIndex = static_cast<uint32_t>(sceneData.indices.size());
            sceneData.indices.reserve(sceneData.indices.size() + primitive.indices.size());
            for (uint32_t index : primitive.indices) {
                sceneData.indices.push_back(baseVertex + index);
            }

            sceneData.surfaces.push_back(SceneSurface{
                .firstIndex = firstIndex,
                .indexCount = static_cast<uint32_t>(primitive.indices.size()),
            });
            sceneData.surfaceBounds.push_back(ComputeSurfaceBounds(sceneData.vertices, baseVertex, primitive.indices));
            sceneObject.surfaceCount += 1;

            for (size_t triangle = 0; triangle + 2 < primitive.indices.size(); triangle += 3) {
                const SceneVertex& v0 = sceneData.vertices[baseVertex + primitive.indices[triangle + 0]];
                const SceneVertex& v1 = sceneData.vertices[baseVertex + primitive.indices[triangle + 1]];
                const SceneVertex& v2 = sceneData.vertices[baseVertex + primitive.indices[triangle + 2]];
                sceneData.triangles.push_back(SceneTriangle{
                    .p0 = glm::vec4(v0.position, 1.0f),
                    .p1 = glm::vec4(v1.position, 1.0f),
                    .p2 = glm::vec4(v2.position, 1.0f),
                    .n0 = glm::vec4(glm::normalize(v0.normal), 0.0f),
                    .n1 = glm::vec4(glm::normalize(v1.normal), 0.0f),
                    .n2 = glm::vec4(glm::normalize(v2.normal), 0.0f),
                    .uv0 = glm::vec4(v0.texCoord, 0.0f, 0.0f),
                    .uv1 = glm::vec4(v1.texCoord, 0.0f, 0.0f),
                    .uv2 = glm::vec4(v2.texCoord, 0.0f, 0.0f),
                    .baseColorFactor = material.baseColorFactor,
                    .emissiveFactor = material.emissiveFactor,
                    .materialParams = material.materialParams,
                    .textureIndices0 = material.textureIndices0,
                    .textureIndices1 = material.textureIndices1,
                });
            }
            sceneObject.triangleCount += static_cast<uint32_t>(primitive.indices.size() / 3);
        }
    }

    FinalizeBounds(sceneData.bounds, sceneData.vertices);
    if (parsed->sceneKind == SceneKind::Gaussian || parsed->sceneKind == SceneKind::PointCloud) {
        if (!sceneData.objects.empty()) {
            sceneData.objects.front().bounds = sceneData.bounds;
        }
    } else {
        for (SceneObject& object : sceneData.objects) {
            object.bounds = ComputeVertexRangeBounds(sceneData.vertices, object.firstVertex, object.vertexCount);
        }
    }

    if ((parsed->sceneKind == SceneKind::Gaussian || parsed->sceneKind == SceneKind::PointCloud) && sceneData.bounds.radius > 0.0f) {
        const float gaussianSceneRadius = sceneData.bounds.radius;
        if (parsed->gaussianUsesNativeScale) {
            for (size_t gaussianIndex = 0; gaussianIndex < sceneData.vertices.size(); ++gaussianIndex) {
                SceneVertex& vertex = sceneData.vertices[gaussianIndex];
                vertex.position = ApplyGaussianImportTransform(vertex.position);
                vertex.normal = NormalizeGaussianScaleForScene(vertex.normal, gaussianSceneRadius);
                vertex.splatParams.x = (vertex.normal.x + vertex.normal.y + vertex.normal.z) / 3.0f;
                vertex.splatParams.y = glm::clamp(vertex.splatParams.y, 0.0f, 1.0f);
                if (gaussianIndex < sceneData.gaussians.size()) {
                    sceneData.gaussians[gaussianIndex].positionOpacity =
                        glm::vec4(ApplyGaussianImportTransform(glm::vec3(sceneData.gaussians[gaussianIndex].positionOpacity)),
                            sceneData.gaussians[gaussianIndex].positionOpacity.w);
                    sceneData.gaussians[gaussianIndex].rotation =
                        ApplyGaussianImportTransform(sceneData.gaussians[gaussianIndex].rotation);
                    sceneData.gaussians[gaussianIndex].scale =
                        glm::vec4(NormalizeGaussianScaleForScene(glm::vec3(sceneData.gaussians[gaussianIndex].scale), gaussianSceneRadius), 1.0f);
                    sceneData.gaussians[gaussianIndex].positionOpacity.w = vertex.splatParams.y;
                    vertex.tangent = sceneData.gaussians[gaussianIndex].rotation;
                }
            }
        } else {
            const float pointCloudBaseSize =
                glm::clamp(120.0f / std::sqrt(static_cast<float>(std::max<size_t>(sceneData.vertices.size(), 1))), 0.16f, 0.42f);
            for (size_t gaussianIndex = 0; gaussianIndex < sceneData.vertices.size(); ++gaussianIndex) {
                SceneVertex& vertex = sceneData.vertices[gaussianIndex];
                vertex.normal = glm::vec3(1.0f);
                vertex.tangent = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                vertex.splatParams.x = pointCloudBaseSize;
                vertex.splatParams.y = glm::clamp(std::max(vertex.splatParams.y, 0.85f), 0.85f, 1.0f);
                if (gaussianIndex < sceneData.gaussians.size()) {
                    sceneData.gaussians[gaussianIndex].scale = glm::vec4(pointCloudBaseSize, pointCloudBaseSize, pointCloudBaseSize, 0.0f);
                    sceneData.gaussians[gaussianIndex].rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                    sceneData.gaussians[gaussianIndex].positionOpacity.w = vertex.splatParams.y;
                }
            }
        }

        sceneData.bounds = {};
        FinalizeBounds(sceneData.bounds, sceneData.vertices);
        if (!sceneData.objects.empty()) {
            sceneData.objects.front().bounds = sceneData.bounds;
        }
    }
    sceneData.sourcePath = parsed->sourcePath;
    sceneData.textures = parsed->textures;
    _prepared = std::move(prepared);
    ++_contentVersion;
    return IsLoaded();
}

void Scene::UploadToGpu(vesta::render::RenderDevice& device, const vesta::render::SceneUploadOptions& options)
{
    const auto geometryStart = std::chrono::steady_clock::now();
    AllocateGpuResources(device, options);
    const PreparedScene& prepared = GetPreparedOrEmpty();
    if (_gpu == nullptr || !prepared.IsLoaded()) {
        return;
    }

    if (options.useDeviceLocalSceneBuffers) {
        UploadGpuResourceChunk(device, SceneUploadResource::Vertex, 0, sizeof(SceneVertex) * prepared.vertices.size());
        if (!prepared.gaussians.empty()) {
            UploadGpuResourceChunk(device, SceneUploadResource::Gaussian, 0, sizeof(GaussianPrimitive) * prepared.gaussians.size());
        }
        if (!prepared.materials.empty()) {
            UploadGpuResourceChunk(device, SceneUploadResource::Material, 0, sizeof(SceneMaterial) * prepared.materials.size());
        }
        if (!prepared.indices.empty()) {
            UploadGpuResourceChunk(device, SceneUploadResource::Index, 0, sizeof(uint32_t) * prepared.indices.size());
        }
        if (!prepared.triangles.empty()) {
            UploadGpuResourceChunk(device, SceneUploadResource::Triangle, 0, sizeof(SceneTriangle) * prepared.triangles.size());
        }
        device.FlushUploadBatch();
    }
    _gpu->geometryUploadMs =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - geometryStart).count();

    _gpu->textureUploadMs = 0.0f;
    if (options.textureStreamingEnabled) {
        const auto textureStart = std::chrono::steady_clock::now();
        for (size_t textureIndex = 0; textureIndex < prepared.textures.size(); ++textureIndex) {
            UploadGpuTexture(device, textureIndex);
        }
        _gpu->textureUploadMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - textureStart).count();
    }

    if (options.buildRayTracingStructuresOnLoad && device.IsRayTracingSupported() && !prepared.indices.empty()) {
        const auto bottomLevelStart = std::chrono::steady_clock::now();
        BuildBottomLevelAccelerationStructure(device);
        _gpu->bottomLevelBuildMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - bottomLevelStart).count();

        const auto topLevelStart = std::chrono::steady_clock::now();
        BuildTopLevelAccelerationStructure(device);
        _gpu->topLevelBuildMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - topLevelStart).count();
    }
}

void Scene::AllocateGpuResources(vesta::render::RenderDevice& device, const vesta::render::SceneUploadOptions& options)
{
    const PreparedScene& prepared = GetPreparedOrEmpty();
    if (!prepared.IsLoaded()) {
        return;
    }

    DestroyGpu(device);
    _gpu = std::make_unique<GpuScene>();
    GpuScene& gpu = *_gpu;

    const VkBufferUsageFlags vertexUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkBufferUsageFlags gaussianUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags indexUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkBufferUsageFlags triangleUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags materialUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    gpu.textures.resize(prepared.textures.size());
    if (options.textureStreamingEnabled) {
        const VmaMemoryUsage textureMemoryUsage =
            options.useDeviceLocalTextures ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        for (size_t textureIndex = 0; textureIndex < prepared.textures.size(); ++textureIndex) {
            const SceneTextureAsset& texture = prepared.textures[textureIndex];
            if (!texture.IsValid()) {
                continue;
            }

            gpu.textures[textureIndex].image = device.CreateImage(vesta::render::ImageDesc{
                .extent = VkExtent3D{ texture.width, texture.height, 1 },
                .format = texture.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
                .memoryUsage = textureMemoryUsage,
                .registerBindlessSampled = true,
                .debugName = texture.name.empty() ? "SceneBaseColorTexture" : texture.name,
            });
            gpu.textures[textureIndex].bindlessSampledImage =
                device.GetImageResource(gpu.textures[textureIndex].image).bindless.sampledImage;
        }
    }

    gpu.rasterVertices = prepared.vertices;
    gpu.gaussians = prepared.gaussians;
    gpu.triangles = prepared.triangles;
    gpu.materials = prepared.materials;
    if (!options.textureStreamingEnabled) {
        for (SceneMaterial& material : gpu.materials) {
            material.textureIndices0 = glm::uvec4(render::kInvalidResourceIndex);
            material.textureIndices1 = glm::uvec4(render::kInvalidResourceIndex);
        }
        for (SceneTriangle& triangle : gpu.triangles) {
            triangle.textureIndices0 = glm::uvec4(render::kInvalidResourceIndex);
            triangle.textureIndices1 = glm::uvec4(render::kInvalidResourceIndex);
        }
    } else {
        for (SceneMaterial& material : gpu.materials) {
            RemapMaterialTextures(material, gpu.textures);
        }
        for (SceneTriangle& triangle : gpu.triangles) {
            RemapTriangleTextures(triangle, gpu.textures);
        }
    }

    if (options.useDeviceLocalSceneBuffers) {
        if (!gpu.rasterVertices.empty()) {
            gpu.vertexBuffer = device.CreateBuffer(vesta::render::BufferDesc{
                .size = sizeof(SceneVertex) * gpu.rasterVertices.size(),
                .usage = vertexUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .registerBindlessStorage = false,
                .debugName = "SceneVertices",
            });
        }
        if (!gpu.gaussians.empty()) {
            gpu.gaussianBuffer = device.CreateBuffer(vesta::render::BufferDesc{
                .size = sizeof(GaussianPrimitive) * gpu.gaussians.size(),
                .usage = gaussianUsage,
                .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .registerBindlessStorage = true,
                .debugName = "SceneGaussians",
            });
        }
        if (!prepared.indices.empty()) {
            gpu.indexBuffer = device.CreateBuffer(vesta::render::BufferDesc{
                .size = sizeof(uint32_t) * prepared.indices.size(),
                .usage = indexUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .registerBindlessStorage = false,
                .debugName = "SceneIndices",
            });
        }
        if (!prepared.triangles.empty()) {
            gpu.triangleBuffer = device.CreateBuffer(vesta::render::BufferDesc{
                .size = sizeof(SceneTriangle) * gpu.triangles.size(),
                .usage = triangleUsage,
                .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .registerBindlessStorage = true,
                .debugName = "SceneTriangles",
            });
        }
        if (!gpu.materials.empty()) {
            gpu.materialBuffer = device.CreateBuffer(vesta::render::BufferDesc{
                .size = sizeof(SceneMaterial) * gpu.materials.size(),
                .usage = materialUsage,
                .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .registerBindlessStorage = true,
                .debugName = "SceneMaterials",
            });
        }
    } else {
        if (!gpu.rasterVertices.empty()) {
            gpu.vertexBuffer =
                CreateHostBufferAndCopy(device, std::span<const SceneVertex>(gpu.rasterVertices), vertexUsage, "SceneVertices", false);
        }
        if (!gpu.gaussians.empty()) {
            gpu.gaussianBuffer = CreateHostBufferAndCopy(
                device, std::span<const GaussianPrimitive>(gpu.gaussians), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "SceneGaussians", true);
        }
        if (!prepared.indices.empty()) {
            gpu.indexBuffer =
                CreateHostBufferAndCopy(device, std::span<const uint32_t>(prepared.indices), indexUsage, "SceneIndices", false);
        }
        if (!prepared.triangles.empty()) {
            gpu.triangleBuffer = CreateHostBufferAndCopy(
                device, std::span<const SceneTriangle>(gpu.triangles), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "SceneTriangles", true);
        }
        if (!gpu.materials.empty()) {
            gpu.materialBuffer = CreateHostBufferAndCopy(
                device, std::span<const SceneMaterial>(gpu.materials), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "SceneMaterials", true);
        }
    }
}

void Scene::UploadGpuResourceChunk(
    vesta::render::RenderDevice& device, SceneUploadResource resource, size_t offsetBytes, size_t sizeBytes)
{
    if (_gpu == nullptr || sizeBytes == 0) {
        return;
    }

    const PreparedScene& prepared = GetPreparedOrEmpty();
    std::span<const std::byte> sourceBytes;
    vesta::render::BufferHandle destinationBuffer;
    switch (resource) {
    case SceneUploadResource::Vertex:
        sourceBytes = AsBytes(std::span<const SceneVertex>(GetGpuOrEmpty().rasterVertices));
        destinationBuffer = _gpu->vertexBuffer;
        break;
    case SceneUploadResource::Gaussian:
        sourceBytes = AsBytes(std::span<const GaussianPrimitive>(GetGpuOrEmpty().gaussians));
        destinationBuffer = _gpu->gaussianBuffer;
        break;
    case SceneUploadResource::Material:
        sourceBytes = AsBytes(std::span<const SceneMaterial>(GetGpuOrEmpty().materials));
        destinationBuffer = _gpu->materialBuffer;
        break;
    case SceneUploadResource::Index:
        sourceBytes = AsBytes(std::span<const uint32_t>(prepared.indices));
        destinationBuffer = _gpu->indexBuffer;
        break;
    case SceneUploadResource::Triangle:
    default:
        sourceBytes = AsBytes(std::span<const SceneTriangle>(GetGpuOrEmpty().triangles));
        destinationBuffer = _gpu->triangleBuffer;
        break;
    }

    if (!destinationBuffer || offsetBytes >= sourceBytes.size()) {
        VESTA_ASSERT_STATE(destinationBuffer || sourceBytes.empty(), "UploadGpuResourceChunk lost its destination buffer.");
        return;
    }

    const size_t clampedSize = std::min(sizeBytes, sourceBytes.size() - offsetBytes);
    VESTA_ASSERT_STATE(offsetBytes + clampedSize <= sourceBytes.size(), "UploadGpuResourceChunk exceeded source byte range.");
    const std::span<const std::byte> chunkBytes = sourceBytes.subspan(offsetBytes, clampedSize);
    device.UploadBufferData(destinationBuffer, static_cast<VkDeviceSize>(offsetBytes), chunkBytes);
}

void Scene::UploadGpuTexture(vesta::render::RenderDevice& device, size_t textureIndex)
{
    if (_gpu == nullptr) {
        return;
    }

    const PreparedScene& prepared = GetPreparedOrEmpty();
    if (textureIndex >= prepared.textures.size() || textureIndex >= _gpu->textures.size()) {
        return;
    }

    const SceneTextureAsset& texture = prepared.textures[textureIndex];
    GpuSceneTexture& gpuTexture = _gpu->textures[textureIndex];
    if (!texture.IsValid() || !gpuTexture.image || gpuTexture.resident) {
        return;
    }

    device.UploadImageData(gpuTexture.image,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(texture.rgba8Pixels.data()), texture.rgba8Pixels.size()));
    gpuTexture.resident = true;
}

void Scene::BuildBottomLevelAccelerationStructure(vesta::render::RenderDevice& device)
{
    if (_gpu == nullptr) {
        return;
    }

    const PreparedScene& prepared = GetPreparedOrEmpty();
    if (prepared.indices.empty() || !device.IsRayTracingSupported()) {
        return;
    }

    VESTA_ASSERT_STATE(_gpu->vertexBuffer && _gpu->indexBuffer, "BLAS build requires uploaded vertex and index buffers.");

    GpuScene& gpu = *_gpu;
    const auto& rt = device.GetRayTracingFunctions();

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = device.GetBufferDeviceAddress(gpu.vertexBuffer) + offsetof(SceneVertex, position);
    triangles.vertexStride = sizeof(SceneVertex);
    triangles.maxVertex = static_cast<uint32_t>(prepared.vertices.size());
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = device.GetBufferDeviceAddress(gpu.indexBuffer);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;

    const uint32_t primitiveCount = static_cast<uint32_t>(prepared.indices.size() / 3);
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR buildSizes{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    rt.vkGetAccelerationStructureBuildSizesKHR(device.GetDevice(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &buildSizes);

    gpu.bottomLevelBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = buildSizes.accelerationStructureSize,
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "SceneBLASBuffer",
    });

    VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    createInfo.size = buildSizes.accelerationStructureSize;
    createInfo.buffer = device.GetBuffer(gpu.bottomLevelBuffer);
    VK_CHECK(rt.vkCreateAccelerationStructureKHR(device.GetDevice(), &createInfo, nullptr, &gpu.bottomLevelAccelerationStructure));

    const vesta::render::BufferHandle scratchBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = buildSizes.buildScratchSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "SceneBLASScratch",
    });

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = gpu.bottomLevelAccelerationStructure;
    buildInfo.scratchData.deviceAddress = device.GetBufferDeviceAddress(scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = { &rangeInfo };

    device.ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
        rt.vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, rangeInfos);
    });

    device.DestroyBuffer(scratchBuffer);
}

void Scene::BuildTopLevelAccelerationStructure(vesta::render::RenderDevice& device)
{
    if (_gpu == nullptr || _gpu->bottomLevelAccelerationStructure == VK_NULL_HANDLE || !device.IsRayTracingSupported()) {
        return;
    }

    VESTA_ASSERT_STATE(_gpu->bottomLevelBuffer, "TLAS build requires a valid BLAS buffer.");

    GpuScene& gpu = *_gpu;
    const auto& rt = device.GetRayTracingFunctions();

    VkAccelerationStructureDeviceAddressInfoKHR blasAddressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR
    };
    blasAddressInfo.accelerationStructure = gpu.bottomLevelAccelerationStructure;
    const VkDeviceAddress blasAddress = rt.vkGetAccelerationStructureDeviceAddressKHR(device.GetDevice(), &blasAddressInfo);

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = MakeIdentityTransformMatrix();
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blasAddress;

    const render::BufferHandle instanceBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = sizeof(VkAccelerationStructureInstanceKHR),
        .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .debugName = "SceneTLASInstances",
    });
    CopyToMappedBuffer(device, instanceBuffer, std::span<const VkAccelerationStructureInstanceKHR>(&instance, 1));

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
    };
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = device.GetBufferDeviceAddress(instanceBuffer);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    const uint32_t primitiveCount = 1;
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR buildSizes{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    rt.vkGetAccelerationStructureBuildSizesKHR(device.GetDevice(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &buildSizes);

    gpu.topLevelBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = buildSizes.accelerationStructureSize,
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "SceneTLASBuffer",
    });

    VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    createInfo.size = buildSizes.accelerationStructureSize;
    createInfo.buffer = device.GetBuffer(gpu.topLevelBuffer);
    VK_CHECK(rt.vkCreateAccelerationStructureKHR(device.GetDevice(), &createInfo, nullptr, &gpu.topLevelAccelerationStructure));

    const vesta::render::BufferHandle scratchBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = buildSizes.buildScratchSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "SceneTLASScratch",
    });

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = gpu.topLevelAccelerationStructure;
    buildInfo.scratchData.deviceAddress = device.GetBufferDeviceAddress(scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = { &rangeInfo };

    device.ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
        rt.vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, rangeInfos);
    });

    device.DestroyBuffer(scratchBuffer);
    device.DestroyBuffer(instanceBuffer);
}

void Scene::DestroyGpu(vesta::render::RenderDevice& device)
{
    if (_gpu == nullptr) {
        return;
    }

    if (_gpu->topLevelAccelerationStructure != VK_NULL_HANDLE) {
        device.GetRayTracingFunctions().vkDestroyAccelerationStructureKHR(
            device.GetDevice(), _gpu->topLevelAccelerationStructure, nullptr);
        _gpu->topLevelAccelerationStructure = VK_NULL_HANDLE;
    }
    if (_gpu->bottomLevelAccelerationStructure != VK_NULL_HANDLE) {
        device.GetRayTracingFunctions().vkDestroyAccelerationStructureKHR(
            device.GetDevice(), _gpu->bottomLevelAccelerationStructure, nullptr);
        _gpu->bottomLevelAccelerationStructure = VK_NULL_HANDLE;
    }
    if (_gpu->topLevelBuffer) {
        device.DestroyBuffer(_gpu->topLevelBuffer);
        _gpu->topLevelBuffer = {};
    }
    if (_gpu->bottomLevelBuffer) {
        device.DestroyBuffer(_gpu->bottomLevelBuffer);
        _gpu->bottomLevelBuffer = {};
    }
    if (_gpu->triangleBuffer) {
        device.DestroyBuffer(_gpu->triangleBuffer);
        _gpu->triangleBuffer = {};
    }
    if (_gpu->materialBuffer) {
        device.DestroyBuffer(_gpu->materialBuffer);
        _gpu->materialBuffer = {};
    }
    if (_gpu->gaussianBuffer) {
        device.DestroyBuffer(_gpu->gaussianBuffer);
        _gpu->gaussianBuffer = {};
    }
    for (GpuSceneTexture& texture : _gpu->textures) {
        if (texture.image) {
            device.DestroyImage(texture.image);
            texture.image = {};
        }
        texture.bindlessSampledImage = render::kInvalidResourceIndex;
        texture.resident = false;
    }
    _gpu->textures.clear();
    if (_gpu->indexBuffer) {
        device.DestroyBuffer(_gpu->indexBuffer);
        _gpu->indexBuffer = {};
    }
    if (_gpu->vertexBuffer) {
        device.DestroyBuffer(_gpu->vertexBuffer);
        _gpu->vertexBuffer = {};
    }
    _gpu->triangles.clear();
    _gpu->materials.clear();
    _gpu->rasterVertices.clear();
    _gpu->gaussians.clear();
    _gpu.reset();
}

size_t Scene::GetResidentTextureCount() const
{
    const GpuScene& gpu = GetGpuOrEmpty();
    return std::count_if(gpu.textures.begin(), gpu.textures.end(), [](const GpuSceneTexture& texture) {
        return texture.resident;
    });
}

bool Scene::HasResidentTexture(size_t textureIndex) const
{
    const GpuScene& gpu = GetGpuOrEmpty();
    return textureIndex < gpu.textures.size() && gpu.textures[textureIndex].resident;
}

uint32_t Scene::GetTextureBindlessIndex(size_t textureIndex) const
{
    const GpuScene& gpu = GetGpuOrEmpty();
    if (textureIndex >= gpu.textures.size()) {
        return render::kInvalidResourceIndex;
    }
    return gpu.textures[textureIndex].bindlessSampledImage;
}

std::optional<uint32_t> Scene::PickObject(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const
{
    const PreparedScene& prepared = GetPreparedOrEmpty();
    if (prepared.objects.empty()) {
        return std::nullopt;
    }

    float closestDistance = std::numeric_limits<float>::max();
    std::optional<uint32_t> pickedObject;
    for (uint32_t objectIndex = 0; objectIndex < static_cast<uint32_t>(prepared.objects.size()); ++objectIndex) {
        const SceneObject& object = prepared.objects[objectIndex];
        float hitDistance = 0.0f;
        const float radius = std::max(object.bounds.radius, 0.15f);
        if (!IntersectRaySphere(rayOrigin, rayDirection, object.bounds.center, radius, hitDistance)) {
            continue;
        }
        if (hitDistance < closestDistance) {
            closestDistance = hitDistance;
            pickedObject = objectIndex;
        }
    }

    return pickedObject;
}

bool Scene::TranslateObject(render::RenderDevice& device, uint32_t objectIndex, const glm::vec3& deltaWorld)
{
    if (glm::dot(deltaWorld, deltaWorld) <= 1.0e-10f) {
        return true;
    }

    const std::shared_ptr<PreparedScene> prepared = _prepared;
    const std::shared_ptr<ParsedScene> parsed = _parsed;
    if (!prepared || objectIndex >= prepared->objects.size()) {
        return false;
    }

    SceneObject& object = prepared->objects[objectIndex];
    object.worldTransform[3] = glm::vec4(glm::vec3(object.worldTransform[3]) + deltaWorld, 1.0f);
    object.bounds.minimum += deltaWorld;
    object.bounds.maximum += deltaWorld;
    object.bounds.center += deltaWorld;

    if (parsed && objectIndex < parsed->objects.size()) {
        ParsedSceneObject& parsedObject = parsed->objects[objectIndex];
        parsedObject.worldTransform[3] = glm::vec4(glm::vec3(parsedObject.worldTransform[3]) + deltaWorld, 1.0f);
        const uint32_t primitiveEnd = parsedObject.firstPrimitive + parsedObject.primitiveCount;
        for (uint32_t primitiveIndex = parsedObject.firstPrimitive; primitiveIndex < primitiveEnd; ++primitiveIndex) {
            parsed->primitives[primitiveIndex].worldTransform[3] =
                glm::vec4(glm::vec3(parsed->primitives[primitiveIndex].worldTransform[3]) + deltaWorld, 1.0f);
        }
    }

    const uint32_t vertexEnd = std::min<uint32_t>(object.firstVertex + object.vertexCount, static_cast<uint32_t>(prepared->vertices.size()));
    for (uint32_t vertexIndex = object.firstVertex; vertexIndex < vertexEnd; ++vertexIndex) {
        prepared->vertices[vertexIndex].position += deltaWorld;
    }
    if (prepared->sceneKind == SceneKind::Gaussian || prepared->sceneKind == SceneKind::PointCloud) {
        for (GaussianPrimitive& gaussian : prepared->gaussians) {
            gaussian.positionOpacity += glm::vec4(deltaWorld, 0.0f);
        }
    }

    const uint32_t surfaceEnd =
        std::min<uint32_t>(object.firstSurface + object.surfaceCount, static_cast<uint32_t>(prepared->surfaceBounds.size()));
    for (uint32_t surfaceIndex = object.firstSurface; surfaceIndex < surfaceEnd; ++surfaceIndex) {
        prepared->surfaceBounds[surfaceIndex].center += deltaWorld;
    }

    const uint32_t triangleEnd =
        std::min<uint32_t>(object.firstTriangle + object.triangleCount, static_cast<uint32_t>(prepared->triangles.size()));
    for (uint32_t triangleIndex = object.firstTriangle; triangleIndex < triangleEnd; ++triangleIndex) {
        prepared->triangles[triangleIndex].p0 += glm::vec4(deltaWorld, 0.0f);
        prepared->triangles[triangleIndex].p1 += glm::vec4(deltaWorld, 0.0f);
        prepared->triangles[triangleIndex].p2 += glm::vec4(deltaWorld, 0.0f);
    }

    FinalizeBounds(prepared->bounds, prepared->vertices);

    if (_gpu == nullptr) {
        return true;
    }

    GpuScene& gpu = *_gpu;
    const uint32_t gpuVertexEnd =
        std::min<uint32_t>(object.firstVertex + object.vertexCount, static_cast<uint32_t>(gpu.rasterVertices.size()));
    for (uint32_t vertexIndex = object.firstVertex; vertexIndex < gpuVertexEnd; ++vertexIndex) {
        gpu.rasterVertices[vertexIndex].position += deltaWorld;
    }
    if (prepared->sceneKind == SceneKind::Gaussian || prepared->sceneKind == SceneKind::PointCloud) {
        for (GaussianPrimitive& gaussian : gpu.gaussians) {
            gaussian.positionOpacity += glm::vec4(deltaWorld, 0.0f);
        }
    }

    const uint32_t gpuTriangleEnd =
        std::min<uint32_t>(object.firstTriangle + object.triangleCount, static_cast<uint32_t>(gpu.triangles.size()));
    for (uint32_t triangleIndex = object.firstTriangle; triangleIndex < gpuTriangleEnd; ++triangleIndex) {
        gpu.triangles[triangleIndex].p0 += glm::vec4(deltaWorld, 0.0f);
        gpu.triangles[triangleIndex].p1 += glm::vec4(deltaWorld, 0.0f);
        gpu.triangles[triangleIndex].p2 += glm::vec4(deltaWorld, 0.0f);
    }

    if (gpu.vertexBuffer && object.vertexCount > 0) {
        const std::span<const std::byte> vertexBytes = AsBytes(std::span<const SceneVertex>(gpu.rasterVertices));
        const size_t offsetBytes = static_cast<size_t>(object.firstVertex) * sizeof(SceneVertex);
        const size_t sizeBytes = static_cast<size_t>(object.vertexCount) * sizeof(SceneVertex);
        device.UploadBufferData(gpu.vertexBuffer, offsetBytes, vertexBytes.subspan(offsetBytes, sizeBytes));
    }
    if (gpu.triangleBuffer && object.triangleCount > 0) {
        const std::span<const std::byte> triangleBytes = AsBytes(std::span<const SceneTriangle>(gpu.triangles));
        const size_t offsetBytes = static_cast<size_t>(object.firstTriangle) * sizeof(SceneTriangle);
        const size_t sizeBytes = static_cast<size_t>(object.triangleCount) * sizeof(SceneTriangle);
        device.UploadBufferData(gpu.triangleBuffer, offsetBytes, triangleBytes.subspan(offsetBytes, sizeBytes));
    }
    if (gpu.gaussianBuffer && !gpu.gaussians.empty()) {
        const std::span<const std::byte> gaussianBytes = AsBytes(std::span<const GaussianPrimitive>(gpu.gaussians));
        device.UploadBufferData(gpu.gaussianBuffer, 0, gaussianBytes);
    }
    device.FlushUploadBatch();
    ++_contentVersion;
    return true;
}

bool Scene::RebuildRayTracing(render::RenderDevice& device)
{
    if (_gpu == nullptr || !device.IsRayTracingSupported() || GetPreparedOrEmpty().indices.empty()) {
        return false;
    }

    device.WaitIdle();
    DestroyRayTracingResources(device, *_gpu);
    BuildBottomLevelAccelerationStructure(device);
    BuildTopLevelAccelerationStructure(device);
    return _gpu->topLevelAccelerationStructure != VK_NULL_HANDLE;
}

bool Scene::ResortGaussians(render::RenderDevice& device, const Camera& camera)
{
    if (_gpu == nullptr || !_gpu->gaussianBuffer || GetPreparedOrEmpty().sceneKind != SceneKind::Gaussian || _gpu->gaussians.empty()) {
        return false;
    }

    if (!SupportsRealtimeGaussianSorting()) {
        return false;
    }

    const glm::vec3 cameraPosition = camera.GetPosition();
    const glm::vec3 cameraForward = camera.GetForward();
    std::stable_sort(_gpu->gaussians.begin(), _gpu->gaussians.end(), [&](const GaussianPrimitive& lhs, const GaussianPrimitive& rhs) {
        const float lhsDepth = glm::dot(glm::vec3(lhs.positionOpacity) - cameraPosition, cameraForward);
        const float rhsDepth = glm::dot(glm::vec3(rhs.positionOpacity) - cameraPosition, cameraForward);
        return lhsDepth > rhsDepth;
    });

    const std::span<const std::byte> gaussianBytes = AsBytes(std::span<const GaussianPrimitive>(_gpu->gaussians));
    device.UploadBufferData(_gpu->gaussianBuffer, 0, gaussianBytes);
    device.FlushUploadBatch();
    return true;
}

bool Scene::SupportsRealtimeGaussianSorting() const
{
    return GetPreparedOrEmpty().sceneKind == SceneKind::Gaussian && GetGaussianCount() <= kRealtimeGaussianSortLimit;
}
} // namespace vesta::scene

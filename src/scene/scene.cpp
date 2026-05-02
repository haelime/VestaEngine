#include <vesta/scene/scene.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cctype>
#include <cstdlib>
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

std::string ToLowerExtension(std::filesystem::path path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension;
}

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

std::optional<SceneTextureAsset> DecodeTextureFile(const std::filesystem::path& imagePath, bool srgb)
{
    int width = 0;
    int height = 0;
    int channelCount = 0;
    stbi_uc* decodedPixels = stbi_load(imagePath.string().c_str(), &width, &height, &channelCount, STBI_rgb_alpha);
    if (decodedPixels == nullptr || width <= 0 || height <= 0) {
        if (decodedPixels != nullptr) {
            stbi_image_free(decodedPixels);
        }
        return std::nullopt;
    }

    SceneTextureAsset textureAsset;
    textureAsset.name = imagePath.filename().string();
    textureAsset.srgb = srgb;
    textureAsset.width = static_cast<uint32_t>(width);
    textureAsset.height = static_cast<uint32_t>(height);
    textureAsset.rgba8Pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    std::memcpy(textureAsset.rgba8Pixels.data(), decodedPixels, textureAsset.rgba8Pixels.size());
    stbi_image_free(decodedPixels);
    return textureAsset;
}

SceneMaterial MakeDefaultMaterial()
{
    return SceneMaterial{};
}

SceneMaterial MakeDefaultObjMaterial()
{
    SceneMaterial material = MakeDefaultMaterial();
    material.materialParams.x = 0.0f;
    return material;
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
bool ReadLittleEndian(std::span<const std::byte> bytes, size_t& offset, T& value)
{
    if (offset + sizeof(T) > bytes.size()) {
        return false;
    }

    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

struct FbxProperty {
    char type{ 0 };
    int16_t int16Value{ 0 };
    int32_t int32Value{ 0 };
    int64_t int64Value{ 0 };
    float floatValue{ 0.0f };
    double doubleValue{ 0.0 };
    bool boolValue{ false };
    std::string stringValue;
    std::vector<int32_t> int32Array;
    std::vector<int64_t> int64Array;
    std::vector<double> doubleArray;
};

struct FbxNode {
    std::string name;
    std::vector<FbxProperty> properties;
    std::vector<FbxNode> children;
};

const FbxNode* FindChildNode(const FbxNode& node, std::string_view name)
{
    for (const FbxNode& child : node.children) {
        if (child.name == name) {
            return &child;
        }
    }
    return nullptr;
}

int64_t FbxPropertyAsInt64(const FbxProperty& property)
{
    switch (property.type) {
    case 'Y':
        return property.int16Value;
    case 'I':
        return property.int32Value;
    case 'L':
        return property.int64Value;
    default:
        return 0;
    }
}

double FbxPropertyAsDouble(const FbxProperty& property)
{
    switch (property.type) {
    case 'F':
        return property.floatValue;
    case 'D':
        return property.doubleValue;
    case 'Y':
    case 'I':
    case 'L':
        return static_cast<double>(FbxPropertyAsInt64(property));
    default:
        return 0.0;
    }
}

std::string FbxObjectDisplayName(const FbxNode& node, std::string_view fallback)
{
    if (node.properties.size() < 2 || node.properties[1].stringValue.empty()) {
        return std::string(fallback);
    }

    std::string name = node.properties[1].stringValue;
    const size_t separator = name.find('\0');
    if (separator != std::string::npos) {
        name.resize(separator);
    }
    return name.empty() ? std::string(fallback) : name;
}

std::optional<glm::vec3> ReadFbxPropertyVec3(const FbxNode& propertyNode, std::string_view propertyName)
{
    if (propertyNode.name != "P" || propertyNode.properties.size() < 7 || propertyNode.properties[0].stringValue != propertyName) {
        return std::nullopt;
    }

    return glm::vec3(static_cast<float>(FbxPropertyAsDouble(propertyNode.properties[4])),
        static_cast<float>(FbxPropertyAsDouble(propertyNode.properties[5])),
        static_cast<float>(FbxPropertyAsDouble(propertyNode.properties[6])));
}

glm::mat4 FbxTransformFromProperties(const FbxNode& modelNode)
{
    glm::vec3 translation(0.0f);
    glm::vec3 rotationDegrees(0.0f);
    glm::vec3 scaling(1.0f);
    glm::vec3 geometricTranslation(0.0f);
    glm::vec3 geometricRotationDegrees(0.0f);
    glm::vec3 geometricScaling(1.0f);

    if (const FbxNode* properties = FindChildNode(modelNode, "Properties70")) {
        for (const FbxNode& property : properties->children) {
            if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "Lcl Translation")) {
                translation = *value;
            } else if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "Lcl Rotation")) {
                rotationDegrees = *value;
            } else if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "Lcl Scaling")) {
                scaling = *value;
            } else if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "GeometricTranslation")) {
                geometricTranslation = *value;
            } else if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "GeometricRotation")) {
                geometricRotationDegrees = *value;
            } else if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "GeometricScaling")) {
                geometricScaling = *value;
            }
        }
    }

    const glm::quat rotation = glm::quat(glm::radians(rotationDegrees));
    const glm::quat geometricRotation = glm::quat(glm::radians(geometricRotationDegrees));
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scaling)
        * glm::translate(glm::mat4(1.0f), geometricTranslation) * glm::mat4_cast(geometricRotation)
        * glm::scale(glm::mat4(1.0f), geometricScaling);
}

float FbxUnitScaleToMeters(const FbxNode& root)
{
    constexpr float kCentimetersToMeters = 0.01f;
    const FbxNode* globalSettings = FindChildNode(root, "GlobalSettings");
    const FbxNode* properties = globalSettings != nullptr ? FindChildNode(*globalSettings, "Properties70") : nullptr;
    if (properties == nullptr) {
        return kCentimetersToMeters;
    }

    for (const FbxNode& property : properties->children) {
        if (property.name == "P" && property.properties.size() >= 5 && property.properties[0].stringValue == "UnitScaleFactor") {
            return static_cast<float>(FbxPropertyAsDouble(property.properties[4])) * kCentimetersToMeters;
        }
    }
    return kCentimetersToMeters;
}

struct FbxModelInfo {
    std::string name;
    glm::mat4 transform{ 1.0f };
    uint32_t materialIndex{ 0 };
};

struct FbxTextureSet {
    std::string key;
    std::filesystem::path baseColor;
    std::filesystem::path normal;
    std::filesystem::path roughness;
};

struct FbxTextureInfo {
    std::filesystem::path path;
};

struct FbxMaterialInfo {
    std::string name;
    uint32_t materialIndex{ 0 };
    std::filesystem::path baseColorPath;
    std::filesystem::path normalPath;
    std::filesystem::path roughnessPath;
};

std::string ToLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string NormalizeTextureMatchKey(std::string_view value)
{
    std::string key;
    key.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            key.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return key;
}

bool HasTextureExtension(const std::filesystem::path& path)
{
    const std::string extension = ToLowerAscii(path.extension().string());
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".tga" || extension == ".bmp"
        || extension == ".webp";
}

bool ContainsAny(std::string_view value, std::initializer_list<std::string_view> needles)
{
    for (std::string_view needle : needles) {
        if (value.find(needle) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

std::filesystem::path FbxFilenameFromRawPath(std::string_view rawPath)
{
    const size_t separator = rawPath.find_last_of("/\\");
    const std::string_view filename = separator == std::string_view::npos ? rawPath : rawPath.substr(separator + 1);
    return std::filesystem::path(std::string(filename));
}

std::filesystem::path ResolveFbxTexturePath(const std::filesystem::path& fbxPath, const std::filesystem::path& rawTexturePath)
{
    if (rawTexturePath.empty()) {
        return {};
    }

    if (std::filesystem::exists(rawTexturePath)) {
        return rawTexturePath;
    }

    const std::filesystem::path localPath = fbxPath.parent_path() / rawTexturePath.filename();
    if (std::filesystem::exists(localPath)) {
        return localPath;
    }

    for (std::string_view textureFolder : { "Texture", "Textures", "texture", "textures" }) {
        const std::filesystem::path root = fbxPath.parent_path() / textureFolder;
        if (!std::filesystem::exists(root)) {
            continue;
        }

        const std::filesystem::path direct = root / rawTexturePath.filename();
        if (std::filesystem::exists(direct)) {
            return direct;
        }

        for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_regular_file() && entry.path().filename() == rawTexturePath.filename()) {
                return entry.path();
            }
        }
    }
    return {};
}

std::vector<FbxTextureSet> DiscoverFbxTextureSets(const std::filesystem::path& fbxPath)
{
    std::unordered_map<std::string, FbxTextureSet> sets;
    for (std::string_view textureFolder : { "Texture", "Textures", "texture", "textures" }) {
        const std::filesystem::path root = fbxPath.parent_path() / textureFolder;
        if (!std::filesystem::exists(root)) {
            continue;
        }

        for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file() || !HasTextureExtension(entry.path())) {
                continue;
            }

            const std::string filename = ToLowerAscii(entry.path().stem().string());
            const std::string key = NormalizeTextureMatchKey(entry.path().parent_path().filename().string());
            FbxTextureSet& set = sets[key.empty() ? "default" : key];
            set.key = key.empty() ? "default" : key;
            if (ContainsAny(filename, { "basecolor", "base_color", "diffuse", "albedo", "color" })) {
                set.baseColor = entry.path();
            } else if (ContainsAny(filename, { "normal", "normalmap", "bump" })) {
                set.normal = entry.path();
            } else if (ContainsAny(filename, { "roughness", "rough" })) {
                set.roughness = entry.path();
            }
        }
    }

    std::vector<FbxTextureSet> result;
    result.reserve(sets.size());
    for (auto& [_, set] : sets) {
        if (!set.baseColor.empty() || !set.normal.empty() || !set.roughness.empty()) {
            result.push_back(std::move(set));
        }
    }

    std::sort(result.begin(), result.end(), [](const FbxTextureSet& lhs, const FbxTextureSet& rhs) {
        return lhs.key < rhs.key;
    });
    return result;
}

const FbxTextureSet* SelectFbxTextureSet(std::span<const FbxTextureSet> textureSets, std::string_view name)
{
    if (textureSets.empty()) {
        return nullptr;
    }

    const std::string matchKey = NormalizeTextureMatchKey(name);
    for (const FbxTextureSet& set : textureSets) {
        if (!matchKey.empty() && (matchKey.find(set.key) != std::string::npos || set.key.find(matchKey) != std::string::npos)) {
            return &set;
        }
    }
    return &textureSets.front();
}

uint32_t AddFbxTextureAsset(ParsedScene& parsedScene,
    std::unordered_map<std::string, uint32_t>& textureCache,
    const std::filesystem::path& path,
    bool srgb)
{
    if (path.empty()) {
        return render::kInvalidResourceIndex;
    }

    const std::string cacheKey = path.lexically_normal().string() + (srgb ? "|srgb" : "|linear");
    if (const auto it = textureCache.find(cacheKey); it != textureCache.end()) {
        return it->second;
    }

    const std::optional<SceneTextureAsset> texture = DecodeTextureFile(path, srgb);
    if (!texture.has_value()) {
        return render::kInvalidResourceIndex;
    }

    const uint32_t index = static_cast<uint32_t>(parsedScene.textures.size());
    parsedScene.textures.push_back(*texture);
    textureCache.emplace(cacheKey, index);
    return index;
}

std::string TrimAscii(std::string_view value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

std::string_view TrimAsciiView(std::string_view value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

void SkipObjWhitespace(const char*& cursor, const char* end)
{
    while (cursor < end && std::isspace(static_cast<unsigned char>(*cursor))) {
        ++cursor;
    }
}

bool ReadObjToken(const char*& cursor, const char* end, std::string_view& token)
{
    SkipObjWhitespace(cursor, end);
    const char* begin = cursor;
    while (cursor < end && !std::isspace(static_cast<unsigned char>(*cursor))) {
        ++cursor;
    }
    if (begin == cursor) {
        token = {};
        return false;
    }
    token = std::string_view(begin, static_cast<size_t>(cursor - begin));
    return true;
}

bool ReadObjFloat(const char*& cursor, const char* end, float& value)
{
    SkipObjWhitespace(cursor, end);
    const auto result = std::from_chars(cursor, end, value);
    if (result.ec != std::errc{}) {
        return false;
    }
    cursor = result.ptr;
    return true;
}

std::filesystem::path ResolveObjTexturePath(const std::filesystem::path& objPath, std::string rawPath)
{
    rawPath = TrimAscii(rawPath);
    if (rawPath.empty()) {
        return {};
    }
    std::replace(rawPath.begin(), rawPath.end(), '\\', '/');
    const std::filesystem::path texturePath(rawPath);
    if (std::filesystem::exists(texturePath)) {
        return texturePath;
    }

    const std::filesystem::path localPath = objPath.parent_path() / texturePath;
    if (std::filesystem::exists(localPath)) {
        return localPath;
    }
    const std::filesystem::path localFilename = objPath.parent_path() / texturePath.filename();
    if (std::filesystem::exists(localFilename)) {
        return localFilename;
    }
    for (std::string_view folderName : { "textures", "Textures", "texture", "Texture", "maps", "Maps" }) {
        const std::filesystem::path root = objPath.parent_path() / folderName;
        if (!std::filesystem::exists(root)) {
            continue;
        }
        const std::filesystem::path direct = root / texturePath.filename();
        if (std::filesystem::exists(direct)) {
            return direct;
        }
        for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_regular_file() && entry.path().filename() == texturePath.filename()) {
                return entry.path();
            }
        }
    }
    return {};
}

uint32_t AddObjTextureAsset(ParsedScene& parsedScene,
    std::unordered_map<std::string, uint32_t>& textureCache,
    const std::filesystem::path& path,
    bool srgb)
{
    return AddFbxTextureAsset(parsedScene, textureCache, path, srgb);
}

struct ObjMaterialRecord {
    std::string name;
    SceneMaterial material;
};

std::vector<std::filesystem::path> ParseObjMtllibs(const std::filesystem::path& objPath, std::string_view arguments)
{
    std::vector<std::filesystem::path> result;
    std::istringstream stream{ std::string(arguments) };
    std::string filename;
    while (stream >> filename) {
        const std::filesystem::path path = objPath.parent_path() / filename;
        if (std::filesystem::exists(path)) {
            result.push_back(path);
        }
    }
    return result;
}

std::string ExtractObjMapFilename(std::string_view arguments)
{
    std::istringstream stream{ std::string(arguments) };
    std::string token;
    std::string lastToken;
    while (stream >> token) {
        lastToken = token;
    }
    return lastToken;
}

bool IsLikelyObjLightMaterial(std::string_view name)
{
    std::string lowered(name);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered == "light" || lowered == "lights" || lowered == "area_light" || lowered == "arealight"
        || lowered.find("emissive") != std::string::npos || lowered.find("lamp") != std::string::npos
        || lowered.find("light_") != std::string::npos || lowered.find("_light") != std::string::npos;
}

float Luminance(glm::vec3 color)
{
    return glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

bool SceneParseCancelled(const SceneParseCallbacks* callbacks)
{
    return callbacks != nullptr && callbacks->isCancelled && callbacks->isCancelled();
}

void ReportSceneParseProgress(const SceneParseCallbacks* callbacks, float progress, std::string_view message)
{
    if (callbacks != nullptr && callbacks->reportProgress) {
        callbacks->reportProgress(glm::clamp(progress, 0.0f, 1.0f), message);
    }
}

std::vector<ObjMaterialRecord> ParseObjMaterialLibraries(
    const std::filesystem::path& objPath,
    ParsedScene& parsedScene,
    std::span<const std::filesystem::path> materialPaths)
{
    std::vector<ObjMaterialRecord> records;
    std::unordered_map<std::string, uint32_t> textureCache;
    ObjMaterialRecord current;
    bool hasCurrent = false;

    auto flushCurrent = [&]() {
        if (hasCurrent) {
            if (IsLikelyObjLightMaterial(current.name) && Luminance(glm::vec3(current.material.emissiveFactor)) <= 1.0e-4f) {
                current.material.emissiveFactor = glm::vec4(glm::vec3(16.0f), 0.0f);
                current.material.baseColorFactor = glm::vec4(glm::vec3(1.0f), current.material.baseColorFactor.a);
                current.material.materialParams.x = 0.0f;
                current.material.materialParams.y = 0.18f;
            }
            records.push_back(std::move(current));
            current = {};
            hasCurrent = false;
        }
    };

    for (const std::filesystem::path& materialPath : materialPaths) {
        std::ifstream input(materialPath);
        if (!input.is_open()) {
            continue;
        }

        std::string line;
        while (std::getline(input, line)) {
            const std::string trimmed = TrimAscii(line);
            if (trimmed.empty() || trimmed.front() == '#') {
                continue;
            }
            std::istringstream stream(trimmed);
            std::string tag;
            stream >> tag;
            if (tag == "newmtl") {
                flushCurrent();
                hasCurrent = true;
                current.name = TrimAscii(trimmed.substr(tag.size()));
                current.material = MakeDefaultObjMaterial();
            } else if (hasCurrent && tag == "Kd") {
                float r = 0.8f;
                float g = 0.8f;
                float b = 0.85f;
                if (stream >> r >> g >> b) {
                    current.material.baseColorFactor = glm::vec4(r, g, b, current.material.baseColorFactor.a);
                }
            } else if (hasCurrent && tag == "Ke") {
                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                if (stream >> r >> g >> b) {
                    current.material.emissiveFactor = glm::vec4(r, g, b, 0.0f);
                }
            } else if (hasCurrent && (tag == "d" || tag == "Tr")) {
                float alpha = 1.0f;
                if (stream >> alpha) {
                    current.material.baseColorFactor.a = tag == "Tr" ? 1.0f - alpha : alpha;
                }
            } else if (hasCurrent && tag == "Ns") {
                float shininess = 32.0f;
                if (stream >> shininess) {
                    current.material.materialParams.y = glm::clamp(std::sqrt(2.0f / (std::max(shininess, 1.0f) + 2.0f)), 0.04f, 1.0f);
                }
            } else if (hasCurrent && tag == "Pr") {
                float roughness = 1.0f;
                if (stream >> roughness) {
                    current.material.materialParams.y = glm::clamp(roughness, 0.04f, 1.0f);
                }
            } else if (hasCurrent && tag == "Pm") {
                float metallic = 0.0f;
                if (stream >> metallic) {
                    current.material.materialParams.x = glm::clamp(metallic, 0.0f, 1.0f);
                }
            } else if (hasCurrent && (tag == "map_Kd" || tag == "map_BaseColor")) {
                std::string rest;
                std::getline(stream, rest);
                const std::filesystem::path texturePath = ResolveObjTexturePath(objPath, ExtractObjMapFilename(rest));
                current.material.textureIndices0.x = AddObjTextureAsset(parsedScene, textureCache, texturePath, true);
            } else if (hasCurrent && (tag == "map_Bump" || tag == "bump" || tag == "map_Kn" || tag == "map_normal")) {
                std::string rest;
                std::getline(stream, rest);
                const std::filesystem::path texturePath = ResolveObjTexturePath(objPath, ExtractObjMapFilename(rest));
                current.material.textureIndices0.z = AddObjTextureAsset(parsedScene, textureCache, texturePath, false);
                if (current.material.textureIndices0.z != render::kInvalidResourceIndex) {
                    current.material.materialParams.w = 1.0f;
                }
            } else if (hasCurrent && (tag == "map_Pr" || tag == "map_Ns")) {
                std::string rest;
                std::getline(stream, rest);
                const std::filesystem::path texturePath = ResolveObjTexturePath(objPath, ExtractObjMapFilename(rest));
                current.material.textureIndices0.y = AddObjTextureAsset(parsedScene, textureCache, texturePath, false);
            }
        }
    }
    flushCurrent();
    return records;
}

struct ObjVertexKey {
    int position{ 0 };
    int texCoord{ 0 };
    int normal{ 0 };

    [[nodiscard]] bool operator==(const ObjVertexKey& other) const
    {
        return position == other.position && texCoord == other.texCoord && normal == other.normal;
    }
};

struct ObjVertexKeyHash {
    [[nodiscard]] size_t operator()(const ObjVertexKey& key) const
    {
        size_t hash = static_cast<size_t>(key.position) * 73856093u;
        hash ^= static_cast<size_t>(key.texCoord) * 19349663u;
        hash ^= static_cast<size_t>(key.normal) * 83492791u;
        return hash;
    }
};

int ParseObjIndexToken(std::string_view token)
{
    if (token.empty()) {
        return 0;
    }
    int value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    return result.ec == std::errc{} ? value : 0;
}

ObjVertexKey ParseObjFaceVertex(std::string_view token)
{
    ObjVertexKey key{};
    const size_t firstSlash = token.find('/');
    if (firstSlash == std::string_view::npos) {
        key.position = ParseObjIndexToken(token);
        return key;
    }

    key.position = ParseObjIndexToken(token.substr(0, firstSlash));
    const size_t secondSlash = token.find('/', firstSlash + 1);
    if (secondSlash == std::string_view::npos) {
        key.texCoord = ParseObjIndexToken(token.substr(firstSlash + 1));
        return key;
    }
    key.texCoord = ParseObjIndexToken(token.substr(firstSlash + 1, secondSlash - firstSlash - 1));
    key.normal = ParseObjIndexToken(token.substr(secondSlash + 1));
    return key;
}

template <typename T>
const T* ResolveObjIndexedValue(std::span<const T> values, int objIndex)
{
    if (objIndex == 0 || values.empty()) {
        return nullptr;
    }
    const int resolved = objIndex > 0 ? objIndex - 1 : static_cast<int>(values.size()) + objIndex;
    if (resolved < 0 || resolved >= static_cast<int>(values.size())) {
        return nullptr;
    }
    return &values[static_cast<size_t>(resolved)];
}

bool ParseObjMesh(const std::filesystem::path& path, ParsedScene& parsedScene, const SceneParseCallbacks* callbacks)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }
    const uint64_t fileBytes = std::filesystem::file_size(path);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::unordered_map<std::string, uint32_t> materialLookup;

    parsedScene.materials.clear();
    parsedScene.materials.push_back(MakeDefaultObjMaterial());
    materialLookup.emplace("default", 0u);

    struct ObjPrimitiveBuilder {
        ParsedPrimitive primitive;
        std::unordered_map<ObjVertexKey, uint32_t, ObjVertexKeyHash> vertexLookup;
    };
    std::unordered_map<uint32_t, ObjPrimitiveBuilder> primitiveBuilders;
    uint32_t currentMaterial = 0u;

    auto resolveMaterial = [&](std::string_view name) -> uint32_t {
        const std::string key = TrimAscii(name);
        if (const auto it = materialLookup.find(key); it != materialLookup.end()) {
            return it->second;
        }
        return 0u;
    };
    auto getBuilder = [&](uint32_t materialIndex) -> ObjPrimitiveBuilder& {
        ObjPrimitiveBuilder& builder = primitiveBuilders[materialIndex];
        builder.primitive.materialIndex = materialIndex;
        return builder;
    };
    auto appendVertex = [&](ObjPrimitiveBuilder& builder, const ObjVertexKey& key) -> uint32_t {
        if (const auto it = builder.vertexLookup.find(key); it != builder.vertexLookup.end()) {
            return it->second;
        }

        ParsedPrimitive& primitive = builder.primitive;
        const glm::vec3* position = ResolveObjIndexedValue(std::span<const glm::vec3>(positions.data(), positions.size()), key.position);
        if (position == nullptr) {
            return 0u;
        }
        primitive.positions.push_back(*position);
        if (const glm::vec3* normal = ResolveObjIndexedValue(std::span<const glm::vec3>(normals.data(), normals.size()), key.normal)) {
            primitive.normals.push_back(*normal);
            primitive.hasNormals = true;
        } else {
            primitive.normals.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
        }
        if (const glm::vec2* uv = ResolveObjIndexedValue(std::span<const glm::vec2>(texCoords.data(), texCoords.size()), key.texCoord)) {
            primitive.texCoords.push_back(*uv);
        } else {
            primitive.texCoords.push_back(glm::vec2(0.0f));
        }

        const uint32_t index = static_cast<uint32_t>(primitive.positions.size() - 1);
        builder.vertexLookup.emplace(key, index);
        return index;
    };

    std::string line;
    std::vector<ObjVertexKey> face;
    face.reserve(8);
    uint64_t consumedBytes = 0;
    uint32_t lineCounter = 0;
    while (std::getline(input, line)) {
        consumedBytes += static_cast<uint64_t>(line.size() + 1u);
        if ((++lineCounter & 4095u) == 0u) {
            if (SceneParseCancelled(callbacks)) {
                return false;
            }
            const float progress = fileBytes > 0u ? static_cast<float>(
                std::min<uint64_t>(consumedBytes, fileBytes)) / static_cast<float>(fileBytes)
                                                  : 0.0f;
            ReportSceneParseProgress(callbacks, progress, fmt::format("Parsing OBJ {:.0f}%", progress * 100.0f));
        }
        const std::string_view trimmed = TrimAsciiView(line);
        if (trimmed.empty() || trimmed.front() == '#') {
            continue;
        }
        const char* cursor = trimmed.data();
        const char* end = cursor + trimmed.size();
        std::string_view tag;
        if (!ReadObjToken(cursor, end, tag)) {
            continue;
        }
        if (tag == "v") {
            glm::vec3 value(0.0f);
            if (ReadObjFloat(cursor, end, value.x) && ReadObjFloat(cursor, end, value.y) && ReadObjFloat(cursor, end, value.z)) {
                positions.push_back(value);
            }
        } else if (tag == "vn") {
            glm::vec3 value(0.0f, 1.0f, 0.0f);
            if (ReadObjFloat(cursor, end, value.x) && ReadObjFloat(cursor, end, value.y) && ReadObjFloat(cursor, end, value.z)) {
                normals.push_back(value);
            }
        } else if (tag == "vt") {
            glm::vec2 value(0.0f);
            if (ReadObjFloat(cursor, end, value.x) && ReadObjFloat(cursor, end, value.y)) {
                texCoords.push_back(glm::vec2(value.x, 1.0f - value.y));
            }
        } else if (tag == "mtllib") {
            const std::string_view rest = TrimAsciiView(std::string_view(cursor, static_cast<size_t>(end - cursor)));
            std::vector<std::filesystem::path> libraries = ParseObjMtllibs(path, rest);
            for (const ObjMaterialRecord& record : ParseObjMaterialLibraries(path, parsedScene, libraries)) {
                const uint32_t index = static_cast<uint32_t>(parsedScene.materials.size());
                parsedScene.materials.push_back(record.material);
                materialLookup.emplace(record.name, index);
            }
        } else if (tag == "usemtl") {
            const std::string_view rest = TrimAsciiView(std::string_view(cursor, static_cast<size_t>(end - cursor)));
            currentMaterial = resolveMaterial(rest);
        } else if (tag == "f") {
            face.clear();
            std::string_view token;
            while (ReadObjToken(cursor, end, token)) {
                if (!token.empty() && token.front() == '#') {
                    break;
                }
                face.push_back(ParseObjFaceVertex(token));
            }
            if (face.size() < 3) {
                continue;
            }
            ObjPrimitiveBuilder& builder = getBuilder(currentMaterial);
            for (size_t corner = 1; corner + 1 < face.size(); ++corner) {
                builder.primitive.indices.push_back(appendVertex(builder, face[0]));
                builder.primitive.indices.push_back(appendVertex(builder, face[corner]));
                builder.primitive.indices.push_back(appendVertex(builder, face[corner + 1]));
            }
        }
    }
    if (SceneParseCancelled(callbacks)) {
        return false;
    }
    ReportSceneParseProgress(callbacks, 1.0f, "Parsing OBJ 100%");

    const uint32_t objectIndex = static_cast<uint32_t>(parsedScene.objects.size());
    const uint32_t firstPrimitive = static_cast<uint32_t>(parsedScene.primitives.size());
    for (auto& [_, builder] : primitiveBuilders) {
        ParsedPrimitive& primitive = builder.primitive;
        if (primitive.positions.empty() || primitive.indices.empty()) {
            continue;
        }
        primitive.objectIndex = objectIndex;
        primitive.worldTransform = glm::mat4(1.0f);
        primitive.tangents = GenerateTangents(primitive.positions, primitive.normals, primitive.texCoords, primitive.indices);
        primitive.hasTangents = true;
        parsedScene.primitives.push_back(std::move(primitive));
    }

    if (parsedScene.primitives.size() == firstPrimitive) {
        return false;
    }
    parsedScene.objects.push_back(ParsedSceneObject{
        .name = path.stem().string(),
        .initialWorldTransform = glm::mat4(1.0f),
        .worldTransform = glm::mat4(1.0f),
        .firstPrimitive = firstPrimitive,
        .primitiveCount = static_cast<uint32_t>(parsedScene.primitives.size() - firstPrimitive),
    });
    parsedScene.sceneKind = SceneKind::Mesh;
    return true;
}

struct MeshPlyHeader {
    size_t vertexCount{ 0 };
    size_t faceCount{ 0 };
    std::vector<std::string> vertexProperties;
    bool ascii{ false };
};

bool ParseMeshPlyHeader(std::ifstream& input, MeshPlyHeader& header)
{
    std::string line;
    if (!std::getline(input, line) || line != "ply") {
        return false;
    }

    enum class Element {
        None,
        Vertex,
        Face,
    } currentElement = Element::None;

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
            header.ascii = formatName == "ascii";
        } else if (token == "element") {
            std::string elementName;
            size_t count = 0;
            stream >> elementName >> count;
            if (elementName == "vertex") {
                currentElement = Element::Vertex;
                header.vertexCount = count;
            } else if (elementName == "face") {
                currentElement = Element::Face;
                header.faceCount = count;
            } else {
                currentElement = Element::None;
            }
        } else if (token == "property" && currentElement == Element::Vertex) {
            std::string typeName;
            std::string propertyName;
            stream >> typeName;
            if (typeName != "list") {
                stream >> propertyName;
                header.vertexProperties.push_back(propertyName);
            }
        } else if (token == "end_header") {
            return header.ascii && header.vertexCount > 0 && header.faceCount > 0 && !header.vertexProperties.empty();
        }
    }

    return false;
}

bool ParseMeshPly(const std::filesystem::path& path, ParsedScene& parsedScene)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }

    MeshPlyHeader header;
    if (!ParseMeshPlyHeader(input, header)) {
        return false;
    }

    std::unordered_map<std::string, size_t> propertyIndex;
    for (size_t index = 0; index < header.vertexProperties.size(); ++index) {
        propertyIndex.emplace(header.vertexProperties[index], index);
    }
    const auto findProperty = [&](std::string_view name) -> int {
        const auto it = propertyIndex.find(std::string(name));
        return it == propertyIndex.end() ? -1 : static_cast<int>(it->second);
    };

    const int xIndex = findProperty("x");
    const int yIndex = findProperty("y");
    const int zIndex = findProperty("z");
    if (xIndex < 0 || yIndex < 0 || zIndex < 0) {
        return false;
    }
    const int nxIndex = findProperty("nx");
    const int nyIndex = findProperty("ny");
    const int nzIndex = findProperty("nz");
    const int redIndex = findProperty("red");
    const int greenIndex = findProperty("green");
    const int blueIndex = findProperty("blue");

    ParsedPrimitive primitive;
    primitive.positions.reserve(header.vertexCount);
    primitive.texCoords.resize(header.vertexCount, glm::vec2(0.0f));
    if (nxIndex >= 0 && nyIndex >= 0 && nzIndex >= 0) {
        primitive.normals.reserve(header.vertexCount);
        primitive.hasNormals = true;
    }

    parsedScene.materials.clear();
    SceneMaterial material = MakeDefaultMaterial();
    material.baseColorFactor = glm::vec4(0.78f, 0.76f, 0.72f, 1.0f);
    material.materialParams = glm::vec4(0.0f, 0.58f, 1.0f, 1.0f);
    parsedScene.materials.push_back(material);

    std::string line;
    std::vector<float> values;
    values.reserve(header.vertexProperties.size());
    glm::vec3 accumulatedColor(0.0f);
    size_t colorCount = 0;
    for (size_t vertexIndex = 0; vertexIndex < header.vertexCount; ++vertexIndex) {
        if (!std::getline(input, line)) {
            return false;
        }
        values.clear();
        std::istringstream stream(line);
        for (size_t property = 0; property < header.vertexProperties.size(); ++property) {
            float value = 0.0f;
            stream >> value;
            values.push_back(value);
        }
        if (values.size() < header.vertexProperties.size()) {
            return false;
        }

        const auto valueAt = [&](int index, float fallback = 0.0f) {
            return index >= 0 && static_cast<size_t>(index) < values.size() ? values[static_cast<size_t>(index)] : fallback;
        };
        primitive.positions.push_back(glm::vec3(valueAt(xIndex), valueAt(yIndex), valueAt(zIndex)));
        if (primitive.hasNormals) {
            glm::vec3 normal(valueAt(nxIndex), valueAt(nyIndex), valueAt(nzIndex));
            normal = glm::length(normal) > 1.0e-5f ? glm::normalize(normal) : glm::vec3(0.0f, 1.0f, 0.0f);
            primitive.normals.push_back(normal);
        }
        if (redIndex >= 0 && greenIndex >= 0 && blueIndex >= 0) {
            accumulatedColor += glm::vec3(valueAt(redIndex), valueAt(greenIndex), valueAt(blueIndex)) / 255.0f;
            ++colorCount;
        }
    }

    if (colorCount > 0) {
        parsedScene.materials[0].baseColorFactor = glm::vec4(glm::clamp(accumulatedColor / static_cast<float>(colorCount),
                                                               glm::vec3(0.05f),
                                                               glm::vec3(1.0f)),
            1.0f);
    }

    for (size_t faceIndex = 0; faceIndex < header.faceCount; ++faceIndex) {
        if (!std::getline(input, line)) {
            return false;
        }
        std::istringstream stream(line);
        uint32_t count = 0;
        stream >> count;
        if (count < 3) {
            continue;
        }
        std::vector<uint32_t> face(count);
        for (uint32_t corner = 0; corner < count; ++corner) {
            stream >> face[corner];
            if (face[corner] >= primitive.positions.size()) {
                return false;
            }
        }
        for (uint32_t corner = 1; corner + 1 < count; ++corner) {
            primitive.indices.push_back(face[0]);
            primitive.indices.push_back(face[corner]);
            primitive.indices.push_back(face[corner + 1]);
        }
    }

    if (primitive.positions.empty() || primitive.indices.empty()) {
        return false;
    }

    if (!primitive.hasNormals) {
        primitive.normals.resize(primitive.positions.size(), glm::vec3(0.0f, 1.0f, 0.0f));
    }
    primitive.materialIndex = 0;
    primitive.objectIndex = static_cast<uint32_t>(parsedScene.objects.size());
    primitive.worldTransform = glm::mat4(1.0f);
    primitive.tangents = GenerateTangents(primitive.positions, primitive.normals, primitive.texCoords, primitive.indices);
    primitive.hasTangents = true;

    const uint32_t firstPrimitive = static_cast<uint32_t>(parsedScene.primitives.size());
    parsedScene.primitives.push_back(std::move(primitive));
    parsedScene.objects.push_back(ParsedSceneObject{
        .name = path.stem().string(),
        .initialWorldTransform = glm::mat4(1.0f),
        .worldTransform = glm::mat4(1.0f),
        .firstPrimitive = firstPrimitive,
        .primitiveCount = 1,
    });
    parsedScene.sceneKind = SceneKind::Mesh;
    return true;
}

template <typename T>
bool DecodeFbxArray(std::span<const std::byte> bytes, size_t& offset, size_t elementSize, std::vector<T>& out)
{
    uint32_t count = 0;
    uint32_t encoding = 0;
    uint32_t encodedLength = 0;
    if (!ReadLittleEndian(bytes, offset, count) || !ReadLittleEndian(bytes, offset, encoding) || !ReadLittleEndian(bytes, offset, encodedLength)) {
        return false;
    }
    if (offset + encodedLength > bytes.size()) {
        return false;
    }

    const size_t decodedBytes = static_cast<size_t>(count) * elementSize;
    std::vector<std::byte> decoded;
    if (encoding == 0) {
        if (encodedLength != decodedBytes) {
            return false;
        }
        decoded.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + encodedLength));
    } else if (encoding == 1) {
        decoded.resize(decodedBytes);
        const int written = stbi_zlib_decode_buffer(reinterpret_cast<char*>(decoded.data()),
            static_cast<int>(decoded.size()),
            reinterpret_cast<const char*>(bytes.data() + offset),
            static_cast<int>(encodedLength));
        if (written != static_cast<int>(decodedBytes)) {
            return false;
        }
    } else {
        return false;
    }
    offset += encodedLength;

    out.resize(count);
    if (decodedBytes > 0) {
        std::memcpy(out.data(), decoded.data(), decodedBytes);
    }
    return true;
}

bool ReadFbxProperty(std::span<const std::byte> bytes, size_t& offset, FbxProperty& property)
{
    if (offset >= bytes.size()) {
        return false;
    }

    property.type = static_cast<char>(bytes[offset]);
    ++offset;
    switch (property.type) {
    case 'Y':
        return ReadLittleEndian(bytes, offset, property.int16Value);
    case 'C': {
        uint8_t value = 0;
        if (!ReadLittleEndian(bytes, offset, value)) {
            return false;
        }
        property.boolValue = value != 0;
        return true;
    }
    case 'I':
        return ReadLittleEndian(bytes, offset, property.int32Value);
    case 'F':
        return ReadLittleEndian(bytes, offset, property.floatValue);
    case 'D':
        return ReadLittleEndian(bytes, offset, property.doubleValue);
    case 'L':
        return ReadLittleEndian(bytes, offset, property.int64Value);
    case 'i':
        return DecodeFbxArray(bytes, offset, sizeof(int32_t), property.int32Array);
    case 'l':
        return DecodeFbxArray(bytes, offset, sizeof(int64_t), property.int64Array);
    case 'd':
        return DecodeFbxArray(bytes, offset, sizeof(double), property.doubleArray);
    case 'S':
    case 'R': {
        uint32_t length = 0;
        if (!ReadLittleEndian(bytes, offset, length) || offset + length > bytes.size()) {
            return false;
        }
        property.stringValue.assign(reinterpret_cast<const char*>(bytes.data() + offset), length);
        offset += length;
        return true;
    }
    default:
        return false;
    }
}

bool ReadFbxNode(std::span<const std::byte> bytes, size_t& offset, bool wideOffsets, FbxNode& node, bool& isNull)
{
    const size_t headerOffset = offset;
    uint64_t endOffset = 0;
    uint64_t propertyCount = 0;
    uint64_t propertyListLength = 0;
    if (wideOffsets) {
        if (!ReadLittleEndian(bytes, offset, endOffset) || !ReadLittleEndian(bytes, offset, propertyCount)
            || !ReadLittleEndian(bytes, offset, propertyListLength)) {
            return false;
        }
    } else {
        uint32_t endOffset32 = 0;
        uint32_t propertyCount32 = 0;
        uint32_t propertyListLength32 = 0;
        if (!ReadLittleEndian(bytes, offset, endOffset32) || !ReadLittleEndian(bytes, offset, propertyCount32)
            || !ReadLittleEndian(bytes, offset, propertyListLength32)) {
            return false;
        }
        endOffset = endOffset32;
        propertyCount = propertyCount32;
        propertyListLength = propertyListLength32;
    }

    uint8_t nameLength = 0;
    if (!ReadLittleEndian(bytes, offset, nameLength) || offset + nameLength > bytes.size()) {
        return false;
    }

    isNull = endOffset == 0 && propertyCount == 0 && propertyListLength == 0 && nameLength == 0;
    if (isNull) {
        return true;
    }
    if (endOffset <= headerOffset || endOffset > bytes.size()) {
        return false;
    }

    node.name.assign(reinterpret_cast<const char*>(bytes.data() + offset), nameLength);
    offset += nameLength;
    node.properties.reserve(static_cast<size_t>(propertyCount));
    for (uint64_t propertyIndex = 0; propertyIndex < propertyCount; ++propertyIndex) {
        FbxProperty property;
        if (!ReadFbxProperty(bytes, offset, property)) {
            return false;
        }
        node.properties.push_back(std::move(property));
    }

    const uint64_t nullRecordBytes = wideOffsets ? 25u : 13u;
    while (offset + nullRecordBytes <= endOffset) {
        FbxNode child;
        bool childIsNull = false;
        const size_t childOffset = offset;
        if (!ReadFbxNode(bytes, offset, wideOffsets, child, childIsNull)) {
            return false;
        }
        if (childIsNull) {
            break;
        }
        if (offset <= childOffset) {
            return false;
        }
        node.children.push_back(std::move(child));
    }
    offset = static_cast<size_t>(endOffset);
    return true;
}

std::optional<FbxNode> ParseBinaryFbxTree(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        return std::nullopt;
    }

    const std::streamsize fileSize = input.tellg();
    if (fileSize <= 27) {
        return std::nullopt;
    }
    input.seekg(0, std::ios::beg);

    std::vector<std::byte> storage(static_cast<size_t>(fileSize));
    if (!input.read(reinterpret_cast<char*>(storage.data()), fileSize)) {
        return std::nullopt;
    }
    const std::span<const std::byte> bytes(storage);
    constexpr std::array<std::byte, 23> kMagic = {
        std::byte{ 'K' }, std::byte{ 'a' }, std::byte{ 'y' }, std::byte{ 'd' }, std::byte{ 'a' }, std::byte{ 'r' }, std::byte{ 'a' },
        std::byte{ ' ' }, std::byte{ 'F' }, std::byte{ 'B' }, std::byte{ 'X' }, std::byte{ ' ' }, std::byte{ 'B' }, std::byte{ 'i' },
        std::byte{ 'n' }, std::byte{ 'a' }, std::byte{ 'r' }, std::byte{ 'y' }, std::byte{ ' ' }, std::byte{ ' ' }, std::byte{ 0 },
        std::byte{ 0x1a }, std::byte{ 0 },
    };
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        return std::nullopt;
    }

    uint32_t version = 0;
    size_t offset = kMagic.size();
    if (!ReadLittleEndian(bytes, offset, version)) {
        return std::nullopt;
    }

    FbxNode root;
    root.name = "Root";
    const bool wideOffsets = version >= 7500;
    while (offset < bytes.size()) {
        FbxNode node;
        bool isNull = false;
        const size_t before = offset;
        if (!ReadFbxNode(bytes, offset, wideOffsets, node, isNull)) {
            break;
        }
        if (isNull) {
            break;
        }
        if (offset <= before) {
            return std::nullopt;
        }
        root.children.push_back(std::move(node));
    }
    return root.children.empty() ? std::nullopt : std::optional<FbxNode>(std::move(root));
}

bool AppendFbxGeometry(
    const FbxNode& geometryNode, ParsedScene& parsedScene, uint32_t materialIndex, const glm::mat4& worldTransform, std::string_view objectName)
{
    const FbxNode* verticesNode = FindChildNode(geometryNode, "Vertices");
    const FbxNode* indicesNode = FindChildNode(geometryNode, "PolygonVertexIndex");
    if (verticesNode == nullptr || verticesNode->properties.empty() || indicesNode == nullptr || indicesNode->properties.empty()) {
        return false;
    }

    const std::vector<double>& verticesRaw = verticesNode->properties.front().doubleArray;
    const std::vector<int32_t>& polygonIndicesRaw = indicesNode->properties.front().int32Array;
    if (verticesRaw.size() < 3 || polygonIndicesRaw.size() < 3 || verticesRaw.size() % 3 != 0) {
        return false;
    }

    std::vector<glm::vec3> controlPoints(verticesRaw.size() / 3);
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        controlPoints[i] = glm::vec3(static_cast<float>(verticesRaw[i * 3 + 0]),
            static_cast<float>(verticesRaw[i * 3 + 1]),
            static_cast<float>(verticesRaw[i * 3 + 2]));
    }

    const FbxNode* normalsNode = nullptr;
    if (const FbxNode* layerNormal = FindChildNode(geometryNode, "LayerElementNormal")) {
        normalsNode = FindChildNode(*layerNormal, "Normals");
    }
    const std::vector<double>* normalsRaw = normalsNode != nullptr && !normalsNode->properties.empty()
        ? &normalsNode->properties.front().doubleArray
        : nullptr;

    const FbxNode* uvNode = nullptr;
    const FbxNode* uvIndexNode = nullptr;
    for (const FbxNode& child : geometryNode.children) {
        if (child.name == "LayerElementUV") {
            uvNode = FindChildNode(child, "UV");
            uvIndexNode = FindChildNode(child, "UVIndex");
            break;
        }
    }
    const std::vector<double>* uvRaw = uvNode != nullptr && !uvNode->properties.empty() ? &uvNode->properties.front().doubleArray : nullptr;
    const std::vector<int32_t>* uvIndices = uvIndexNode != nullptr && !uvIndexNode->properties.empty() ? &uvIndexNode->properties.front().int32Array : nullptr;

    ParsedPrimitive primitive;
    primitive.materialIndex = materialIndex;
    primitive.hasNormals = normalsRaw != nullptr && !normalsRaw->empty();

    std::vector<uint32_t> polygonVertexToPrimitiveVertex(polygonIndicesRaw.size(), std::numeric_limits<uint32_t>::max());
    auto appendPolygonVertex = [&](size_t polygonVertexIndex) -> uint32_t {
        uint32_t& cachedIndex = polygonVertexToPrimitiveVertex[polygonVertexIndex];
        if (cachedIndex != std::numeric_limits<uint32_t>::max()) {
            return cachedIndex;
        }

        const int32_t rawIndex = polygonIndicesRaw[polygonVertexIndex];
        const uint32_t controlPointIndex = rawIndex < 0 ? static_cast<uint32_t>(-rawIndex - 1) : static_cast<uint32_t>(rawIndex);
        if (controlPointIndex >= controlPoints.size()) {
            return 0;
        }

        primitive.positions.push_back(controlPoints[controlPointIndex]);
        glm::vec3 normal(0.0f, 1.0f, 0.0f);
        if (normalsRaw != nullptr) {
            const size_t polygonVertexNormalOffset = polygonVertexIndex * 3;
            const size_t controlPointNormalOffset = static_cast<size_t>(controlPointIndex) * 3;
            const size_t normalOffset = polygonVertexNormalOffset + 2 < normalsRaw->size() ? polygonVertexNormalOffset : controlPointNormalOffset;
            if (normalOffset + 2 < normalsRaw->size()) {
                normal = glm::vec3(static_cast<float>((*normalsRaw)[normalOffset + 0]),
                    static_cast<float>((*normalsRaw)[normalOffset + 1]),
                    static_cast<float>((*normalsRaw)[normalOffset + 2]));
            }
        }
        primitive.normals.push_back(normal);

        glm::vec2 uv(0.0f);
        if (uvRaw != nullptr) {
            size_t uvIndex = controlPointIndex;
            if (uvIndices != nullptr && polygonVertexIndex < uvIndices->size() && (*uvIndices)[polygonVertexIndex] >= 0) {
                uvIndex = static_cast<size_t>((*uvIndices)[polygonVertexIndex]);
            }
            if (uvIndex * 2 + 1 < uvRaw->size()) {
                uv = glm::vec2(static_cast<float>((*uvRaw)[uvIndex * 2 + 0]), 1.0f - static_cast<float>((*uvRaw)[uvIndex * 2 + 1]));
            }
        }
        primitive.texCoords.push_back(uv);
        cachedIndex = static_cast<uint32_t>(primitive.positions.size() - 1);
        return cachedIndex;
    };

    std::vector<size_t> polygon;
    polygon.reserve(8);
    for (size_t polygonVertexIndex = 0; polygonVertexIndex < polygonIndicesRaw.size(); ++polygonVertexIndex) {
        polygon.push_back(polygonVertexIndex);
        if (polygonIndicesRaw[polygonVertexIndex] < 0) {
            if (polygon.size() >= 3) {
                for (size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
                    primitive.indices.push_back(appendPolygonVertex(polygon[0]));
                    primitive.indices.push_back(appendPolygonVertex(polygon[corner]));
                    primitive.indices.push_back(appendPolygonVertex(polygon[corner + 1]));
                }
            }
            polygon.clear();
        }
    }

    if (primitive.positions.empty() || primitive.indices.empty()) {
        return false;
    }
    primitive.tangents = GenerateTangents(primitive.positions, primitive.normals, primitive.texCoords, primitive.indices);
    primitive.hasTangents = true;

    const uint32_t objectIndex = static_cast<uint32_t>(parsedScene.objects.size());
    primitive.objectIndex = objectIndex;
    primitive.worldTransform = worldTransform;
    const uint32_t firstPrimitive = static_cast<uint32_t>(parsedScene.primitives.size());
    parsedScene.objects.push_back(ParsedSceneObject{
        .name = objectName.empty() ? FbxObjectDisplayName(geometryNode, fmt::format("FBX Object {}", objectIndex)) : std::string(objectName),
        .initialWorldTransform = worldTransform,
        .worldTransform = worldTransform,
        .firstPrimitive = firstPrimitive,
        .primitiveCount = 1,
    });
    parsedScene.primitives.push_back(std::move(primitive));
    return true;
}

bool ParseFbxMesh(const std::filesystem::path& path, ParsedScene& parsedScene)
{
    std::optional<FbxNode> root = ParseBinaryFbxTree(path);
    if (!root.has_value()) {
        return false;
    }

    const FbxNode* objectsNode = FindChildNode(*root, "Objects");
    if (objectsNode == nullptr) {
        return false;
    }
    const glm::mat4 unitTransform = glm::scale(glm::mat4(1.0f), glm::vec3(FbxUnitScaleToMeters(*root)));

    parsedScene.materials.clear();
    std::unordered_map<int64_t, FbxMaterialInfo> materials;
    std::unordered_map<int64_t, FbxTextureInfo> textures;
    std::unordered_map<int64_t, int64_t> textureToVideo;
    std::unordered_map<int64_t, std::string> textureToMaterialProperty;
    std::unordered_map<int64_t, int64_t> textureToMaterial;
    std::unordered_map<std::string, uint32_t> textureCache;

    for (const FbxNode& child : objectsNode->children) {
        if (child.name == "Material" && !child.properties.empty()) {
            SceneMaterial material = MakeDefaultMaterial();
            if (const FbxNode* properties = FindChildNode(child, "Properties70")) {
                for (const FbxNode& property : properties->children) {
                    if (const std::optional<glm::vec3> value = ReadFbxPropertyVec3(property, "DiffuseColor")) {
                        material.baseColorFactor = glm::vec4(*value, 1.0f);
                    }
                }
            }

            const int64_t materialId = FbxPropertyAsInt64(child.properties[0]);
            const uint32_t materialIndex = static_cast<uint32_t>(parsedScene.materials.size());
            const std::string materialName = FbxObjectDisplayName(child, fmt::format("FBX Material {}", materialId));
            parsedScene.materials.push_back(material);
            materials.emplace(materialId,
                FbxMaterialInfo{
                    .name = materialName,
                    .materialIndex = materialIndex,
                });
        } else if (child.name == "Video" && !child.properties.empty()) {
            std::filesystem::path texturePath;
            if (const FbxNode* relativeFilename = FindChildNode(child, "RelativeFilename"); relativeFilename != nullptr && !relativeFilename->properties.empty()) {
                texturePath = FbxFilenameFromRawPath(relativeFilename->properties.front().stringValue);
            } else if (const FbxNode* filename = FindChildNode(child, "FileName"); filename != nullptr && !filename->properties.empty()) {
                texturePath = FbxFilenameFromRawPath(filename->properties.front().stringValue);
            }

            const int64_t videoId = FbxPropertyAsInt64(child.properties[0]);
            textures.emplace(videoId, FbxTextureInfo{ .path = ResolveFbxTexturePath(path, texturePath) });
        } else if (child.name == "Texture" && !child.properties.empty()) {
            const int64_t textureId = FbxPropertyAsInt64(child.properties[0]);
            textures.emplace(textureId, FbxTextureInfo{});
        }
    }

    if (parsedScene.materials.empty()) {
        parsedScene.materials.push_back(MakeDefaultMaterial());
        materials.emplace(0,
            FbxMaterialInfo{
                .name = path.stem().string(),
                .materialIndex = 0,
            });
    }
    const uint32_t defaultMaterialIndex = 0;

    std::unordered_map<int64_t, FbxModelInfo> models;
    std::unordered_map<int64_t, int64_t> geometryToModel;
    for (const FbxNode& child : objectsNode->children) {
        if (child.name == "Model" && !child.properties.empty()) {
            const int64_t modelId = FbxPropertyAsInt64(child.properties[0]);
            models.emplace(modelId,
                FbxModelInfo{
                    .name = FbxObjectDisplayName(child, fmt::format("FBX Model {}", modelId)),
                    .transform = FbxTransformFromProperties(child),
                });
        }
    }

    if (const FbxNode* connections = FindChildNode(*root, "Connections")) {
        for (const FbxNode& connection : connections->children) {
            if (connection.name != "C" || connection.properties.size() < 3) {
                continue;
            }

            const std::string& relation = connection.properties[0].stringValue;
            const int64_t childId = FbxPropertyAsInt64(connection.properties[1]);
            const int64_t parentId = FbxPropertyAsInt64(connection.properties[2]);
            if (relation == "OO") {
                if (models.contains(parentId)) {
                    geometryToModel[childId] = parentId;
                }
                if (materials.contains(childId) && models.contains(parentId)) {
                    models[parentId].materialIndex = materials[childId].materialIndex;
                }
                if (textures.contains(childId) && textures.contains(parentId)) {
                    if (!textures[childId].path.empty()) {
                        textureToVideo[parentId] = childId;
                    } else if (!textures[parentId].path.empty()) {
                        textureToVideo[childId] = parentId;
                    }
                }
            } else if (relation == "OP" && connection.properties.size() >= 4) {
                if (textures.contains(childId) && materials.contains(parentId)) {
                    textureToMaterial[childId] = parentId;
                    textureToMaterialProperty[childId] = connection.properties[3].stringValue;
                }
            }
        }
    }

    for (const auto& [textureId, videoId] : textureToVideo) {
        if (textures[textureId].path.empty()) {
            textures[textureId].path = textures[videoId].path;
        }
    }

    for (const auto& [textureId, materialId] : textureToMaterial) {
        FbxMaterialInfo& material = materials[materialId];
        const std::filesystem::path texturePath = textures[textureId].path;
        const std::string property = ToLowerAscii(textureToMaterialProperty[textureId]);
        if (ContainsAny(property, { "diffuse", "basecolor", "base_color", "color" })) {
            material.baseColorPath = texturePath;
        } else if (ContainsAny(property, { "normal", "bump" })) {
            material.normalPath = texturePath;
        } else if (ContainsAny(property, { "roughness", "rough" })) {
            material.roughnessPath = texturePath;
        }
    }

    const std::vector<FbxTextureSet> discoveredTextureSets = DiscoverFbxTextureSets(path);
    for (auto& [_, material] : materials) {
        FbxTextureSet explicitSet{
            .key = NormalizeTextureMatchKey(material.name),
            .baseColor = material.baseColorPath,
            .normal = material.normalPath,
            .roughness = material.roughnessPath,
        };
        const FbxTextureSet* selectedSet = &explicitSet;
        if (selectedSet->baseColor.empty() && selectedSet->normal.empty() && selectedSet->roughness.empty()) {
            selectedSet = SelectFbxTextureSet(discoveredTextureSets, material.name);
        }
        if (selectedSet == nullptr) {
            continue;
        }

        SceneMaterial& sceneMaterial = parsedScene.materials[material.materialIndex];
        const uint32_t baseColorIndex = AddFbxTextureAsset(parsedScene, textureCache, selectedSet->baseColor, true);
        const uint32_t roughnessIndex = AddFbxTextureAsset(parsedScene, textureCache, selectedSet->roughness, false);
        const uint32_t normalIndex = AddFbxTextureAsset(parsedScene, textureCache, selectedSet->normal, false);
        if (baseColorIndex != render::kInvalidResourceIndex) {
            sceneMaterial.textureIndices0.x = baseColorIndex;
            sceneMaterial.baseColorFactor = glm::vec4(1.0f);
        }
        if (roughnessIndex != render::kInvalidResourceIndex) {
            sceneMaterial.textureIndices0.y = roughnessIndex;
            sceneMaterial.materialParams.y = 1.0f;
        }
        if (normalIndex != render::kInvalidResourceIndex) {
            sceneMaterial.textureIndices0.z = normalIndex;
            sceneMaterial.materialParams.w = 1.0f;
        }
    }

    if (!materials.empty() && !discoveredTextureSets.empty()) {
        for (auto& [_, model] : models) {
            const FbxTextureSet* selectedSet = SelectFbxTextureSet(discoveredTextureSets, model.name);
            if (selectedSet == nullptr) {
                continue;
            }
            SceneMaterial modelMaterial = parsedScene.materials[model.materialIndex];
            const uint32_t baseColorIndex = AddFbxTextureAsset(parsedScene, textureCache, selectedSet->baseColor, true);
            const uint32_t roughnessIndex = AddFbxTextureAsset(parsedScene, textureCache, selectedSet->roughness, false);
            const uint32_t normalIndex = AddFbxTextureAsset(parsedScene, textureCache, selectedSet->normal, false);
            if (baseColorIndex == render::kInvalidResourceIndex && roughnessIndex == render::kInvalidResourceIndex
                && normalIndex == render::kInvalidResourceIndex) {
                continue;
            }
            if (baseColorIndex != render::kInvalidResourceIndex) {
                modelMaterial.textureIndices0.x = baseColorIndex;
                modelMaterial.baseColorFactor = glm::vec4(1.0f);
            }
            if (roughnessIndex != render::kInvalidResourceIndex) {
                modelMaterial.textureIndices0.y = roughnessIndex;
                modelMaterial.materialParams.y = 1.0f;
            }
            if (normalIndex != render::kInvalidResourceIndex) {
                modelMaterial.textureIndices0.z = normalIndex;
                modelMaterial.materialParams.w = 1.0f;
            }
            if (modelMaterial.textureIndices0 != parsedScene.materials[model.materialIndex].textureIndices0) {
                model.materialIndex = static_cast<uint32_t>(parsedScene.materials.size());
                parsedScene.materials.push_back(modelMaterial);
            }
        }
    }

    for (const FbxNode& child : objectsNode->children) {
        if (child.name == "Geometry") {
            const int64_t geometryId = child.properties.empty() ? 0 : FbxPropertyAsInt64(child.properties[0]);
            glm::mat4 worldTransform(1.0f);
            std::string objectName;
            uint32_t materialIndex = defaultMaterialIndex;
            if (const auto mapping = geometryToModel.find(geometryId); mapping != geometryToModel.end()) {
                if (const auto model = models.find(mapping->second); model != models.end()) {
                    worldTransform = model->second.transform;
                    objectName = model->second.name;
                    materialIndex = model->second.materialIndex;
                }
            }
            worldTransform = unitTransform * worldTransform;
            AppendFbxGeometry(child, parsedScene, materialIndex, worldTransform, objectName);
        }
    }

    if (parsedScene.primitives.empty()) {
        return false;
    }
    parsedScene.sceneKind = SceneKind::Mesh;
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

double GaussianImportImportance(const SceneVertex& vertex, const GaussianPrimitive& gaussian)
{
    const glm::vec3 scale = glm::max(glm::abs(glm::vec3(gaussian.scale)), glm::vec3(1.0e-8f));
    const float opacity = glm::clamp(gaussian.positionOpacity.w > 0.0f ? gaussian.positionOpacity.w : vertex.splatParams.y, 0.0f, 1.0f);
    const double importance = static_cast<double>(scale.x) * static_cast<double>(scale.y) * static_cast<double>(scale.z)
        * static_cast<double>(opacity);
    return std::isfinite(importance) ? importance : std::numeric_limits<double>::max();
}

void SortGaussianImportByImportance(ParsedScene& parsedScene)
{
    const size_t count = parsedScene.gaussianVertices.size();
    if (count < 2 || parsedScene.gaussianPrimitives.size() != count) {
        return;
    }

    std::vector<size_t> order(count);
    for (size_t index = 0; index < count; ++index) {
        order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return GaussianImportImportance(parsedScene.gaussianVertices[lhs], parsedScene.gaussianPrimitives[lhs])
            > GaussianImportImportance(parsedScene.gaussianVertices[rhs], parsedScene.gaussianPrimitives[rhs]);
    });

    std::vector<SceneVertex> sortedVertices;
    std::vector<GaussianPrimitive> sortedGaussians;
    sortedVertices.reserve(count);
    sortedGaussians.reserve(count);
    for (size_t index : order) {
        sortedVertices.push_back(parsedScene.gaussianVertices[index]);
        sortedGaussians.push_back(parsedScene.gaussianPrimitives[index]);
    }
    parsedScene.gaussianVertices = std::move(sortedVertices);
    parsedScene.gaussianPrimitives = std::move(sortedGaussians);
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
        SortGaussianImportByImportance(parsedScene);
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

std::vector<SceneEmissiveTriangle> BuildEmissiveTriangles(std::span<const SceneTriangle> triangles)
{
    std::vector<SceneEmissiveTriangle> lights;
    std::vector<float> weights;
    for (uint32_t triangleIndex = 0; triangleIndex < static_cast<uint32_t>(triangles.size()); ++triangleIndex) {
        const SceneTriangle& triangle = triangles[triangleIndex];
        const glm::vec3 e0 = glm::vec3(triangle.p1 - triangle.p0);
        const glm::vec3 e1 = glm::vec3(triangle.p2 - triangle.p0);
        const float area = glm::length(glm::cross(e0, e1)) * 0.5f;
        if (area <= 1.0e-6f) {
            continue;
        }

        const bool hasEmissiveTexture = triangle.textureIndices1.x != render::kInvalidResourceIndex;
        const glm::vec3 emissive = glm::vec3(triangle.emissiveFactor);
        const float luminance = glm::dot(emissive, glm::vec3(0.2126f, 0.7152f, 0.0722f));
        const float powerEstimate = hasEmissiveTexture ? std::max(luminance, 1.0f) : luminance;
        if (powerEstimate > 1.0e-4f) {
            lights.push_back(SceneEmissiveTriangle{ .triangleIndex = triangleIndex });
            weights.push_back(area * powerEstimate);
        }
    }

    float totalWeight = 0.0f;
    for (float weight : weights) {
        totalWeight += weight;
    }
    if (totalWeight <= 0.0f) {
        return {};
    }

    float cumulative = 0.0f;
    for (size_t lightIndex = 0; lightIndex < lights.size(); ++lightIndex) {
        const SceneTriangle& triangle = triangles[lights[lightIndex].triangleIndex];
        const glm::vec3 e0 = glm::vec3(triangle.p1 - triangle.p0);
        const glm::vec3 e1 = glm::vec3(triangle.p2 - triangle.p0);
        const float area = glm::length(glm::cross(e0, e1)) * 0.5f;
        const float probability = weights[lightIndex] / totalWeight;
        cumulative += probability;
        lights[lightIndex].areaOverProbability = probability > 0.0f ? area / probability : 0.0f;
        lights[lightIndex].cdf = lightIndex + 1 == lights.size() ? 1.0f : cumulative;
    }
    return lights;
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
    static const SceneParseCallbacks kNoCallbacks{};
    return ParseFromFile(path, kNoCallbacks);
}

bool Scene::ParseFromFile(const std::filesystem::path& path, const SceneParseCallbacks& callbacks)
{
    _parsed.reset();
    _prepared.reset();

    if (!std::filesystem::exists(path)) {
        return false;
    }
    const SceneParseCallbacks* callbackPtr = &callbacks;
    if (SceneParseCancelled(callbackPtr)) {
        return false;
    }

    auto parsed = std::make_shared<ParsedScene>();
    ParsedScene& parsedScene = *parsed;
    parsedScene.sourcePath = path;
    parsedScene.sceneKind = SceneKind::Empty;
    const std::string extension = ToLowerExtension(path);

    if (std::filesystem::is_directory(path)) {
        ReportSceneParseProgress(callbackPtr, 0.05f, "Parsing Gaussian folder");
        if (!ParseGaussianPly(path, parsedScene)) {
            return false;
        }
        _parsed = std::move(parsed);
        return _parsed->IsLoaded();
    }

    if (extension == ".ply") {
        ReportSceneParseProgress(callbackPtr, 0.05f, "Parsing PLY");
        if (!ParseMeshPly(path, parsedScene) && !ParseGaussianPly(path, parsedScene)) {
            return false;
        }
        _parsed = std::move(parsed);
        return _parsed->IsLoaded();
    }

    if (extension == ".fbx") {
        ReportSceneParseProgress(callbackPtr, 0.05f, "Parsing FBX");
        if (!ParseFbxMesh(path, parsedScene)) {
            return false;
        }
        _parsed = std::move(parsed);
        return _parsed->IsLoaded();
    }

    if (extension == ".obj") {
        if (!ParseObjMesh(path, parsedScene, callbackPtr)) {
            return false;
        }
        _parsed = std::move(parsed);
        return _parsed->IsLoaded();
    }

    fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization);
    fastgltf::GltfDataBuffer data;
    ReportSceneParseProgress(callbackPtr, 0.05f, "Reading glTF");
    if (!data.loadFromFile(path)) {
        return false;
    }
    if (SceneParseCancelled(callbackPtr)) {
        return false;
    }

    const fastgltf::GltfType type = fastgltf::determineGltfFileType(&data);
    std::optional<fastgltf::Expected<fastgltf::Asset>> asset;
    ReportSceneParseProgress(callbackPtr, 0.20f, "Parsing glTF");
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
    if (SceneParseCancelled(callbackPtr)) {
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
        if (material.alphaMode == fastgltf::AlphaMode::Mask) {
            sceneMaterial.emissiveFactor.w = static_cast<float>(material.alphaCutoff);
        }
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
    ReportSceneParseProgress(callbackPtr, 0.50f, "Flattening glTF");

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
                    .objectIndex = primitive.objectIndex,
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
    sceneData.emissiveTriangles = BuildEmissiveTriangles(sceneData.triangles);
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
    gpu.emissiveTriangles = BuildEmissiveTriangles(gpu.triangles);

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
        if (!gpu.emissiveTriangles.empty()) {
            gpu.emissiveTriangleBuffer = CreateHostBufferAndCopy(device,
                std::span<const SceneEmissiveTriangle>(gpu.emissiveTriangles),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                "SceneEmissiveTriangles",
                true);
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
        if (!gpu.emissiveTriangles.empty()) {
            gpu.emissiveTriangleBuffer = CreateHostBufferAndCopy(device,
                std::span<const SceneEmissiveTriangle>(gpu.emissiveTriangles),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                "SceneEmissiveTriangles",
                true);
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

bool Scene::UpdateMaterial(vesta::render::RenderDevice& device, uint32_t materialIndex, const SceneMaterial& material)
{
    if (_prepared == nullptr || materialIndex >= _prepared->materials.size()) {
        return false;
    }

    _prepared->materials[materialIndex] = material;

    if (_gpu != nullptr && materialIndex < _gpu->materials.size()) {
        _gpu->materials[materialIndex] = material;
        UploadGpuResourceChunk(device, SceneUploadResource::Material, sizeof(SceneMaterial) * materialIndex, sizeof(SceneMaterial));
    }

    ++_contentVersion;
    return true;
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
    if (_gpu->emissiveTriangleBuffer) {
        device.DestroyBuffer(_gpu->emissiveTriangleBuffer);
        _gpu->emissiveTriangleBuffer = {};
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

render::ImageHandle Scene::GetTextureImage(size_t textureIndex) const
{
    const GpuScene& gpu = GetGpuOrEmpty();
    if (textureIndex >= gpu.textures.size() || !gpu.textures[textureIndex].resident) {
        return {};
    }
    return gpu.textures[textureIndex].image;
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

bool Scene::RotateObject(render::RenderDevice& device, uint32_t objectIndex, const glm::quat& rotationDelta)
{
    if (glm::length(rotationDelta) <= 1.0e-5f) {
        return true;
    }

    const std::shared_ptr<PreparedScene> prepared = _prepared;
    const std::shared_ptr<ParsedScene> parsed = _parsed;
    if (!prepared || objectIndex >= prepared->objects.size()) {
        return false;
    }

    SceneObject& object = prepared->objects[objectIndex];
    const glm::vec3 center = object.bounds.center;
    const glm::mat3 rotation = glm::mat3_cast(glm::normalize(rotationDelta));
    auto rotatePoint = [&](glm::vec3 position) {
        return center + rotation * (position - center);
    };
    auto rotateDirection = [&](glm::vec3 direction) {
        const glm::vec3 rotated = rotation * direction;
        return glm::length(rotated) > 1.0e-5f ? glm::normalize(rotated) : direction;
    };

    object.worldTransform = glm::translate(glm::mat4(1.0f), center) * glm::mat4(rotation)
        * glm::translate(glm::mat4(1.0f), -center) * object.worldTransform;

    if (parsed && objectIndex < parsed->objects.size()) {
        ParsedSceneObject& parsedObject = parsed->objects[objectIndex];
        parsedObject.worldTransform = glm::translate(glm::mat4(1.0f), center) * glm::mat4(rotation)
            * glm::translate(glm::mat4(1.0f), -center) * parsedObject.worldTransform;
        const uint32_t primitiveEnd = parsedObject.firstPrimitive + parsedObject.primitiveCount;
        for (uint32_t primitiveIndex = parsedObject.firstPrimitive; primitiveIndex < primitiveEnd; ++primitiveIndex) {
            parsed->primitives[primitiveIndex].worldTransform = glm::translate(glm::mat4(1.0f), center) * glm::mat4(rotation)
                * glm::translate(glm::mat4(1.0f), -center) * parsed->primitives[primitiveIndex].worldTransform;
        }
    }

    const uint32_t vertexEnd = std::min<uint32_t>(object.firstVertex + object.vertexCount, static_cast<uint32_t>(prepared->vertices.size()));
    for (uint32_t vertexIndex = object.firstVertex; vertexIndex < vertexEnd; ++vertexIndex) {
        SceneVertex& vertex = prepared->vertices[vertexIndex];
        vertex.position = rotatePoint(vertex.position);
        vertex.normal = rotateDirection(vertex.normal);
        vertex.tangent = glm::vec4(rotateDirection(glm::vec3(vertex.tangent)), vertex.tangent.w);
    }
    if (prepared->sceneKind == SceneKind::Gaussian || prepared->sceneKind == SceneKind::PointCloud) {
        for (GaussianPrimitive& gaussian : prepared->gaussians) {
            gaussian.positionOpacity = glm::vec4(rotatePoint(glm::vec3(gaussian.positionOpacity)), gaussian.positionOpacity.w);
            const glm::quat currentRotation(gaussian.rotation.w, gaussian.rotation.x, gaussian.rotation.y, gaussian.rotation.z);
            const glm::quat rotated = glm::normalize(rotationDelta * currentRotation);
            gaussian.rotation = glm::vec4(rotated.x, rotated.y, rotated.z, rotated.w);
        }
    }

    const uint32_t surfaceEnd =
        std::min<uint32_t>(object.firstSurface + object.surfaceCount, static_cast<uint32_t>(prepared->surfaceBounds.size()));
    for (uint32_t surfaceIndex = object.firstSurface; surfaceIndex < surfaceEnd; ++surfaceIndex) {
        prepared->surfaceBounds[surfaceIndex].center = rotatePoint(prepared->surfaceBounds[surfaceIndex].center);
    }

    const uint32_t triangleEnd =
        std::min<uint32_t>(object.firstTriangle + object.triangleCount, static_cast<uint32_t>(prepared->triangles.size()));
    for (uint32_t triangleIndex = object.firstTriangle; triangleIndex < triangleEnd; ++triangleIndex) {
        SceneTriangle& triangle = prepared->triangles[triangleIndex];
        triangle.p0 = glm::vec4(rotatePoint(glm::vec3(triangle.p0)), triangle.p0.w);
        triangle.p1 = glm::vec4(rotatePoint(glm::vec3(triangle.p1)), triangle.p1.w);
        triangle.p2 = glm::vec4(rotatePoint(glm::vec3(triangle.p2)), triangle.p2.w);
        triangle.n0 = glm::vec4(rotateDirection(glm::vec3(triangle.n0)), triangle.n0.w);
        triangle.n1 = glm::vec4(rotateDirection(glm::vec3(triangle.n1)), triangle.n1.w);
        triangle.n2 = glm::vec4(rotateDirection(glm::vec3(triangle.n2)), triangle.n2.w);
    }

    object.bounds = ComputeVertexRangeBounds(prepared->vertices, object.firstVertex, object.vertexCount);
    FinalizeBounds(prepared->bounds, prepared->vertices);

    if (_gpu == nullptr) {
        ++_contentVersion;
        return true;
    }

    GpuScene& gpu = *_gpu;
    const uint32_t gpuVertexEnd =
        std::min<uint32_t>(object.firstVertex + object.vertexCount, static_cast<uint32_t>(gpu.rasterVertices.size()));
    for (uint32_t vertexIndex = object.firstVertex; vertexIndex < gpuVertexEnd; ++vertexIndex) {
        SceneVertex& vertex = gpu.rasterVertices[vertexIndex];
        vertex.position = rotatePoint(vertex.position);
        vertex.normal = rotateDirection(vertex.normal);
        vertex.tangent = glm::vec4(rotateDirection(glm::vec3(vertex.tangent)), vertex.tangent.w);
    }
    if (prepared->sceneKind == SceneKind::Gaussian || prepared->sceneKind == SceneKind::PointCloud) {
        for (GaussianPrimitive& gaussian : gpu.gaussians) {
            gaussian.positionOpacity = glm::vec4(rotatePoint(glm::vec3(gaussian.positionOpacity)), gaussian.positionOpacity.w);
            const glm::quat currentRotation(gaussian.rotation.w, gaussian.rotation.x, gaussian.rotation.y, gaussian.rotation.z);
            const glm::quat rotated = glm::normalize(rotationDelta * currentRotation);
            gaussian.rotation = glm::vec4(rotated.x, rotated.y, rotated.z, rotated.w);
        }
    }
    const uint32_t gpuTriangleEnd =
        std::min<uint32_t>(object.firstTriangle + object.triangleCount, static_cast<uint32_t>(gpu.triangles.size()));
    for (uint32_t triangleIndex = object.firstTriangle; triangleIndex < gpuTriangleEnd; ++triangleIndex) {
        SceneTriangle& triangle = gpu.triangles[triangleIndex];
        triangle.p0 = glm::vec4(rotatePoint(glm::vec3(triangle.p0)), triangle.p0.w);
        triangle.p1 = glm::vec4(rotatePoint(glm::vec3(triangle.p1)), triangle.p1.w);
        triangle.p2 = glm::vec4(rotatePoint(glm::vec3(triangle.p2)), triangle.p2.w);
        triangle.n0 = glm::vec4(rotateDirection(glm::vec3(triangle.n0)), triangle.n0.w);
        triangle.n1 = glm::vec4(rotateDirection(glm::vec3(triangle.n1)), triangle.n1.w);
        triangle.n2 = glm::vec4(rotateDirection(glm::vec3(triangle.n2)), triangle.n2.w);
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

bool Scene::ScaleObject(render::RenderDevice& device, uint32_t objectIndex, float uniformScale)
{
    if (!std::isfinite(uniformScale) || uniformScale <= 0.0f) {
        return false;
    }
    if (std::abs(uniformScale - 1.0f) <= 1.0e-5f) {
        return true;
    }

    const std::shared_ptr<PreparedScene> prepared = _prepared;
    const std::shared_ptr<ParsedScene> parsed = _parsed;
    if (!prepared || objectIndex >= prepared->objects.size()) {
        return false;
    }

    SceneObject& object = prepared->objects[objectIndex];
    const glm::vec3 center = object.bounds.center;
    auto scalePoint = [&](glm::vec3 position) {
        return center + (position - center) * uniformScale;
    };

    object.worldTransform = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(uniformScale))
        * glm::translate(glm::mat4(1.0f), -center) * object.worldTransform;
    if (parsed && objectIndex < parsed->objects.size()) {
        ParsedSceneObject& parsedObject = parsed->objects[objectIndex];
        parsedObject.worldTransform = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(uniformScale))
            * glm::translate(glm::mat4(1.0f), -center) * parsedObject.worldTransform;
        const uint32_t primitiveEnd = parsedObject.firstPrimitive + parsedObject.primitiveCount;
        for (uint32_t primitiveIndex = parsedObject.firstPrimitive; primitiveIndex < primitiveEnd; ++primitiveIndex) {
            parsed->primitives[primitiveIndex].worldTransform =
                glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), glm::vec3(uniformScale))
                * glm::translate(glm::mat4(1.0f), -center) * parsed->primitives[primitiveIndex].worldTransform;
        }
    }

    const uint32_t vertexEnd = std::min<uint32_t>(object.firstVertex + object.vertexCount, static_cast<uint32_t>(prepared->vertices.size()));
    for (uint32_t vertexIndex = object.firstVertex; vertexIndex < vertexEnd; ++vertexIndex) {
        prepared->vertices[vertexIndex].position = scalePoint(prepared->vertices[vertexIndex].position);
    }
    if (prepared->sceneKind == SceneKind::Gaussian || prepared->sceneKind == SceneKind::PointCloud) {
        for (GaussianPrimitive& gaussian : prepared->gaussians) {
            gaussian.positionOpacity = glm::vec4(scalePoint(glm::vec3(gaussian.positionOpacity)), gaussian.positionOpacity.w);
            gaussian.scale *= glm::vec4(glm::vec3(uniformScale), 1.0f);
        }
    }

    const uint32_t surfaceEnd =
        std::min<uint32_t>(object.firstSurface + object.surfaceCount, static_cast<uint32_t>(prepared->surfaceBounds.size()));
    for (uint32_t surfaceIndex = object.firstSurface; surfaceIndex < surfaceEnd; ++surfaceIndex) {
        prepared->surfaceBounds[surfaceIndex].center = scalePoint(prepared->surfaceBounds[surfaceIndex].center);
        prepared->surfaceBounds[surfaceIndex].radius *= uniformScale;
    }

    const uint32_t triangleEnd =
        std::min<uint32_t>(object.firstTriangle + object.triangleCount, static_cast<uint32_t>(prepared->triangles.size()));
    for (uint32_t triangleIndex = object.firstTriangle; triangleIndex < triangleEnd; ++triangleIndex) {
        SceneTriangle& triangle = prepared->triangles[triangleIndex];
        triangle.p0 = glm::vec4(scalePoint(glm::vec3(triangle.p0)), triangle.p0.w);
        triangle.p1 = glm::vec4(scalePoint(glm::vec3(triangle.p1)), triangle.p1.w);
        triangle.p2 = glm::vec4(scalePoint(glm::vec3(triangle.p2)), triangle.p2.w);
    }

    object.bounds = ComputeVertexRangeBounds(prepared->vertices, object.firstVertex, object.vertexCount);
    FinalizeBounds(prepared->bounds, prepared->vertices);

    if (_gpu == nullptr) {
        ++_contentVersion;
        return true;
    }

    GpuScene& gpu = *_gpu;
    const uint32_t gpuVertexEnd =
        std::min<uint32_t>(object.firstVertex + object.vertexCount, static_cast<uint32_t>(gpu.rasterVertices.size()));
    for (uint32_t vertexIndex = object.firstVertex; vertexIndex < gpuVertexEnd; ++vertexIndex) {
        gpu.rasterVertices[vertexIndex].position = scalePoint(gpu.rasterVertices[vertexIndex].position);
    }
    if (prepared->sceneKind == SceneKind::Gaussian || prepared->sceneKind == SceneKind::PointCloud) {
        for (GaussianPrimitive& gaussian : gpu.gaussians) {
            gaussian.positionOpacity = glm::vec4(scalePoint(glm::vec3(gaussian.positionOpacity)), gaussian.positionOpacity.w);
            gaussian.scale *= glm::vec4(glm::vec3(uniformScale), 1.0f);
        }
    }
    const uint32_t gpuTriangleEnd =
        std::min<uint32_t>(object.firstTriangle + object.triangleCount, static_cast<uint32_t>(gpu.triangles.size()));
    for (uint32_t triangleIndex = object.firstTriangle; triangleIndex < gpuTriangleEnd; ++triangleIndex) {
        SceneTriangle& triangle = gpu.triangles[triangleIndex];
        triangle.p0 = glm::vec4(scalePoint(glm::vec3(triangle.p0)), triangle.p0.w);
        triangle.p1 = glm::vec4(scalePoint(glm::vec3(triangle.p1)), triangle.p1.w);
        triangle.p2 = glm::vec4(scalePoint(glm::vec3(triangle.p2)), triangle.p2.w);
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

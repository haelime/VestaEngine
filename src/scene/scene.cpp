#include <vesta/scene/scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <glm/common.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <vesta/render/rhi/render_device.h>

namespace vesta::scene {
namespace {
constexpr auto kLoadOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::LoadGLBBuffers
    | fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices;

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

    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation)
        * glm::scale(glm::mat4(1.0f), scale);
}

std::vector<glm::vec3> ReadVec3Accessor(const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
{
    std::vector<glm::vec3> data(accessor.count);
    fastgltf::copyFromAccessor<glm::vec3>(asset, accessor, data.data());
    return data;
}

std::vector<uint32_t> ReadIndexAccessor(const fastgltf::Asset& asset, const fastgltf::Accessor& accessor)
{
    std::vector<uint32_t> data(accessor.count);
    fastgltf::copyFromAccessor<uint32_t>(asset, accessor, data.data());
    return data;
}

glm::vec4 ReadBaseColor(const fastgltf::Asset& asset, const fastgltf::Primitive& primitive)
{
    if (!primitive.materialIndex.has_value()) {
        return glm::vec4(0.8f, 0.8f, 0.85f, 1.0f);
    }

    const fastgltf::Material& material = asset.materials[primitive.materialIndex.value()];
    return glm::vec4(material.pbrData.baseColorFactor[0],
        material.pbrData.baseColorFactor[1],
        material.pbrData.baseColorFactor[2],
        material.pbrData.baseColorFactor[3]);
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

template <typename T>
void CopyToMappedBuffer(vesta::render::RenderDevice& device, vesta::render::BufferHandle handle, std::span<const T> data)
{
    if (data.empty()) {
        return;
    }

    const vesta::render::AllocatedBuffer& buffer = device.GetBufferResource(handle);
    std::memcpy(buffer.allocationInfo.pMappedData, data.data(), data.size_bytes());
}
} // namespace

bool Scene::LoadFromFile(const std::filesystem::path& path)
{
    _sourcePath.clear();
    _vertices.clear();
    _indices.clear();
    _triangles.clear();
    _surfaces.clear();
    _bounds = {};

    if (!std::filesystem::exists(path)) {
        return false;
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

    const size_t sceneIndex = gltf.defaultScene.value_or(0);
    const fastgltf::Scene& rootScene = gltf.scenes.at(sceneIndex);

    std::function<void(size_t, const glm::mat4&)> appendNode = [&](size_t nodeIndex, const glm::mat4& parentMatrix) {
        const fastgltf::Node& node = gltf.nodes.at(nodeIndex);
        const glm::mat4 worldMatrix = parentMatrix * NodeToMatrix(node);
        const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(worldMatrix)));

        if (node.meshIndex.has_value()) {
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

                std::vector<uint32_t> primitiveIndices;
                if (primitive.indicesAccessor.has_value()) {
                    primitiveIndices = ReadIndexAccessor(gltf, gltf.accessors.at(primitive.indicesAccessor.value()));
                } else {
                    primitiveIndices.resize(positions.size());
                    for (uint32_t i = 0; i < static_cast<uint32_t>(primitiveIndices.size()); ++i) {
                        primitiveIndices[i] = i;
                    }
                }

                const glm::vec4 albedo = ReadBaseColor(gltf, primitive);
                const uint32_t baseVertex = static_cast<uint32_t>(_vertices.size());
                _vertices.reserve(_vertices.size() + positions.size());

                for (size_t i = 0; i < positions.size(); ++i) {
                    const glm::vec3 worldPosition = glm::vec3(worldMatrix * glm::vec4(positions[i], 1.0f));
                    glm::vec3 worldNormal = glm::normalize(normalMatrix * normals[i]);
                    const bool normalFinite = std::isfinite(worldNormal.x) && std::isfinite(worldNormal.y)
                        && std::isfinite(worldNormal.z);
                    if (!normalFinite || glm::length(worldNormal) < 0.001f) {
                        worldNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                    }

                    _vertices.push_back(SceneVertex{
                        .position = worldPosition,
                        .normal = worldNormal,
                        .color = albedo,
                    });
                }

                if (!hasNormals) {
                    for (size_t triangle = 0; triangle + 2 < primitiveIndices.size(); triangle += 3) {
                        const uint32_t i0 = baseVertex + primitiveIndices[triangle + 0];
                        const uint32_t i1 = baseVertex + primitiveIndices[triangle + 1];
                        const uint32_t i2 = baseVertex + primitiveIndices[triangle + 2];
                        const glm::vec3 faceNormal = glm::normalize(
                            glm::cross(_vertices[i1].position - _vertices[i0].position, _vertices[i2].position - _vertices[i0].position));
                        _vertices[i0].normal = faceNormal;
                        _vertices[i1].normal = faceNormal;
                        _vertices[i2].normal = faceNormal;
                    }
                }

                const uint32_t firstIndex = static_cast<uint32_t>(_indices.size());
                _indices.reserve(_indices.size() + primitiveIndices.size());
                for (uint32_t index : primitiveIndices) {
                    _indices.push_back(baseVertex + index);
                }
                _surfaces.push_back(SceneSurface{
                    .firstIndex = firstIndex,
                    .indexCount = static_cast<uint32_t>(primitiveIndices.size()),
                });

                for (size_t triangle = 0; triangle + 2 < primitiveIndices.size(); triangle += 3) {
                    const SceneVertex& v0 = _vertices[baseVertex + primitiveIndices[triangle + 0]];
                    const SceneVertex& v1 = _vertices[baseVertex + primitiveIndices[triangle + 1]];
                    const SceneVertex& v2 = _vertices[baseVertex + primitiveIndices[triangle + 2]];
                    _triangles.push_back(SceneTriangle{
                        .p0 = glm::vec4(v0.position, 1.0f),
                        .p1 = glm::vec4(v1.position, 1.0f),
                        .p2 = glm::vec4(v2.position, 1.0f),
                        .albedo = albedo,
                    });
                }
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

    FinalizeBounds(_bounds, _vertices);
    _sourcePath = path;
    return IsLoaded();
}

void Scene::UploadToGpu(vesta::render::RenderDevice& device)
{
    DestroyGpu(device);
    if (!IsLoaded()) {
        return;
    }

    constexpr VmaAllocationCreateFlags kMappedHostFlags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    _vertexBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = sizeof(SceneVertex) * _vertices.size(),
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .registerBindlessStorage = false,
        .debugName = "SceneVertices",
    });

    _indexBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = sizeof(uint32_t) * _indices.size(),
        .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .registerBindlessStorage = false,
        .debugName = "SceneIndices",
    });

    _triangleBuffer = device.CreateBuffer(vesta::render::BufferDesc{
        .size = sizeof(SceneTriangle) * _triangles.size(),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .registerBindlessStorage = true,
        .debugName = "SceneTriangles",
    });

    CopyToMappedBuffer(device, _vertexBuffer, std::span<const SceneVertex>(_vertices));
    CopyToMappedBuffer(device, _indexBuffer, std::span<const uint32_t>(_indices));
    CopyToMappedBuffer(device, _triangleBuffer, std::span<const SceneTriangle>(_triangles));
}

void Scene::DestroyGpu(vesta::render::RenderDevice& device)
{
    if (_triangleBuffer) {
        device.DestroyBuffer(_triangleBuffer);
        _triangleBuffer = {};
    }
    if (_indexBuffer) {
        device.DestroyBuffer(_indexBuffer);
        _indexBuffer = {};
    }
    if (_vertexBuffer) {
        device.DestroyBuffer(_vertexBuffer);
        _vertexBuffer = {};
    }
}
} // namespace vesta::scene

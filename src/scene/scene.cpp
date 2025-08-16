#include <vesta/scene/scene.h>

#include <algorithm>
#include <array>
#include <chrono>
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

#include <vesta/render/renderer.h>
#include <vesta/render/rhi/render_device.h>

namespace vesta::scene {
namespace {
constexpr auto kLoadOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::LoadGLBBuffers
    | fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices;
constexpr VmaAllocationCreateFlags kMappedHostFlags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

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

template <typename T>
void CopyToMappedBuffer(vesta::render::RenderDevice& device, vesta::render::BufferHandle handle, std::span<const T> data)
{
    if (data.empty()) {
        return;
    }

    const vesta::render::AllocatedBuffer& buffer = device.GetBufferResource(handle);
    std::memcpy(buffer.allocationInfo.pMappedData, data.data(), data.size_bytes());
}

VkTransformMatrixKHR MakeIdentityTransformMatrix()
{
    VkTransformMatrixKHR matrix{};
    matrix.matrix[0][0] = 1.0f;
    matrix.matrix[1][1] = 1.0f;
    matrix.matrix[2][2] = 1.0f;
    return matrix;
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
    _prepared.reset();

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

    auto prepared = std::make_shared<PreparedScene>();
    PreparedScene& sceneData = *prepared;

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
                    for (uint32_t index = 0; index < static_cast<uint32_t>(primitiveIndices.size()); ++index) {
                        primitiveIndices[index] = index;
                    }
                }

                const glm::vec4 albedo = ReadBaseColor(gltf, primitive);
                const uint32_t baseVertex = static_cast<uint32_t>(sceneData.vertices.size());
                sceneData.vertices.reserve(sceneData.vertices.size() + positions.size());

                for (size_t vertexIndex = 0; vertexIndex < positions.size(); ++vertexIndex) {
                    const glm::vec3 worldPosition = glm::vec3(worldMatrix * glm::vec4(positions[vertexIndex], 1.0f));
                    glm::vec3 worldNormal = glm::normalize(normalMatrix * normals[vertexIndex]);
                    const bool normalFinite = std::isfinite(worldNormal.x) && std::isfinite(worldNormal.y) && std::isfinite(worldNormal.z);
                    if (!normalFinite || glm::length(worldNormal) < 0.001f) {
                        worldNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                    }

                    sceneData.vertices.push_back(SceneVertex{
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
                            glm::cross(sceneData.vertices[i1].position - sceneData.vertices[i0].position,
                                sceneData.vertices[i2].position - sceneData.vertices[i0].position));
                        sceneData.vertices[i0].normal = faceNormal;
                        sceneData.vertices[i1].normal = faceNormal;
                        sceneData.vertices[i2].normal = faceNormal;
                    }
                }

                const uint32_t firstIndex = static_cast<uint32_t>(sceneData.indices.size());
                sceneData.indices.reserve(sceneData.indices.size() + primitiveIndices.size());
                for (uint32_t index : primitiveIndices) {
                    sceneData.indices.push_back(baseVertex + index);
                }

                sceneData.surfaces.push_back(SceneSurface{
                    .firstIndex = firstIndex,
                    .indexCount = static_cast<uint32_t>(primitiveIndices.size()),
                });
                sceneData.surfaceBounds.push_back(ComputeSurfaceBounds(sceneData.vertices, baseVertex, primitiveIndices));

                for (size_t triangle = 0; triangle + 2 < primitiveIndices.size(); triangle += 3) {
                    const SceneVertex& v0 = sceneData.vertices[baseVertex + primitiveIndices[triangle + 0]];
                    const SceneVertex& v1 = sceneData.vertices[baseVertex + primitiveIndices[triangle + 1]];
                    const SceneVertex& v2 = sceneData.vertices[baseVertex + primitiveIndices[triangle + 2]];
                    sceneData.triangles.push_back(SceneTriangle{
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

    FinalizeBounds(sceneData.bounds, sceneData.vertices);
    sceneData.sourcePath = path;
    _prepared = std::move(prepared);
    return IsLoaded();
}

void Scene::UploadToGpu(vesta::render::RenderDevice& device, const vesta::render::SceneUploadOptions& options)
{
    DestroyGpu(device);
    const PreparedScene& prepared = GetPreparedOrEmpty();
    if (!prepared.IsLoaded()) {
        return;
    }

    _gpu = std::make_unique<GpuScene>();
    GpuScene& gpu = *_gpu;

    const VkBufferUsageFlags vertexUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkBufferUsageFlags indexUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkBufferUsageFlags triangleUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (options.useDeviceLocalSceneBuffers) {
        const auto vertexData = std::span<const SceneVertex>(prepared.vertices);
        const auto indexData = std::span<const uint32_t>(prepared.indices);
        const auto triangleData = std::span<const SceneTriangle>(prepared.triangles);

        const vesta::render::BufferHandle vertexStaging =
            CreateHostBufferAndCopy(device, vertexData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "SceneVerticesStaging", false);
        const vesta::render::BufferHandle indexStaging =
            CreateHostBufferAndCopy(device, indexData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "SceneIndicesStaging", false);
        const vesta::render::BufferHandle triangleStaging =
            CreateHostBufferAndCopy(device, triangleData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "SceneTrianglesStaging", false);

        gpu.vertexBuffer = device.CreateBuffer(vesta::render::BufferDesc{
            .size = sizeof(SceneVertex) * prepared.vertices.size(),
            .usage = vertexUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .registerBindlessStorage = false,
            .debugName = "SceneVertices",
        });
        gpu.indexBuffer = device.CreateBuffer(vesta::render::BufferDesc{
            .size = sizeof(uint32_t) * prepared.indices.size(),
            .usage = indexUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .registerBindlessStorage = false,
            .debugName = "SceneIndices",
        });
        gpu.triangleBuffer = device.CreateBuffer(vesta::render::BufferDesc{
            .size = sizeof(SceneTriangle) * prepared.triangles.size(),
            .usage = triangleUsage,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .registerBindlessStorage = true,
            .debugName = "SceneTriangles",
        });

        device.ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
            const std::array<VkBufferCopy, 1> vertexCopy{
                VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = sizeof(SceneVertex) * prepared.vertices.size() }
            };
            vkCmdCopyBuffer(commandBuffer, device.GetBuffer(vertexStaging), device.GetBuffer(gpu.vertexBuffer), 1, vertexCopy.data());

            const std::array<VkBufferCopy, 1> indexCopy{
                VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = sizeof(uint32_t) * prepared.indices.size() }
            };
            vkCmdCopyBuffer(commandBuffer, device.GetBuffer(indexStaging), device.GetBuffer(gpu.indexBuffer), 1, indexCopy.data());

            const std::array<VkBufferCopy, 1> triangleCopy{
                VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = sizeof(SceneTriangle) * prepared.triangles.size() }
            };
            vkCmdCopyBuffer(commandBuffer, device.GetBuffer(triangleStaging), device.GetBuffer(gpu.triangleBuffer), 1, triangleCopy.data());
        });

        device.DestroyBuffer(vertexStaging);
        device.DestroyBuffer(indexStaging);
        device.DestroyBuffer(triangleStaging);
    } else {
        gpu.vertexBuffer =
            CreateHostBufferAndCopy(device, std::span<const SceneVertex>(prepared.vertices), vertexUsage, "SceneVertices", false);
        gpu.indexBuffer =
            CreateHostBufferAndCopy(device, std::span<const uint32_t>(prepared.indices), indexUsage, "SceneIndices", false);
        gpu.triangleBuffer = CreateHostBufferAndCopy(
            device, std::span<const SceneTriangle>(prepared.triangles), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "SceneTriangles", true);
    }

    if (!options.buildRayTracingStructuresOnLoad || !device.IsRayTracingSupported() || prepared.indices.empty()) {
        return;
    }

    const auto& rt = device.GetRayTracingFunctions();

    auto buildBottomLevel = [&]() {
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
    };

    auto buildTopLevel = [&]() {
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
    };

    const auto bottomLevelStart = std::chrono::steady_clock::now();
    buildBottomLevel();
    gpu.bottomLevelBuildMs =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - bottomLevelStart).count();

    const auto topLevelStart = std::chrono::steady_clock::now();
    buildTopLevel();
    gpu.topLevelBuildMs =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - topLevelStart).count();
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
    if (_gpu->indexBuffer) {
        device.DestroyBuffer(_gpu->indexBuffer);
        _gpu->indexBuffer = {};
    }
    if (_gpu->vertexBuffer) {
        device.DestroyBuffer(_gpu->vertexBuffer);
        _gpu->vertexBuffer = {};
    }
    _gpu.reset();
}
} // namespace vesta::scene

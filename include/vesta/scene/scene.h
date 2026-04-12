#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vesta/render/resources/resource_handles.h>

namespace vesta::render {
class RenderDevice;
struct SceneUploadOptions;
}

namespace vesta::scene {
enum class SceneKind : uint32_t {
    Empty = 0,
    Mesh = 1,
    PointCloud = 2,
    Gaussian = 3,
};

enum class SceneTextureSemantic : uint32_t {
    BaseColor = 0,
    MetallicRoughness = 1,
    Normal = 2,
    Occlusion = 3,
    Emissive = 4,
};

// CPU-side vertex layout used by rasterization and as the source data for BLAS builds.
struct SceneVertex {
    glm::vec3 position{ 0.0f };
    glm::vec3 normal{ 0.0f, 1.0f, 0.0f };
    glm::vec4 tangent{ 1.0f, 0.0f, 0.0f, 1.0f };
    glm::vec4 color{ 0.8f, 0.8f, 0.8f, 1.0f };
    glm::vec2 texCoord{ 0.0f };
    glm::vec2 splatParams{ 1.0f, 1.0f };
    uint32_t materialIndex{ 0 };
};

struct SceneMaterial {
    glm::vec4 baseColorFactor{ 0.8f, 0.8f, 0.85f, 1.0f };
    glm::vec4 emissiveFactor{ 0.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 materialParams{ 1.0f, 1.0f, 1.0f, 1.0f }; // metallic, roughness, occlusion strength, normal scale
    glm::uvec4 textureIndices0{ render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex }; // baseColor, metallicRoughness, normal, occlusion
    glm::uvec4 textureIndices1{ render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex }; // emissive, reserved, reserved, reserved
};

struct SceneTriangle {
    glm::vec4 p0{ 0.0f };
    glm::vec4 p1{ 0.0f };
    glm::vec4 p2{ 0.0f };
    glm::vec4 n0{ 0.0f, 1.0f, 0.0f, 0.0f };
    glm::vec4 n1{ 0.0f, 1.0f, 0.0f, 0.0f };
    glm::vec4 n2{ 0.0f, 1.0f, 0.0f, 0.0f };
    glm::vec4 uv0{ 0.0f };
    glm::vec4 uv1{ 0.0f };
    glm::vec4 uv2{ 0.0f };
    glm::vec4 baseColorFactor{ 0.8f, 0.8f, 0.85f, 1.0f };
    glm::vec4 emissiveFactor{ 0.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 materialParams{ 1.0f, 1.0f, 1.0f, 1.0f };
    glm::uvec4 textureIndices0{ render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex };
    glm::uvec4 textureIndices1{ render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex,
        render::kInvalidResourceIndex };
};

struct SceneSurface {
    uint32_t firstIndex{ 0 };
    uint32_t indexCount{ 0 };
};

struct SceneSurfaceBounds {
    glm::vec3 center{ 0.0f };
    float radius{ 0.0f };
};

struct ParsedPrimitive {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> texCoords;
    std::vector<uint32_t> indices;
    glm::mat4 worldTransform{ 1.0f };
    uint32_t objectIndex{ 0 };
    uint32_t materialIndex{ 0 };
    uint32_t preparedBaseVertex{ 0 };
    uint32_t preparedFirstTriangle{ 0 };
    bool hasNormals{ false };
    bool hasTangents{ false };
};

struct ParsedSceneObject {
    std::string name;
    glm::mat4 initialWorldTransform{ 1.0f };
    glm::mat4 worldTransform{ 1.0f };
    uint32_t firstPrimitive{ 0 };
    uint32_t primitiveCount{ 0 };
};

struct SceneTextureAsset {
    std::string name;
    uint32_t width{ 0 };
    uint32_t height{ 0 };
    bool srgb{ true };
    std::vector<uint8_t> rgba8Pixels;

    [[nodiscard]] bool IsValid() const { return width > 0 && height > 0 && !rgba8Pixels.empty(); }
};

// ParsedScene is the asset-loading result before it is flattened into the
// renderer-friendly PreparedScene buffers.
struct ParsedScene {
    std::filesystem::path sourcePath;
    std::vector<SceneTextureAsset> textures;
    std::vector<SceneMaterial> materials;
    std::vector<ParsedPrimitive> primitives;
    std::vector<ParsedSceneObject> objects;
    std::vector<SceneVertex> gaussianVertices;
    SceneKind sceneKind{ SceneKind::Empty };
    bool gaussianUsesNativeScale{ false };

    [[nodiscard]] bool IsLoaded() const
    {
        return !primitives.empty() || !gaussianVertices.empty() || !textures.empty() || !objects.empty();
    }
};

struct SceneBounds {
    glm::vec3 minimum{ 0.0f };
    glm::vec3 maximum{ 0.0f };
    glm::vec3 center{ 0.0f };
    float radius{ 1.0f };
};

enum class SceneUploadResource : uint32_t {
    Vertex = 0,
    Index = 1,
    Triangle = 2,
    Material = 3,
};

struct SceneObject {
    std::string name;
    glm::mat4 initialWorldTransform{ 1.0f };
    glm::mat4 worldTransform{ 1.0f };
    SceneBounds bounds{};
    uint32_t firstPrimitive{ 0 };
    uint32_t primitiveCount{ 0 };
    uint32_t firstVertex{ 0 };
    uint32_t vertexCount{ 0 };
    uint32_t firstSurface{ 0 };
    uint32_t surfaceCount{ 0 };
    uint32_t firstTriangle{ 0 };
    uint32_t triangleCount{ 0 };

    [[nodiscard]] glm::vec3 GetTranslation() const { return glm::vec3(worldTransform[3]); }
};

// PreparedScene contains CPU-side data that worker threads can safely read
// without touching Vulkan state.
struct PreparedScene {
    std::filesystem::path sourcePath;
    SceneKind sceneKind{ SceneKind::Empty };
    std::vector<SceneVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SceneTriangle> triangles;
    std::vector<SceneMaterial> materials;
    std::vector<SceneSurface> surfaces;
    std::vector<SceneSurfaceBounds> surfaceBounds;
    std::vector<SceneTextureAsset> textures;
    SceneBounds bounds{};
    std::vector<SceneObject> objects;

    [[nodiscard]] bool IsLoaded() const { return !vertices.empty(); }
    [[nodiscard]] bool HasRasterGeometry() const { return !indices.empty() && !surfaces.empty(); }
    [[nodiscard]] bool HasGaussianSplats() const { return !vertices.empty(); }
};

struct GpuSceneTexture {
    render::ImageHandle image{};
    uint32_t bindlessSampledImage{ render::kInvalidResourceIndex };
    bool resident{ false };
};

// GpuScene owns only Vulkan-side resources. Keeping it separate makes deferred
// destruction and background CPU preparation much easier to reason about.
struct GpuScene {
    render::BufferHandle vertexBuffer{};
    render::BufferHandle indexBuffer{};
    render::BufferHandle triangleBuffer{};
    render::BufferHandle materialBuffer{};
    render::BufferHandle bottomLevelBuffer{};
    render::BufferHandle topLevelBuffer{};
    std::vector<SceneVertex> rasterVertices;
    std::vector<SceneTriangle> triangles;
    std::vector<SceneMaterial> materials;
    std::vector<GpuSceneTexture> textures;
    VkAccelerationStructureKHR bottomLevelAccelerationStructure{ VK_NULL_HANDLE };
    VkAccelerationStructureKHR topLevelAccelerationStructure{ VK_NULL_HANDLE };
    float geometryUploadMs{ 0.0f };
    float textureUploadMs{ 0.0f };
    float bottomLevelBuildMs{ 0.0f };
    float topLevelBuildMs{ 0.0f };
};

class Scene {
public:
    Scene() = default;
    Scene(Scene&&) noexcept = default;
    Scene& operator=(Scene&&) noexcept = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    // ParseFromFile reads the source asset into a CPU-side intermediate form.
    bool ParseFromFile(const std::filesystem::path& path);
    // PrepareParsedScene flattens ParsedScene into the CPU buffers consumed by
    // raster, Gaussian, and path tracing passes.
    bool PrepareParsedScene();
    // LoadFromFile is the convenience wrapper used by most call sites today.
    bool LoadFromFile(const std::filesystem::path& path);
    void AllocateGpuResources(render::RenderDevice& device, const render::SceneUploadOptions& options);
    void UploadGpuResourceChunk(render::RenderDevice& device, SceneUploadResource resource, size_t offsetBytes, size_t sizeBytes);
    void UploadGpuTexture(render::RenderDevice& device, size_t textureIndex);
    void BuildBottomLevelAccelerationStructure(render::RenderDevice& device);
    void BuildTopLevelAccelerationStructure(render::RenderDevice& device);
    void UploadToGpu(render::RenderDevice& device, const render::SceneUploadOptions& options);
    void DestroyGpu(render::RenderDevice& device);

    [[nodiscard]] bool IsLoaded() const { return GetPreparedOrEmpty().IsLoaded(); }
    [[nodiscard]] SceneKind GetSceneKind() const { return GetPreparedOrEmpty().sceneKind; }
    [[nodiscard]] bool HasRasterGeometry() const
    {
        return GetPreparedOrEmpty().HasRasterGeometry() && GetGpuOrEmpty().indexBuffer && GetGpuOrEmpty().materialBuffer;
    }
    [[nodiscard]] bool HasGaussianSplats() const
    {
        const SceneKind kind = GetPreparedOrEmpty().sceneKind;
        return (kind == SceneKind::PointCloud || kind == SceneKind::Gaussian) && !GetPreparedOrEmpty().vertices.empty()
            && GetGpuOrEmpty().vertexBuffer;
    }
    [[nodiscard]] const std::filesystem::path& GetSourcePath() const { return GetPreparedOrEmpty().sourcePath; }
    [[nodiscard]] const std::vector<SceneVertex>& GetVertices() const { return GetPreparedOrEmpty().vertices; }
    [[nodiscard]] const std::vector<uint32_t>& GetIndices() const { return GetPreparedOrEmpty().indices; }
    [[nodiscard]] const std::vector<SceneTriangle>& GetTriangles() const { return GetPreparedOrEmpty().triangles; }
    [[nodiscard]] const std::vector<SceneMaterial>& GetMaterials() const { return GetPreparedOrEmpty().materials; }
    [[nodiscard]] const std::vector<SceneSurface>& GetSurfaces() const { return GetPreparedOrEmpty().surfaces; }
    [[nodiscard]] const std::vector<SceneSurfaceBounds>& GetSurfaceBounds() const { return GetPreparedOrEmpty().surfaceBounds; }
    [[nodiscard]] const std::vector<SceneTextureAsset>& GetTextures() const { return GetPreparedOrEmpty().textures; }
    [[nodiscard]] const std::vector<SceneObject>& GetObjects() const { return GetPreparedOrEmpty().objects; }
    [[nodiscard]] const SceneBounds& GetBounds() const { return GetPreparedOrEmpty().bounds; }
    [[nodiscard]] std::shared_ptr<const ParsedScene> GetParsedScene() const { return _parsed; }
    [[nodiscard]] std::shared_ptr<const PreparedScene> GetPreparedScene() const { return _prepared; }

    [[nodiscard]] render::BufferHandle GetVertexBuffer() const { return GetGpuOrEmpty().vertexBuffer; }
    [[nodiscard]] render::BufferHandle GetIndexBuffer() const { return GetGpuOrEmpty().indexBuffer; }
    [[nodiscard]] render::BufferHandle GetTriangleBuffer() const { return GetGpuOrEmpty().triangleBuffer; }
    [[nodiscard]] render::BufferHandle GetMaterialBuffer() const { return GetGpuOrEmpty().materialBuffer; }
    [[nodiscard]] bool HasRayTracingScene() const { return GetGpuOrEmpty().topLevelAccelerationStructure != VK_NULL_HANDLE; }
    [[nodiscard]] size_t GetResidentTextureCount() const;
    [[nodiscard]] bool HasResidentTexture(size_t textureIndex) const;
    [[nodiscard]] uint32_t GetTextureBindlessIndex(size_t textureIndex) const;
    [[nodiscard]] float GetGeometryUploadMs() const { return GetGpuOrEmpty().geometryUploadMs; }
    [[nodiscard]] float GetTextureUploadMs() const { return GetGpuOrEmpty().textureUploadMs; }
    [[nodiscard]] VkAccelerationStructureKHR GetTopLevelAccelerationStructure() const
    {
        return GetGpuOrEmpty().topLevelAccelerationStructure;
    }
    [[nodiscard]] float GetBottomLevelBuildMs() const { return GetGpuOrEmpty().bottomLevelBuildMs; }
    [[nodiscard]] float GetTopLevelBuildMs() const { return GetGpuOrEmpty().topLevelBuildMs; }
    [[nodiscard]] bool SupportsObjectEditing() const { return !GetPreparedOrEmpty().objects.empty(); }
    [[nodiscard]] std::optional<uint32_t> PickObject(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const;
    bool TranslateObject(render::RenderDevice& device, uint32_t objectIndex, const glm::vec3& deltaWorld);
    bool RebuildRayTracing(render::RenderDevice& device);

private:
    [[nodiscard]] static const PreparedScene& EmptyPreparedScene();
    [[nodiscard]] static const GpuScene& EmptyGpuScene();
    [[nodiscard]] const PreparedScene& GetPreparedOrEmpty() const;
    [[nodiscard]] const GpuScene& GetGpuOrEmpty() const;

    std::shared_ptr<ParsedScene> _parsed;
    std::shared_ptr<PreparedScene> _prepared;
    std::unique_ptr<GpuScene> _gpu;
};
} // namespace vesta::scene

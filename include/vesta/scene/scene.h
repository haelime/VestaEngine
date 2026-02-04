#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vesta/render/resources/resource_handles.h>

namespace vesta::render {
class RenderDevice;
struct SceneUploadOptions;
}

namespace vesta::scene {
// CPU-side vertex layout used by rasterization and as the source data for BLAS builds.
struct SceneVertex {
    glm::vec3 position{ 0.0f };
    glm::vec3 normal{ 0.0f, 1.0f, 0.0f };
    glm::vec4 color{ 0.8f, 0.8f, 0.8f, 1.0f };
};

struct SceneTriangle {
    glm::vec4 p0{ 0.0f };
    glm::vec4 p1{ 0.0f };
    glm::vec4 p2{ 0.0f };
    glm::vec4 albedo{ 0.8f, 0.8f, 0.8f, 1.0f };
};

struct SceneSurface {
    uint32_t firstIndex{ 0 };
    uint32_t indexCount{ 0 };
};

struct SceneSurfaceBounds {
    glm::vec3 center{ 0.0f };
    float radius{ 0.0f };
};

struct SceneBounds {
    glm::vec3 minimum{ 0.0f };
    glm::vec3 maximum{ 0.0f };
    glm::vec3 center{ 0.0f };
    float radius{ 1.0f };
};

// PreparedScene contains CPU-side data that worker threads can safely read
// without touching Vulkan state.
struct PreparedScene {
    std::filesystem::path sourcePath;
    std::vector<SceneVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SceneTriangle> triangles;
    std::vector<SceneSurface> surfaces;
    std::vector<SceneSurfaceBounds> surfaceBounds;
    SceneBounds bounds{};

    [[nodiscard]] bool IsLoaded() const { return !vertices.empty() && !indices.empty(); }
};

// GpuScene owns only Vulkan-side resources. Keeping it separate makes deferred
// destruction and background CPU preparation much easier to reason about.
struct GpuScene {
    render::BufferHandle vertexBuffer{};
    render::BufferHandle indexBuffer{};
    render::BufferHandle triangleBuffer{};
    render::BufferHandle bottomLevelBuffer{};
    render::BufferHandle topLevelBuffer{};
    VkAccelerationStructureKHR bottomLevelAccelerationStructure{ VK_NULL_HANDLE };
    VkAccelerationStructureKHR topLevelAccelerationStructure{ VK_NULL_HANDLE };
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

    // LoadFromFile() builds a simple prepared scene representation that both
    // raster and path tracing passes can consume.
    bool LoadFromFile(const std::filesystem::path& path);
    void UploadToGpu(render::RenderDevice& device, const render::SceneUploadOptions& options);
    void DestroyGpu(render::RenderDevice& device);

    [[nodiscard]] bool IsLoaded() const { return GetPreparedOrEmpty().IsLoaded(); }
    [[nodiscard]] const std::filesystem::path& GetSourcePath() const { return GetPreparedOrEmpty().sourcePath; }
    [[nodiscard]] const std::vector<SceneVertex>& GetVertices() const { return GetPreparedOrEmpty().vertices; }
    [[nodiscard]] const std::vector<uint32_t>& GetIndices() const { return GetPreparedOrEmpty().indices; }
    [[nodiscard]] const std::vector<SceneTriangle>& GetTriangles() const { return GetPreparedOrEmpty().triangles; }
    [[nodiscard]] const std::vector<SceneSurface>& GetSurfaces() const { return GetPreparedOrEmpty().surfaces; }
    [[nodiscard]] const std::vector<SceneSurfaceBounds>& GetSurfaceBounds() const { return GetPreparedOrEmpty().surfaceBounds; }
    [[nodiscard]] const SceneBounds& GetBounds() const { return GetPreparedOrEmpty().bounds; }
    [[nodiscard]] std::shared_ptr<const PreparedScene> GetPreparedScene() const { return _prepared; }

    [[nodiscard]] render::BufferHandle GetVertexBuffer() const { return GetGpuOrEmpty().vertexBuffer; }
    [[nodiscard]] render::BufferHandle GetIndexBuffer() const { return GetGpuOrEmpty().indexBuffer; }
    [[nodiscard]] render::BufferHandle GetTriangleBuffer() const { return GetGpuOrEmpty().triangleBuffer; }
    [[nodiscard]] bool HasRayTracingScene() const { return GetGpuOrEmpty().topLevelAccelerationStructure != VK_NULL_HANDLE; }
    [[nodiscard]] VkAccelerationStructureKHR GetTopLevelAccelerationStructure() const
    {
        return GetGpuOrEmpty().topLevelAccelerationStructure;
    }
    [[nodiscard]] float GetBottomLevelBuildMs() const { return GetGpuOrEmpty().bottomLevelBuildMs; }
    [[nodiscard]] float GetTopLevelBuildMs() const { return GetGpuOrEmpty().topLevelBuildMs; }

private:
    [[nodiscard]] static const PreparedScene& EmptyPreparedScene();
    [[nodiscard]] static const GpuScene& EmptyGpuScene();
    [[nodiscard]] const PreparedScene& GetPreparedOrEmpty() const;
    [[nodiscard]] const GpuScene& GetGpuOrEmpty() const;

    std::shared_ptr<PreparedScene> _prepared;
    std::unique_ptr<GpuScene> _gpu;
};
} // namespace vesta::scene

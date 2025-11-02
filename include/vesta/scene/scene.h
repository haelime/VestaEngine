#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

#include <vesta/render/resources/resource_handles.h>

namespace vesta::render {
class RenderDevice;
}

namespace vesta::scene {
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

struct SceneBounds {
    glm::vec3 minimum{ 0.0f };
    glm::vec3 maximum{ 0.0f };
    glm::vec3 center{ 0.0f };
    float radius{ 1.0f };
};

class Scene {
public:
    bool LoadFromFile(const std::filesystem::path& path);
    void UploadToGpu(render::RenderDevice& device);
    void DestroyGpu(render::RenderDevice& device);

    [[nodiscard]] bool IsLoaded() const { return !_vertices.empty() && !_indices.empty(); }
    [[nodiscard]] const std::filesystem::path& GetSourcePath() const { return _sourcePath; }
    [[nodiscard]] const std::vector<SceneVertex>& GetVertices() const { return _vertices; }
    [[nodiscard]] const std::vector<uint32_t>& GetIndices() const { return _indices; }
    [[nodiscard]] const std::vector<SceneTriangle>& GetTriangles() const { return _triangles; }
    [[nodiscard]] const std::vector<SceneSurface>& GetSurfaces() const { return _surfaces; }
    [[nodiscard]] const SceneBounds& GetBounds() const { return _bounds; }

    [[nodiscard]] render::BufferHandle GetVertexBuffer() const { return _vertexBuffer; }
    [[nodiscard]] render::BufferHandle GetIndexBuffer() const { return _indexBuffer; }
    [[nodiscard]] render::BufferHandle GetTriangleBuffer() const { return _triangleBuffer; }

private:
    std::filesystem::path _sourcePath;
    std::vector<SceneVertex> _vertices;
    std::vector<uint32_t> _indices;
    std::vector<SceneTriangle> _triangles;
    std::vector<SceneSurface> _surfaces;
    SceneBounds _bounds{};
    render::BufferHandle _vertexBuffer{};
    render::BufferHandle _indexBuffer{};
    render::BufferHandle _triangleBuffer{};
};
} // namespace vesta::scene

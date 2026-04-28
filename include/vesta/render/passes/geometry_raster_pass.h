#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
// First pass of the frame. It rasterizes scene meshes into GBuffer targets so
// later passes can light or debug the scene without re-drawing geometry.
class GeometryRasterPass final : public IRenderPass {
public:
    void SetTargets(
        GraphTextureHandle albedo,
        GraphTextureHandle normal,
        GraphTextureHandle geometricNormal,
        GraphTextureHandle material,
        GraphTextureHandle debug,
        GraphTextureHandle motion,
        GraphTextureHandle reactive,
        GraphTextureHandle depth);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetVisibleSurfaceIndices(const std::vector<uint32_t>* visibleSurfaceIndices);
    void SetUseIndirectDraw(bool useIndirectDraw);

    [[nodiscard]] std::string_view Name() const override { return "GeometryRasterPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _albedoTarget{};
    GraphTextureHandle _normalTarget{};
    GraphTextureHandle _geometricNormalTarget{};
    GraphTextureHandle _materialTarget{};
    GraphTextureHandle _debugTarget{};
    GraphTextureHandle _motionTarget{};
    GraphTextureHandle _reactiveTarget{};
    GraphTextureHandle _depthTarget{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    const std::vector<uint32_t>* _visibleSurfaceIndices{ nullptr };
    bool _useIndirectDraw{ false };
    glm::mat4 _previousViewProjection{ 1.0f };
    bool _hasPreviousViewProjection{ false };
    BufferHandle _indirectBuffer{};
    size_t _indirectBufferCapacity{ 0 };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

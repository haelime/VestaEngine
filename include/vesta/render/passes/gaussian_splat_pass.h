#pragma once

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
class GaussianSplatPass final : public IRenderPass {
public:
    void SetDepthInput(GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetParams(float pointSize, float opacity, bool enabled);

    [[nodiscard]] std::string_view Name() const override { return "GaussianSplatPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _depthInput{};
    GraphTextureHandle _output{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    float _pointSize{ 6.0f };
    float _opacity{ 0.35f };
    bool _enabled{ true };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

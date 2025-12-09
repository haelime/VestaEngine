#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

namespace vesta::scene {
class Scene;
}

namespace vesta::render {
class ShadowMapPass final : public IRenderPass {
public:
    void SetOutput(GraphTextureHandle shadowMap);
    void SetScene(const vesta::scene::Scene* scene);
    void SetLight(glm::vec4 lightDirectionAndIntensity);

    [[nodiscard]] std::string_view Name() const override { return "ShadowMapPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _shadowMap{};
    const vesta::scene::Scene* _scene{ nullptr };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

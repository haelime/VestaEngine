#pragma once

#include <array>

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::scene {
class Scene;
struct SceneBounds;
}

namespace vesta::render {
struct DirectionalShadowCascade {
    glm::mat4 viewProjection{ 1.0f };
    glm::vec4 atlasScaleOffset{ 1.0f, 1.0f, 0.0f, 0.0f };
    float splitDepth{ 0.0f };
};

[[nodiscard]] glm::mat4 BuildDirectionalShadowViewProjection(
    const vesta::scene::SceneBounds& bounds,
    glm::vec4 lightDirectionAndIntensity);
[[nodiscard]] std::array<DirectionalShadowCascade, 4> BuildDirectionalShadowCascades(const vesta::scene::SceneBounds& bounds,
    const Camera& camera,
    glm::vec4 lightDirectionAndIntensity,
    uint32_t cascadeCount,
    float splitLambda);

class ShadowMapPass final : public IRenderPass {
public:
    void SetOutput(GraphTextureHandle shadowMap);
    void SetScene(const vesta::scene::Scene* scene);
    void SetCamera(const Camera* camera);
    void SetLight(glm::vec4 lightDirectionAndIntensity);
    void SetCascadeSettings(uint32_t cascadeCount, float splitLambda);

    [[nodiscard]] std::string_view Name() const override { return "ShadowMapPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _shadowMap{};
    const vesta::scene::Scene* _scene{ nullptr };
    const Camera* _camera{ nullptr };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    uint32_t _cascadeCount{ 1 };
    float _splitLambda{ 0.65f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _vertexShader{ VK_NULL_HANDLE };
    VkShaderModule _fragmentShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

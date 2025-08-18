#pragma once

#include <glm/glm.hpp>

#include <vesta/render/graph/render_graph.h>

class Camera;

namespace vesta::render {
// Reads the GBuffer and writes lit scene color into a storage image.
// Because lighting happens after geometry, one mesh pass can feed many lighting models.
class DeferredLightingPass final : public IRenderPass {
public:
    void SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth);
    void SetOutput(GraphTextureHandle output);
    void SetCamera(const Camera* camera);
    void SetLight(glm::vec4 lightDirectionAndIntensity);

    [[nodiscard]] std::string_view Name() const override { return "DeferredLightingPass"; }
    void Initialize(RenderDevice& device) override;
    void Setup(RenderGraphBuilder& builder) override;
    void Execute(const RenderGraphContext& context) override;
    void Shutdown(RenderDevice& device) override;

private:
    GraphTextureHandle _albedo{};
    GraphTextureHandle _normal{};
    GraphTextureHandle _material{};
    GraphTextureHandle _depth{};
    GraphTextureHandle _output{};
    const Camera* _camera{ nullptr };
    glm::vec4 _lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    VkPipelineLayout _pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline _pipeline{ VK_NULL_HANDLE };
    VkShaderModule _computeShader{ VK_NULL_HANDLE };
};
} // namespace vesta::render

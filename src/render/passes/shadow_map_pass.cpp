#include <vesta/render/passes/shadow_map_pass.h>

#include <algorithm>
#include <array>
#include <cmath>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
struct ShadowPushConstants {
    glm::mat4 lightViewProjection{ 1.0f };
};

glm::vec3 ValidLightDirection(glm::vec4 lightDirectionAndIntensity)
{
    glm::vec3 lightDirection = glm::vec3(lightDirectionAndIntensity);
    if (glm::dot(lightDirection, lightDirection) <= 1.0e-6f) {
        lightDirection = glm::vec3(-0.4f, -1.0f, -0.3f);
    }
    return glm::normalize(lightDirection);
}

glm::mat4 BuildLightViewProjection(glm::vec3 center, float radius, glm::vec3 lightDirection)
{
    radius = glm::max(radius, 1.0f);
    const glm::vec3 lightPosition = center - lightDirection * radius * 2.5f;
    const glm::vec3 up = std::abs(glm::dot(lightDirection, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.92f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 lightView = glm::lookAt(lightPosition, center, up);
    glm::mat4 lightProjection = glm::ortho(-radius, radius, -radius, radius, 0.05f, radius * 5.5f);
    lightProjection[1][1] *= -1.0f;
    return lightProjection * lightView;
}

glm::vec4 CascadeAtlasScaleOffset(uint32_t cascadeIndex, uint32_t cascadeCount)
{
    if (cascadeCount <= 1u) {
        return glm::vec4(1.0f, 1.0f, 0.0f, 0.0f);
    }
    if (cascadeCount == 2u) {
        return glm::vec4(0.5f, 1.0f, cascadeIndex == 0u ? 0.0f : 0.5f, 0.0f);
    }
    const uint32_t x = cascadeIndex & 1u;
    const uint32_t y = cascadeIndex >> 1u;
    return glm::vec4(0.5f, 0.5f, static_cast<float>(x) * 0.5f, static_cast<float>(y) * 0.5f);
}
} // namespace

glm::mat4 BuildDirectionalShadowViewProjection(
    const vesta::scene::SceneBounds& bounds,
    glm::vec4 lightDirectionAndIntensity)
{
    const float radius = glm::max(bounds.radius, 1.0f);
    return BuildLightViewProjection(bounds.center, radius, ValidLightDirection(lightDirectionAndIntensity));
}

std::array<DirectionalShadowCascade, 4> BuildDirectionalShadowCascades(const vesta::scene::SceneBounds& bounds,
    const Camera& camera,
    glm::vec4 lightDirectionAndIntensity,
    uint32_t cascadeCount,
    float splitLambda)
{
    std::array<DirectionalShadowCascade, 4> cascades{};
    cascadeCount = std::clamp(cascadeCount, 1u, 4u);
    splitLambda = std::clamp(splitLambda, 0.0f, 1.0f);

    const glm::vec3 lightDirection = ValidLightDirection(lightDirectionAndIntensity);
    const glm::vec3 forward = glm::normalize(camera.GetForward());
    glm::vec3 right = glm::cross(forward, camera.GetUp());
    if (glm::dot(right, right) <= 1.0e-6f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    } else {
        right = glm::normalize(right);
    }
    const glm::vec3 up = glm::normalize(glm::cross(right, forward));
    const float nearPlane = camera.GetNearPlane();
    const float farPlane = std::min(camera.GetFarPlane(), std::max(nearPlane + 1.0f, bounds.radius * 6.0f));
    const float aspect = std::max(camera.GetAspectRatio(), 0.01f);
    const float tanHalfFov = std::tan(glm::radians(camera.GetFovDegrees()) * 0.5f);

    float previousSplit = nearPlane;
    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        const float p = static_cast<float>(cascadeIndex + 1u) / static_cast<float>(cascadeCount);
        const float logSplit = nearPlane * std::pow(farPlane / nearPlane, p);
        const float uniformSplit = nearPlane + (farPlane - nearPlane) * p;
        const float splitDepth = splitLambda * logSplit + (1.0f - splitLambda) * uniformSplit;

        std::array<glm::vec3, 8> corners{};
        uint32_t cornerIndex = 0;
        for (float depth : { previousSplit, splitDepth }) {
            const glm::vec3 center = camera.GetPosition() + forward * depth;
            const float halfHeight = depth * tanHalfFov;
            const float halfWidth = halfHeight * aspect;
            corners[cornerIndex++] = center - right * halfWidth - up * halfHeight;
            corners[cornerIndex++] = center + right * halfWidth - up * halfHeight;
            corners[cornerIndex++] = center - right * halfWidth + up * halfHeight;
            corners[cornerIndex++] = center + right * halfWidth + up * halfHeight;
        }

        glm::vec3 center(0.0f);
        for (const glm::vec3& corner : corners) {
            center += corner;
        }
        center /= static_cast<float>(corners.size());

        float radius = 0.0f;
        for (const glm::vec3& corner : corners) {
            radius = std::max(radius, glm::length(corner - center));
        }
        radius = std::max(radius, bounds.radius * 0.12f);

        cascades[cascadeIndex].viewProjection = BuildLightViewProjection(center, radius, lightDirection);
        cascades[cascadeIndex].atlasScaleOffset = CascadeAtlasScaleOffset(cascadeIndex, cascadeCount);
        cascades[cascadeIndex].splitDepth = splitDepth;
        previousSplit = splitDepth;
    }
    for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < 4u; ++cascadeIndex) {
        cascades[cascadeIndex] = cascades[cascadeCount - 1u];
    }
    return cascades;
}

void ShadowMapPass::SetOutput(GraphTextureHandle shadowMap)
{
    _shadowMap = shadowMap;
}

void ShadowMapPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void ShadowMapPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void ShadowMapPass::SetLight(glm::vec4 lightDirectionAndIntensity)
{
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
}

void ShadowMapPass::SetCascadeSettings(uint32_t cascadeCount, float splitLambda)
{
    _cascadeCount = std::clamp(cascadeCount, 1u, 4u);
    _splitLambda = std::clamp(splitLambda, 0.0f, 1.0f);
}

void ShadowMapPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/shadow_depth.vert.spv"));
    _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/shadow_depth.frag.spv"));

    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(ShadowPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, {}, pushConstants);

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(vesta::scene::SceneVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 1> attributes{};
    attributes[0] = VkVertexInputAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, position),
    };

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.depthFormat = VK_FORMAT_D32_SFLOAT;
    pipelineDesc.vertexShader = _vertexShader;
    pipelineDesc.fragmentShader = _fragmentShader;
    pipelineDesc.cullMode = VK_CULL_MODE_BACK_BIT;
    pipelineDesc.depthTestEnable = true;
    pipelineDesc.depthWriteEnable = true;
    pipelineDesc.vertexBindings = { binding };
    pipelineDesc.vertexAttributes = { attributes.begin(), attributes.end() };

    _pipeline = vkutil::create_graphics_pipeline(vkDevice, pipelineDesc);
}

void ShadowMapPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_shadowMap, ResourceUsage::DepthAttachmentWrite);
}

void ShadowMapPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _scene == nullptr || !_scene->HasRasterGeometry()) {
        return;
    }

    const uint32_t cascadeCount = _camera != nullptr ? std::clamp(_cascadeCount, 1u, 4u) : 1u;
    const auto cascades = _camera != nullptr
        ? BuildDirectionalShadowCascades(_scene->GetBounds(), *_camera, _lightDirectionAndIntensity, cascadeCount, _splitLambda)
        : std::array<DirectionalShadowCascade, 4>{ DirectionalShadowCascade{
              .viewProjection = BuildDirectionalShadowViewProjection(_scene->GetBounds(), _lightDirectionAndIntensity),
          } };

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;

    const VkExtent3D shadowExtent = context.GetTextureExtent(_shadowMap);
    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);

    const VkBuffer vertexBuffer = context.GetDevice().GetBuffer(_scene->GetVertexBuffer());
    const VkBuffer indexBuffer = context.GetDevice().GetBuffer(_scene->GetIndexBuffer());
    constexpr VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        const glm::vec4 atlas = cascades[cascadeIndex].atlasScaleOffset;
        const VkRect2D renderArea{
            VkOffset2D{
                static_cast<int32_t>(std::round(atlas.z * static_cast<float>(shadowExtent.width))),
                static_cast<int32_t>(std::round(atlas.w * static_cast<float>(shadowExtent.height))),
            },
            VkExtent2D{
                static_cast<uint32_t>(std::round(atlas.x * static_cast<float>(shadowExtent.width))),
                static_cast<uint32_t>(std::round(atlas.y * static_cast<float>(shadowExtent.height))),
            },
        };

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = context.GetTextureView(_shadowMap);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue = depthClear;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = renderArea;
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        VkViewport viewport{};
        viewport.x = static_cast<float>(renderArea.offset.x);
        viewport.y = static_cast<float>(renderArea.offset.y);
        viewport.width = static_cast<float>(renderArea.extent.width);
        viewport.height = static_cast<float>(renderArea.extent.height);
        viewport.maxDepth = 1.0f;

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &renderArea);

        const ShadowPushConstants pushConstants{
            .lightViewProjection = cascades[cascadeIndex].viewProjection,
        };
        vkCmdPushConstants(commandBuffer, _pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants), &pushConstants);

        for (const vesta::scene::SceneSurface& surface : _scene->GetSurfaces()) {
            vkCmdDrawIndexed(commandBuffer, surface.indexCount, 1, surface.firstIndex, 0, 0);
        }

        vkCmdEndRendering(commandBuffer);
    }
}

void ShadowMapPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _vertexShader);
    vkutil::destroy_shader_module(vkDevice, _fragmentShader);
}
} // namespace vesta::render

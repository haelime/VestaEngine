#include <vesta/render/passes/deferred_lighting_pass.h>

#include <array>
#include <algorithm>
#include <cstring>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>

namespace vesta::render {
namespace {
// Bindless slots are pushed instead of descriptors so the compute shader can
// fetch the right GBuffer images directly from the global bindless set.
struct DeferredLightingPushConstants {
    uint32_t albedoImageIndex{ 0 };
    uint32_t normalImageIndex{ 0 };
    uint32_t materialImageIndex{ 0 };
    uint32_t depthImageIndex{ 0 };
    uint32_t outputImageIndex{ 0 };
    uint32_t debugOutputImageIndex{ kInvalidResourceIndex };
    uint32_t debugView{ 0 };
    uint32_t lightingConstantsBufferIndex{ kInvalidResourceIndex };
    glm::mat4 inverseViewProjection{ 1.0f };
    glm::mat4 viewProjection{ 1.0f };
    glm::vec4 cameraPosition{ 0.0f };
    glm::vec4 lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::vec4 environmentParams{ 1.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 ssaoParams{ 1.0f, 0.75f, 1.35f, 0.0f };
    glm::vec4 ssrParams{ 1.0f, 18.0f, 0.18f, 0.65f };
    glm::vec4 ssgiParams{ 1.0f, 1.4f, 0.32f, 10.0f };
};

static_assert(sizeof(DeferredLightingPushConstants) <= 256, "Deferred lighting push constants must fit common Vulkan limits.");

struct DeferredLightingConstants {
    glm::mat4 lightViewProjection{ 1.0f };
    glm::mat4 cascadeViewProjections[4]{
        glm::mat4(1.0f),
        glm::mat4(1.0f),
        glm::mat4(1.0f),
        glm::mat4(1.0f),
    };
    glm::vec4 cascadeSplits{ 0.0f };
    glm::vec4 cascadeAtlasScaleOffsets[4]{
        glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
        glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
        glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
        glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
    };
    glm::vec4 shadowParams{ 0.0015f, 0.015f, 0.82f, 0.0f }; // bias, normal bias, strength, enabled
    glm::vec4 shadowFilterParams{ 1.0f, 0.0f, 0.0f, 0.0f }; // filter radius, PCSS enabled
    glm::vec4 contactShadowParams{ 1.0f, 1.2f, 0.35f, 0.0f }; // enabled, length, intensity
    glm::uvec4 shadowIndices{ kInvalidResourceIndex, 0u, 0u, 0u };
    glm::uvec4 iblIndices{ kInvalidResourceIndex, kInvalidResourceIndex, kInvalidResourceIndex, kInvalidResourceIndex };
    glm::uvec4 rayEffects{ kInvalidResourceIndex, 0u, 0u, 0u };
    glm::vec4 directionalColor{ 1.0f, 1.0f, 1.0f, 0.0f };
    glm::vec4 pointPositionAndIntensity{ 0.0f, 2.0f, 0.0f, 0.0f };
    glm::vec4 pointColor{ 1.0f, 0.82f, 0.55f, 0.0f };
    glm::vec4 spotPositionAndIntensity{ 0.0f, 3.0f, 2.5f, 0.0f };
    glm::vec4 spotDirectionAndParams{ 0.0f, -0.8f, -0.6f, glm::radians(28.0f) };
    glm::vec4 spotColor{ 1.0f, 0.88f, 0.68f, 0.0f };
    glm::vec4 areaPositionAndIntensity{ 0.0f, 3.2f, 0.0f, 0.0f };
    glm::vec4 areaNormalAndSize{ 0.0f, -1.0f, 0.0f, 2.0f };
    glm::vec4 areaColor{ 0.86f, 0.92f, 1.0f, 0.0f };
};
} // namespace

void DeferredLightingPass::SetInputs(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth)
{
    _albedo = albedo;
    _normal = normal;
    _material = material;
    _depth = depth;
}

void DeferredLightingPass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void DeferredLightingPass::SetDebugOutput(GraphTextureHandle output, uint32_t debugView)
{
    _debugOutput = output;
    _debugView = debugView;
}

void DeferredLightingPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void DeferredLightingPass::SetLight(glm::vec4 lightDirectionAndIntensity)
{
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
}

void DeferredLightingPass::SetLightColors(glm::vec4 directional, glm::vec4 point, glm::vec4 spot, glm::vec4 area)
{
    _directionalLightColor = glm::vec4(glm::max(glm::vec3(directional), glm::vec3(0.0f)), 0.0f);
    _pointLightColor = glm::vec4(glm::max(glm::vec3(point), glm::vec3(0.0f)), 0.0f);
    _spotLightColor = glm::vec4(glm::max(glm::vec3(spot), glm::vec3(0.0f)), 0.0f);
    _areaLightColor = glm::vec4(glm::max(glm::vec3(area), glm::vec3(0.0f)), 0.0f);
}

void DeferredLightingPass::SetPointLight(bool enabled, glm::vec4 positionAndIntensity)
{
    _pointLightPositionAndIntensity = glm::vec4(positionAndIntensity.x,
        positionAndIntensity.y,
        positionAndIntensity.z,
        enabled ? std::max(positionAndIntensity.w, 0.0f) : 0.0f);
}

void DeferredLightingPass::SetSpotLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 directionAndAngle)
{
    glm::vec3 direction(directionAndAngle);
    if (glm::dot(direction, direction) <= 1.0e-6f) {
        direction = glm::vec3(0.0f, -0.8f, -0.6f);
    }
    direction = glm::normalize(direction);
    _spotLightPositionAndIntensity = glm::vec4(positionAndIntensity.x,
        positionAndIntensity.y,
        positionAndIntensity.z,
        enabled ? std::max(positionAndIntensity.w, 0.0f) : 0.0f);
    _spotLightDirectionAndAngle = glm::vec4(direction, std::clamp(directionAndAngle.w, 5.0f, 80.0f));
}

void DeferredLightingPass::SetAreaLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 normalAndSize)
{
    glm::vec3 normal(normalAndSize);
    if (glm::dot(normal, normal) <= 1.0e-6f) {
        normal = glm::vec3(0.0f, -1.0f, 0.0f);
    }
    normal = glm::normalize(normal);
    _areaLightPositionAndIntensity = glm::vec4(positionAndIntensity.x,
        positionAndIntensity.y,
        positionAndIntensity.z,
        enabled ? std::max(positionAndIntensity.w, 0.0f) : 0.0f);
    _areaLightNormalAndSize = glm::vec4(normal, std::clamp(normalAndSize.w, 0.1f, 12.0f));
}

void DeferredLightingPass::SetEnvironment(glm::vec4 environmentParams)
{
    _environmentParams = environmentParams;
}

void DeferredLightingPass::SetEnvironmentImage(uint32_t sampledImageIndex)
{
    _environmentImageIndex = sampledImageIndex;
}

void DeferredLightingPass::SetIblDiffuseIrradianceImage(uint32_t sampledImageIndex)
{
    _iblDiffuseIrradianceImageIndex = sampledImageIndex;
}

void DeferredLightingPass::SetIblBrdfLutImage(uint32_t sampledImageIndex)
{
    _iblBrdfLutImageIndex = sampledImageIndex;
}

void DeferredLightingPass::SetIblSpecularPrefilterImage(uint32_t sampledImageIndex)
{
    _iblSpecularPrefilterImageIndex = sampledImageIndex;
}

void DeferredLightingPass::SetEnvironmentSpecularStrength(float strength)
{
    _environmentSpecularStrength = std::clamp(strength, 0.0f, 2.0f);
}

void DeferredLightingPass::SetRayEffects(
    GraphTextureHandle rayEffects, bool shadowsEnabled, bool ambientOcclusionEnabled, bool reflectionsEnabled)
{
    _rayEffects = rayEffects;
    _rayEffectsFlags = glm::uvec4(
        shadowsEnabled ? 1u : 0u,
        ambientOcclusionEnabled ? 1u : 0u,
        reflectionsEnabled ? 1u : 0u,
        0u);
}

void DeferredLightingPass::SetAmbientOcclusion(bool enabled, float radius, float intensity)
{
    _ssaoParams = glm::vec4(enabled ? 1.0f : 0.0f, std::max(radius, 0.01f), std::clamp(intensity, 0.0f, 4.0f), 0.0f);
}

void DeferredLightingPass::SetScreenSpaceReflections(bool enabled, float maxDistance, float thickness, float intensity)
{
    _ssrParams = glm::vec4(enabled ? 1.0f : 0.0f,
        std::clamp(maxDistance, 0.5f, 100.0f),
        std::clamp(thickness, 0.01f, 2.0f),
        std::clamp(intensity, 0.0f, 2.0f));
}

void DeferredLightingPass::SetScreenSpaceGlobalIllumination(bool enabled, float radius, float intensity, uint32_t sampleCount)
{
    _ssgiParams = glm::vec4(enabled ? 1.0f : 0.0f,
        std::clamp(radius, 0.05f, 8.0f),
        std::clamp(intensity, 0.0f, 2.0f),
        static_cast<float>(std::clamp(sampleCount, 4u, 16u)));
}

void DeferredLightingPass::SetContactShadows(bool enabled, float length, float intensity)
{
    _contactShadowParams = glm::vec4(enabled ? 1.0f : 0.0f,
        std::clamp(length, 0.05f, 8.0f),
        std::clamp(intensity, 0.0f, 1.0f),
        0.0f);
}

void DeferredLightingPass::SetShadowMap(
    GraphTextureHandle shadowMap,
    const std::array<DirectionalShadowCascade, 4>& cascades,
    uint32_t cascadeCount,
    float splitLambda,
    float bias,
    float normalBias,
    float strength,
    bool pcssEnabled,
    float filterRadius)
{
    _shadowMap = shadowMap;
    _shadowCascades = cascades;
    _shadowCascadeCount = std::clamp(cascadeCount, 1u, 4u);
    _shadowCascadeLambda = std::clamp(splitLambda, 0.0f, 1.0f);
    _shadowParams = glm::vec4(std::clamp(bias, 0.0f, 0.02f),
        std::clamp(normalBias, 0.0f, 0.2f),
        std::clamp(strength, 0.0f, 1.0f),
        shadowMap ? 1.0f : 0.0f);
    _shadowFilterParams = glm::vec4(std::clamp(filterRadius, 0.5f, 4.0f), pcssEnabled ? 1.0f : 0.0f, 0.0f, 0.0f);
}

void DeferredLightingPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/deferred_lighting.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(DeferredLightingPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);

    _lightingConstantsBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(DeferredLightingConstants),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .registerBindlessStorage = true,
        .debugName = "DeferredLightingConstants",
    });
}

void DeferredLightingPass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_albedo, ResourceUsage::StorageRead);
    builder.Read(_normal, ResourceUsage::StorageRead);
    builder.Read(_material, ResourceUsage::StorageRead);
    builder.Read(_depth, ResourceUsage::SampledRead);
    if (_shadowMap) {
        builder.Read(_shadowMap, ResourceUsage::SampledRead);
    }
    if (_rayEffects) {
        builder.Read(_rayEffects, ResourceUsage::StorageRead);
    }
    builder.Write(_output, ResourceUsage::StorageWrite);
    if (_debugOutput) {
        builder.Write(_debugOutput, ResourceUsage::StorageWrite);
    }
}

void DeferredLightingPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || _camera == nullptr) {
        return;
    }

    const ImageHandle albedoHandle = context.GetTextureHandle(_albedo);
    const ImageHandle normalHandle = context.GetTextureHandle(_normal);
    const ImageHandle materialHandle = context.GetTextureHandle(_material);
    const ImageHandle depthHandle = context.GetTextureHandle(_depth);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    const uint32_t shadowMapIndex = _shadowMap
        ? context.GetDevice().GetImageResource(context.GetTextureHandle(_shadowMap)).bindless.sampledImage
        : kInvalidResourceIndex;
    const uint32_t debugOutputImageIndex = _debugOutput
        ? context.GetDevice().GetImageResource(context.GetTextureHandle(_debugOutput)).bindless.storageImage
        : kInvalidResourceIndex;
    const uint32_t rayEffectsImageIndex = _rayEffects
        ? context.GetDevice().GetImageResource(context.GetTextureHandle(_rayEffects)).bindless.storageImage
        : kInvalidResourceIndex;

    const AllocatedBuffer& lightingConstantsBuffer = context.GetDevice().GetBufferResource(_lightingConstantsBuffer);
    if (lightingConstantsBuffer.allocationInfo.pMappedData != nullptr) {
        glm::mat4 cascadeMatrices[4]{};
        glm::vec4 cascadeSplits(0.0f);
        glm::vec4 cascadeAtlasScaleOffsets[4]{};
        for (uint32_t cascadeIndex = 0; cascadeIndex < 4u; ++cascadeIndex) {
            cascadeMatrices[cascadeIndex] = _shadowCascades[cascadeIndex].viewProjection;
            cascadeSplits[cascadeIndex] = _shadowCascades[cascadeIndex].splitDepth;
            cascadeAtlasScaleOffsets[cascadeIndex] = _shadowCascades[cascadeIndex].atlasScaleOffset;
        }
        const DeferredLightingConstants constants{
            .lightViewProjection = _shadowCascades[0].viewProjection,
            .cascadeViewProjections = {
                cascadeMatrices[0],
                cascadeMatrices[1],
                cascadeMatrices[2],
                cascadeMatrices[3],
            },
            .cascadeSplits = cascadeSplits,
            .cascadeAtlasScaleOffsets = {
                cascadeAtlasScaleOffsets[0],
                cascadeAtlasScaleOffsets[1],
                cascadeAtlasScaleOffsets[2],
                cascadeAtlasScaleOffsets[3],
            },
            .shadowParams = glm::vec4(_shadowParams.x, _shadowParams.y, _shadowParams.z, shadowMapIndex != kInvalidResourceIndex ? _shadowParams.w : 0.0f),
            .shadowFilterParams = _shadowFilterParams,
            .contactShadowParams = _contactShadowParams,
            .shadowIndices = glm::uvec4(shadowMapIndex, _environmentImageIndex, _shadowCascadeCount, _iblBrdfLutImageIndex),
            .iblIndices = glm::uvec4(_environmentImageIndex, _iblDiffuseIrradianceImageIndex, _iblBrdfLutImageIndex, _iblSpecularPrefilterImageIndex),
            .rayEffects = glm::uvec4(rayEffectsImageIndex, _rayEffectsFlags.x, _rayEffectsFlags.y, _rayEffectsFlags.z),
            .directionalColor = _directionalLightColor,
            .pointPositionAndIntensity = _pointLightPositionAndIntensity,
            .pointColor = _pointLightColor,
            .spotPositionAndIntensity = _spotLightPositionAndIntensity,
            .spotDirectionAndParams = glm::vec4(glm::vec3(_spotLightDirectionAndAngle), glm::radians(_spotLightDirectionAndAngle.w)),
            .spotColor = _spotLightColor,
            .areaPositionAndIntensity = _areaLightPositionAndIntensity,
            .areaNormalAndSize = _areaLightNormalAndSize,
            .areaColor = _areaLightColor,
        };
        std::memcpy(lightingConstantsBuffer.allocationInfo.pMappedData, &constants, sizeof(constants));
        context.GetDevice().FlushBuffer(_lightingConstantsBuffer, 0, sizeof(constants));
    }

    glm::vec4 cameraPosition = glm::vec4(_camera->GetPosition(), 0.0f);
    glm::vec4 environmentParams = _environmentParams;
    glm::vec4 ssaoParams = _ssaoParams;
    ssaoParams.w = _environmentSpecularStrength;

    DeferredLightingPushConstants pushConstants{
        .albedoImageIndex = context.GetDevice().GetImageResource(albedoHandle).bindless.storageImage,
        .normalImageIndex = context.GetDevice().GetImageResource(normalHandle).bindless.storageImage,
        .materialImageIndex = context.GetDevice().GetImageResource(materialHandle).bindless.storageImage,
        .depthImageIndex = context.GetDevice().GetImageResource(depthHandle).bindless.sampledImage,
        .outputImageIndex = context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
        .debugOutputImageIndex = debugOutputImageIndex,
        .debugView = _debugView,
        .lightingConstantsBufferIndex = lightingConstantsBuffer.bindless.storageBuffer,
        .inverseViewProjection = _camera->GetInverseViewProjection(),
        .viewProjection = _camera->GetViewProjection(),
        .cameraPosition = cameraPosition,
        .lightDirectionAndIntensity = _lightDirectionAndIntensity,
        .environmentParams = environmentParams,
        .ssaoParams = ssaoParams,
        .ssrParams = _ssrParams,
        .ssgiParams = _ssgiParams,
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);

    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
        _pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(DeferredLightingPushConstants),
        &pushConstants);

    // One thread group shades an 8x8 tile.
    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    vkCmdDispatch(commandBuffer, (outputExtent.width + 7) / 8, (outputExtent.height + 7) / 8, 1);
}

void DeferredLightingPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
    device.DestroyBuffer(_lightingConstantsBuffer);
}
} // namespace vesta::render

#include <vesta/render/passes/restir_di_resolve_pass.h>

#include <algorithm>
#include <array>
#include <cstring>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>

namespace vesta::render {
namespace {
constexpr uint32_t kResolveConstantVec4Count = 16u;

struct RestirDiResolvePushConstants {
    glm::uvec4 imageIndices{ kInvalidResourceIndex, kInvalidResourceIndex, kInvalidResourceIndex, kInvalidResourceIndex };
    glm::uvec4 outputIndices{ kInvalidResourceIndex, kInvalidResourceIndex, kInvalidResourceIndex, 0u };
    glm::uvec4 dispatchParams{ 1u, 1u, 1u, 1u };
    glm::uvec4 lightParams{ 1u, 1u, 0u, 0u };
    glm::uvec4 reuseParams{ 0u, 1u, 0u, 0u };
};

static_assert(sizeof(RestirDiResolvePushConstants) <= 128, "ReSTIR resolve push constants must stay compact.");

uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

glm::uvec4 PackVec4(glm::vec4 value)
{
    return glm::uvec4(FloatBits(value.x), FloatBits(value.y), FloatBits(value.z), FloatBits(value.w));
}

void ClearResolveOutput(const RenderGraphContext& context, GraphTextureHandle output)
{
    VkClearColorValue clearValue{};
    const VkImageSubresourceRange clearRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(context.GetCommandBuffer(),
        context.GetDevice().GetImage(context.GetTextureHandle(output)),
        VK_IMAGE_LAYOUT_GENERAL,
        &clearValue,
        1,
        &clearRange);
}
} // namespace

void RestirDiResolvePass::SetInputs(
    GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth)
{
    _albedo = albedo;
    _normal = normal;
    _material = material;
    _depth = depth;
}

void RestirDiResolvePass::SetOutput(GraphTextureHandle output)
{
    _output = output;
}

void RestirDiResolvePass::SetReservoirBuffer(BufferHandle reservoir)
{
    _reservoir = reservoir;
}

void RestirDiResolvePass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void RestirDiResolvePass::SetLight(glm::vec4 lightDirectionAndIntensity)
{
    _lightDirectionAndIntensity = lightDirectionAndIntensity;
}

void RestirDiResolvePass::SetLightColors(glm::vec4 directional, glm::vec4 point, glm::vec4 spot, glm::vec4 area)
{
    _directionalLightColor = glm::vec4(glm::max(glm::vec3(directional), glm::vec3(0.0f)), 0.0f);
    _pointLightColor = glm::vec4(glm::max(glm::vec3(point), glm::vec3(0.0f)), 0.0f);
    _spotLightColor = glm::vec4(glm::max(glm::vec3(spot), glm::vec3(0.0f)), 0.0f);
    _areaLightColor = glm::vec4(glm::max(glm::vec3(area), glm::vec3(0.0f)), 0.0f);
}

void RestirDiResolvePass::SetPointLight(bool enabled, glm::vec4 positionAndIntensity)
{
    _pointLightEnabled = enabled;
    _pointLightPositionAndIntensity = glm::vec4(positionAndIntensity.x,
        positionAndIntensity.y,
        positionAndIntensity.z,
        enabled ? std::max(positionAndIntensity.w, 0.0f) : 0.0f);
}

void RestirDiResolvePass::SetSpotLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 directionAndAngle)
{
    glm::vec3 direction(directionAndAngle);
    if (glm::dot(direction, direction) <= 1.0e-6f) {
        direction = glm::vec3(0.0f, -0.8f, -0.6f);
    }
    _spotLightEnabled = enabled;
    _spotLightPositionAndIntensity = glm::vec4(positionAndIntensity.x,
        positionAndIntensity.y,
        positionAndIntensity.z,
        enabled ? std::max(positionAndIntensity.w, 0.0f) : 0.0f);
    _spotLightDirectionAndAngle = glm::vec4(glm::normalize(direction), std::clamp(directionAndAngle.w, 5.0f, 80.0f));
}

void RestirDiResolvePass::SetAreaLight(bool enabled, glm::vec4 positionAndIntensity, glm::vec4 normalAndSize)
{
    glm::vec3 normal(normalAndSize);
    if (glm::dot(normal, normal) <= 1.0e-6f) {
        normal = glm::vec3(0.0f, -1.0f, 0.0f);
    }
    _areaLightEnabled = enabled;
    _areaLightPositionAndIntensity = glm::vec4(positionAndIntensity.x,
        positionAndIntensity.y,
        positionAndIntensity.z,
        enabled ? std::max(positionAndIntensity.w, 0.0f) : 0.0f);
    _areaLightNormalAndSize = glm::vec4(glm::normalize(normal), std::clamp(normalAndSize.w, 0.1f, 12.0f));
}

void RestirDiResolvePass::SetControls(uint32_t frameIndex,
    uint32_t reservoirCount,
    uint32_t candidateLightCount,
    uint32_t activeLightCount,
    uint32_t localLightCount,
    uint32_t emissiveTriangleCount,
    uint32_t spatialSamples,
    float intensity,
    bool spatialReuse,
    bool showReservoirs,
    bool showSelectedLight)
{
    _frameIndex = frameIndex;
    _reservoirCount = std::clamp(reservoirCount, 1u, 8u);
    _candidateLightCount = std::clamp(candidateLightCount, 1u, 64u);
    _activeLightCount = std::max(1u, activeLightCount);
    _localLightCount = std::min(std::max(1u, localLightCount), _activeLightCount);
    _emissiveTriangleCount = emissiveTriangleCount;
    _spatialSamples = std::clamp(spatialSamples, 0u, 16u);
    _intensity = std::clamp(intensity, 0.0f, 2.0f);
    _spatialReuse = spatialReuse;
    _showReservoirs = showReservoirs;
    _showSelectedLight = showSelectedLight;
}

void RestirDiResolvePass::Initialize(RenderDevice& device)
{
    _backendAvailable = false;
    if (device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }
    if (_pipeline != VK_NULL_HANDLE) {
        _backendAvailable = true;
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _computeShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/restir_di_resolve.comp.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{
        device.GetBindless().GetLayout(),
    };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(RestirDiResolvePushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    vkutil::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.computeShader = _computeShader;
    _pipeline = vkutil::create_compute_pipeline(vkDevice, pipelineDesc);
    _resolveConstantsBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(glm::uvec4) * kResolveConstantVec4Count,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .registerBindlessStorage = true,
        .debugName = "ReSTIR.ResolveConstants",
    });
    _backendAvailable = _pipeline != VK_NULL_HANDLE && _resolveConstantsBuffer;
}

void RestirDiResolvePass::Setup(RenderGraphBuilder& builder)
{
    builder.Read(_albedo, ResourceUsage::StorageRead);
    builder.Read(_normal, ResourceUsage::StorageRead);
    builder.Read(_material, ResourceUsage::StorageRead);
    builder.Read(_depth, ResourceUsage::SampledRead);
    builder.Write(_output, ResourceUsage::StorageWrite);
}

void RestirDiResolvePass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE || !_reservoir || !_resolveConstantsBuffer || _camera == nullptr) {
        ClearResolveOutput(context, _output);
        return;
    }

    const AllocatedBuffer& constantsBuffer = context.GetDevice().GetBufferResource(_resolveConstantsBuffer);
    if (constantsBuffer.allocationInfo.pMappedData != nullptr) {
        std::array<glm::uvec4, kResolveConstantVec4Count> constants{};
        const glm::mat4 inverseViewProjection = _camera->GetInverseViewProjection();
        constants[0] = PackVec4(inverseViewProjection[0]);
        constants[1] = PackVec4(inverseViewProjection[1]);
        constants[2] = PackVec4(inverseViewProjection[2]);
        constants[3] = PackVec4(inverseViewProjection[3]);
        constants[4] = PackVec4(glm::vec4(_camera->GetPosition(), 0.0f));
        constants[5] = PackVec4(_lightDirectionAndIntensity);
        constants[6] = PackVec4(_directionalLightColor);
        constants[7] = PackVec4(_pointLightPositionAndIntensity);
        constants[8] = PackVec4(_pointLightColor);
        constants[9] = PackVec4(_spotLightPositionAndIntensity);
        constants[10] = PackVec4(glm::vec4(glm::vec3(_spotLightDirectionAndAngle), glm::radians(_spotLightDirectionAndAngle.w)));
        constants[11] = PackVec4(_spotLightColor);
        constants[12] = PackVec4(_areaLightPositionAndIntensity);
        constants[13] = PackVec4(_areaLightNormalAndSize);
        constants[14] = PackVec4(_areaLightColor);
        constants[15] = PackVec4(glm::vec4(_intensity, _showReservoirs ? 1.0f : 0.0f, _showSelectedLight ? 1.0f : 0.0f, 0.0f));
        std::memcpy(constantsBuffer.allocationInfo.pMappedData, constants.data(), sizeof(constants));
        context.GetDevice().FlushBuffer(_resolveConstantsBuffer, 0, sizeof(constants));
    }

    const ImageHandle albedoHandle = context.GetTextureHandle(_albedo);
    const ImageHandle normalHandle = context.GetTextureHandle(_normal);
    const ImageHandle materialHandle = context.GetTextureHandle(_material);
    const ImageHandle depthHandle = context.GetTextureHandle(_depth);
    const ImageHandle outputHandle = context.GetTextureHandle(_output);
    const AllocatedBuffer& reservoirBuffer = context.GetDevice().GetBufferResource(_reservoir);
    const RestirDiResolvePushConstants pushConstants{
        .imageIndices = glm::uvec4(context.GetDevice().GetImageResource(albedoHandle).bindless.storageImage,
            context.GetDevice().GetImageResource(normalHandle).bindless.storageImage,
            context.GetDevice().GetImageResource(materialHandle).bindless.storageImage,
            context.GetDevice().GetImageResource(depthHandle).bindless.sampledImage),
        .outputIndices = glm::uvec4(context.GetDevice().GetImageResource(outputHandle).bindless.storageImage,
            reservoirBuffer.bindless.storageBuffer,
            constantsBuffer.bindless.storageBuffer,
            _frameIndex),
        .dispatchParams = glm::uvec4(context.GetTextureExtent(_output).width,
            context.GetTextureExtent(_output).height,
            _reservoirCount,
            _candidateLightCount),
        .lightParams = glm::uvec4(_activeLightCount,
            _localLightCount,
            _emissiveTriangleCount,
            (_pointLightEnabled ? 1u : 0u) | (_spotLightEnabled ? 2u : 0u) | (_areaLightEnabled ? 4u : 0u)),
        .reuseParams = glm::uvec4(_spatialSamples, _spatialReuse ? 1u : 0u, 0u, 0u),
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
        _pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(RestirDiResolvePushConstants),
        &pushConstants);

    const VkExtent3D outputExtent = context.GetTextureExtent(_output);
    vkCmdDispatch(commandBuffer, (outputExtent.width + 7u) / 8u, (outputExtent.height + 7u) / 8u, 1);
}

void RestirDiResolvePass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }
    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _computeShader);
    device.DestroyBuffer(_resolveConstantsBuffer);
    _backendAvailable = false;
}
} // namespace vesta::render

#include <vesta/render/passes/gaussian_splat_pass.h>

#include <array>
#include <algorithm>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
constexpr uint32_t kInvalidImageIndex = kInvalidResourceIndex;
constexpr float kPointCloudPointSize = 8.0f;
constexpr float kGaussianAlphaThreshold = 1.0e-4f;
constexpr float kGaussianRevealThreshold = 7.5e-4f;

struct GaussianGraphicsPushConstants {
    glm::mat4 viewMatrix{ 1.0f };
    glm::mat4 viewProjection{ 1.0f };
    glm::vec4 cameraPositionAndSceneType{ 0.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 params0{ kPointCloudPointSize, 1.0f, 0.0f, 0.0f };
    glm::vec4 params1{ kGaussianAlphaThreshold, kGaussianRevealThreshold, 0.0f, 0.0f };
    glm::uvec4 bufferIndices{ 0xFFFFFFFFu, 0u, 0u, 0u };
    glm::uvec4 options{ 0u, 0u, 0u, 0u };
};

struct GaussianComputePushConstants {
    glm::mat4 viewMatrix{ 1.0f };
    glm::mat4 viewProjection{ 1.0f };
    glm::vec4 cameraPositionAndSceneType{ 0.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 params0{ kPointCloudPointSize, 1.0f, 0.0f, 0.0f };
    glm::uvec4 params1{ 0u, 0u, 0u, 0u };
    glm::uvec4 params2{ kInvalidImageIndex, 0u, 0u, 0u };
    glm::vec4 params3{ kGaussianAlphaThreshold, kGaussianRevealThreshold, 0.0f, 0.0f };
};

struct ProjectedGaussianGPU {
    glm::vec4 centerDepthOpacity{ 0.0f };
    glm::vec4 conicOpacity{ 0.0f };
    glm::vec4 color{ 0.0f };
};

bool UseComputeGaussianPath(const vesta::scene::Scene* scene)
{
    return scene != nullptr && scene->HasTrainedGaussians();
}

void ClearStorageOutput(const RenderGraphContext& context, GraphTextureHandle accumOutput, GraphTextureHandle revealOutput)
{
    VkClearColorValue accumClear{};
    accumClear.float32[0] = 0.0f;
    accumClear.float32[1] = 0.0f;
    accumClear.float32[2] = 0.0f;
    accumClear.float32[3] = 0.0f;

    VkClearColorValue revealClear{};
    revealClear.float32[0] = 1.0f;
    revealClear.float32[1] = 1.0f;
    revealClear.float32[2] = 1.0f;
    revealClear.float32[3] = 1.0f;

    const VkImageSubresourceRange range = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkCmdClearColorImage(context.GetCommandBuffer(),
        context.GetDevice().GetImage(context.GetTextureHandle(accumOutput)),
        VK_IMAGE_LAYOUT_GENERAL,
        &accumClear,
        1,
        &range);
    vkCmdClearColorImage(context.GetCommandBuffer(),
        context.GetDevice().GetImage(context.GetTextureHandle(revealOutput)),
        VK_IMAGE_LAYOUT_GENERAL,
        &revealClear,
        1,
        &range);
}

void InsertBufferBarrier(VkCommandBuffer commandBuffer,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask)
{
    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;

    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}
} // namespace

void GaussianSplatPass::SetDepthInput(GraphTextureHandle depth)
{
    _depthInput = depth;
}

void GaussianSplatPass::SetOutputs(GraphTextureHandle accum, GraphTextureHandle reveal)
{
    _accumOutput = accum;
    _revealOutput = reveal;
}

void GaussianSplatPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void GaussianSplatPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void GaussianSplatPass::SetParams(
    float opacity, bool enabled, uint32_t shDegree, bool viewDependentColor, bool antialiasing, bool fastCulling)
{
    _opacity = opacity;
    _enabled = enabled;
    _shDegree = shDegree;
    _viewDependentColor = viewDependentColor;
    _antialiasing = antialiasing;
    _fastCulling = fastCulling;
}

void GaussianSplatPass::Initialize(RenderDevice& device)
{
    if (device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    if (_graphicsPipeline == VK_NULL_HANDLE) {
        _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/gaussian.vert.spv"));
        _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/gaussian.frag.spv"));

        const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
        const std::array<VkPushConstantRange, 1> pushConstants{
            VkPushConstantRange{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(GaussianGraphicsPushConstants),
            },
        };
        _graphicsPipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

        vkutil::GraphicsPipelineDesc pipelineDesc{};
        pipelineDesc.layout = _graphicsPipelineLayout;
        pipelineDesc.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT };
        pipelineDesc.depthFormat = VK_FORMAT_D32_SFLOAT;
        pipelineDesc.vertexShader = _vertexShader;
        pipelineDesc.fragmentShader = _fragmentShader;
        pipelineDesc.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        pipelineDesc.cullMode = VK_CULL_MODE_NONE;
        pipelineDesc.depthTestEnable = true;
        pipelineDesc.depthWriteEnable = false;
        pipelineDesc.colorBlendAttachments = {
            VkPipelineColorBlendAttachmentState{
                .blendEnable = VK_TRUE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                    | VK_COLOR_COMPONENT_A_BIT,
            },
            VkPipelineColorBlendAttachmentState{
                .blendEnable = VK_TRUE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                    | VK_COLOR_COMPONENT_A_BIT,
            },
        };
        _graphicsPipeline = vkutil::create_graphics_pipeline(vkDevice, pipelineDesc);
    }

    if (_binPipeline == VK_NULL_HANDLE) {
        _binShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/gaussian_bin.comp.spv"));
        _tileShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/gaussian_tile.comp.spv"));

        std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
            VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &_computeDescriptorPool));

        std::array<VkDescriptorSetLayoutBinding, 6> bindings{
            vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
            vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
            vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
            vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
            vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 4),
            vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 5),
        };
        std::array<VkDescriptorBindingFlags, 6> bindingFlags{
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO
        };
        bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();
        VkDescriptorSetLayoutCreateInfo layoutInfo =
            vkinit::descriptorset_layout_create_info(bindings.data(), static_cast<uint32_t>(bindings.size()));
        layoutInfo.pNext = &bindingFlagsInfo;
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        VK_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &_computeDescriptorSetLayout));

        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = _computeDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &_computeDescriptorSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(vkDevice, &allocInfo, &_computeDescriptorSet));

        const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{
            device.GetBindless().GetLayout(),
            _computeDescriptorSetLayout,
        };
        const std::array<VkPushConstantRange, 1> pushConstants{
            VkPushConstantRange{
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = sizeof(GaussianComputePushConstants),
            },
        };
        _computePipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

        _binPipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{
                                                                     .layout = _computePipelineLayout,
                                                                     .computeShader = _binShader,
                                                                 });
        _tilePipeline = vkutil::create_compute_pipeline(vkDevice, vkutil::ComputePipelineDesc{
                                                                      .layout = _computePipelineLayout,
                                                                      .computeShader = _tileShader,
                                                                  });
    }
}

void GaussianSplatPass::Setup(RenderGraphBuilder& builder)
{
    if (_depthInput) {
        builder.Read(_depthInput, ResourceUsage::DepthRead);
    }

    if (UseComputeGaussianPath(_scene)) {
        builder.Write(_accumOutput, ResourceUsage::StorageWrite);
        builder.Write(_revealOutput, ResourceUsage::StorageWrite);
    } else {
        builder.Write(_accumOutput, ResourceUsage::ColorAttachmentWrite);
        builder.Write(_revealOutput, ResourceUsage::ColorAttachmentWrite);
    }
}

void GaussianSplatPass::EnsureComputeResources(RenderDevice& device, VkExtent2D extent, uint32_t gaussianCount)
{
    const uint32_t tileCountX = std::max(1u, (extent.width + kGaussianTileSize - 1u) / kGaussianTileSize);
    const uint32_t tileCountY = std::max(1u, (extent.height + kGaussianTileSize - 1u) / kGaussianTileSize);
    const uint32_t tileCount = tileCountX * tileCountY;
    if (_projectedGaussianBuffer && _tileCountBuffer && _tileEntryBuffer && _cachedGaussianCount == gaussianCount
        && _cachedExtent.width == extent.width && _cachedExtent.height == extent.height && _cachedTileCount == tileCount) {
        return;
    }

    DestroyComputeResources(device);

    if (gaussianCount == 0) {
        _cachedGaussianCount = 0;
        _cachedExtent = extent;
        _cachedTileCount = tileCount;
        return;
    }

    _projectedGaussianBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(ProjectedGaussianGPU) * gaussianCount,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "GaussianProjected",
    });
    _tileCountBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(uint32_t) * tileCount,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "GaussianTileCounts",
    });
    _tileEntryBuffer = device.CreateBuffer(BufferDesc{
        .size = sizeof(uint32_t) * static_cast<VkDeviceSize>(tileCount) * kGaussianMaxEntriesPerTile,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .debugName = "GaussianTileEntries",
    });

    _cachedGaussianCount = gaussianCount;
    _cachedExtent = extent;
    _cachedTileCount = tileCount;
}

void GaussianSplatPass::DestroyComputeResources(RenderDevice& device)
{
    if (_projectedGaussianBuffer) {
        device.DestroyBuffer(_projectedGaussianBuffer);
        _projectedGaussianBuffer = {};
    }
    if (_tileCountBuffer) {
        device.DestroyBuffer(_tileCountBuffer);
        _tileCountBuffer = {};
    }
    if (_tileEntryBuffer) {
        device.DestroyBuffer(_tileEntryBuffer);
        _tileEntryBuffer = {};
    }
    _cachedGaussianCount = 0;
    _cachedExtent = {};
    _cachedTileCount = 0;
}

void GaussianSplatPass::ExecuteGraphicsPath(const RenderGraphContext& context)
{
    VkClearValue accumClear{};
    accumClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkClearValue revealClear{};
    revealClear.color = { { 1.0f, 1.0f, 1.0f, 1.0f } };

    std::array<VkRenderingAttachmentInfo, 2> colorAttachments{};
    colorAttachments[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachments[0].imageView = context.GetTextureView(_accumOutput);
    colorAttachments[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachments[0].clearValue = accumClear;
    colorAttachments[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachments[1].imageView = context.GetTextureView(_revealOutput);
    colorAttachments[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachments[1].clearValue = revealClear;

    VkRenderingAttachmentInfo depthAttachment{};
    if (_depthInput) {
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = context.GetTextureView(_depthInput);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
    renderingInfo.pColorAttachments = colorAttachments.data();
    renderingInfo.pDepthAttachment = _depthInput ? &depthAttachment : nullptr;

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(context.GetRenderExtent().width);
    viewport.height = static_cast<float>(context.GetRenderExtent().height);
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = context.GetRenderExtent();

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    if (_enabled && _scene != nullptr && _camera != nullptr && _scene->HasGaussianSplats()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphicsPipeline);
        const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphicsPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

        GaussianGraphicsPushConstants pushConstants{
            .viewMatrix = _camera->GetViewMatrix(),
            .viewProjection = _camera->GetViewProjection(),
            .cameraPositionAndSceneType = glm::vec4(
                _camera->GetPosition(),
                _scene->HasTrainedGaussians() ? 1.0f : 0.0f),
            .params0 = glm::vec4(kPointCloudPointSize,
                _opacity,
                static_cast<float>(context.GetRenderExtent().width),
                static_cast<float>(context.GetRenderExtent().height)),
            .params1 = glm::vec4(kGaussianAlphaThreshold, kGaussianRevealThreshold, 0.0f, 0.0f),
            .bufferIndices = glm::uvec4(
                context.GetDevice().GetBufferResource(_scene->GetGaussianBuffer()).bindless.storageBuffer,
                static_cast<uint32_t>(_scene->GetGaussianCount()),
                std::min(_shDegree, _scene->GetGaussianShDegree()),
                _viewDependentColor ? 1u : 0u),
            .options = glm::uvec4(_antialiasing ? 1u : 0u, _fastCulling ? 1u : 0u, 0u, 0u),
        };
        vkCmdPushConstants(commandBuffer,
            _graphicsPipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(GaussianGraphicsPushConstants),
            &pushConstants);
        vkCmdDraw(commandBuffer, 4, pushConstants.bufferIndices.y, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);
}

void GaussianSplatPass::ExecuteComputePath(const RenderGraphContext& context)
{
    if (_scene == nullptr || _camera == nullptr || !_scene->HasGaussianSplats()) {
        ClearStorageOutput(context, _accumOutput, _revealOutput);
        return;
    }

    RenderDevice& device = context.GetDevice();
    EnsureComputeResources(device, context.GetRenderExtent(), _scene->GetGaussianCount());
    if (!_projectedGaussianBuffer || !_tileCountBuffer || !_tileEntryBuffer) {
        ClearStorageOutput(context, _accumOutput, _revealOutput);
        return;
    }

    VkDescriptorBufferInfo gaussianInfo = vkinit::buffer_info(device.GetBuffer(_scene->GetGaussianBuffer()),
        0,
        VK_WHOLE_SIZE);
    VkDescriptorBufferInfo projectedInfo =
        vkinit::buffer_info(device.GetBuffer(_projectedGaussianBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo tileCountInfo = vkinit::buffer_info(device.GetBuffer(_tileCountBuffer), 0, VK_WHOLE_SIZE);
    VkDescriptorBufferInfo tileEntryInfo = vkinit::buffer_info(device.GetBuffer(_tileEntryBuffer), 0, VK_WHOLE_SIZE);

    VkDescriptorImageInfo accumInfo{};
    accumInfo.imageView = context.GetTextureView(_accumOutput);
    accumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo revealInfo{};
    revealInfo.imageView = context.GetTextureView(_revealOutput);
    revealInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 6> writes{
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _computeDescriptorSet, &gaussianInfo, 0),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _computeDescriptorSet, &projectedInfo, 1),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _computeDescriptorSet, &tileCountInfo, 2),
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _computeDescriptorSet, &tileEntryInfo, 3),
        vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _computeDescriptorSet, &accumInfo, 4),
        vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _computeDescriptorSet, &revealInfo, 5),
    };
    vkUpdateDescriptorSets(device.GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const uint32_t tileCountX = std::max(1u, (context.GetRenderExtent().width + kGaussianTileSize - 1u) / kGaussianTileSize);
    const uint32_t tileCountY = std::max(1u, (context.GetRenderExtent().height + kGaussianTileSize - 1u) / kGaussianTileSize);
    const uint32_t depthImageIndex =
        _depthInput ? device.GetImageResource(context.GetTextureHandle(_depthInput)).bindless.sampledImage : kInvalidImageIndex;

    GaussianComputePushConstants pushConstants{
        .viewMatrix = _camera->GetViewMatrix(),
        .viewProjection = _camera->GetViewProjection(),
        .cameraPositionAndSceneType = glm::vec4(_camera->GetPosition(), 1.0f),
        .params0 = glm::vec4(
            kPointCloudPointSize, _opacity, static_cast<float>(context.GetRenderExtent().width), static_cast<float>(context.GetRenderExtent().height)),
        .params1 = glm::uvec4(_scene->GetGaussianCount(), tileCountX, tileCountY, std::min(_shDegree, _scene->GetGaussianShDegree())),
        .params2 = glm::uvec4(depthImageIndex, _viewDependentColor ? 1u : 0u, _antialiasing ? 1u : 0u, _fastCulling ? 1u : 0u),
        .params3 = glm::vec4(kGaussianAlphaThreshold, kGaussianRevealThreshold, 0.0f, 0.0f),
    };

    VkCommandBuffer commandBuffer = context.GetCommandBuffer();
    vkCmdFillBuffer(commandBuffer, device.GetBuffer(_tileCountBuffer), 0, VK_WHOLE_SIZE, 0u);
    InsertBufferBarrier(commandBuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    const std::array<VkDescriptorSet, 2> descriptorSets{ device.GetBindless().GetSet(), _computeDescriptorSet };
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _binPipeline);
    vkCmdBindDescriptorSets(commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        _computePipelineLayout,
        0,
        static_cast<uint32_t>(descriptorSets.size()),
        descriptorSets.data(),
        0,
        nullptr);
    vkCmdPushConstants(commandBuffer,
        _computePipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(GaussianComputePushConstants),
        &pushConstants);
    vkCmdDispatch(commandBuffer, (pushConstants.params1.x + 127u) / 128u, 1, 1);

    InsertBufferBarrier(commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _tilePipeline);
    vkCmdBindDescriptorSets(commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        _computePipelineLayout,
        0,
        static_cast<uint32_t>(descriptorSets.size()),
        descriptorSets.data(),
        0,
        nullptr);
    vkCmdPushConstants(commandBuffer,
        _computePipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(GaussianComputePushConstants),
        &pushConstants);
    vkCmdDispatch(commandBuffer, tileCountX, tileCountY, 1);
}

void GaussianSplatPass::Execute(const RenderGraphContext& context)
{
    if (!_enabled || _scene == nullptr || _camera == nullptr || !_scene->HasGaussianSplats()) {
        if (UseComputeGaussianPath(_scene)) {
            ClearStorageOutput(context, _accumOutput, _revealOutput);
        } else {
            ExecuteGraphicsPath(context);
        }
        return;
    }

    if (UseComputeGaussianPath(_scene) && _binPipeline != VK_NULL_HANDLE && _tilePipeline != VK_NULL_HANDLE) {
        ExecuteComputePath(context);
        return;
    }

    if (_graphicsPipeline != VK_NULL_HANDLE) {
        ExecuteGraphicsPath(context);
    }
}

void GaussianSplatPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    DestroyComputeResources(device);

    if (_computeDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vkDevice, _computeDescriptorPool, nullptr);
        _computeDescriptorPool = VK_NULL_HANDLE;
    }
    if (_computeDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vkDevice, _computeDescriptorSetLayout, nullptr);
        _computeDescriptorSetLayout = VK_NULL_HANDLE;
    }
    _computeDescriptorSet = VK_NULL_HANDLE;

    vkutil::destroy_pipeline(vkDevice, _graphicsPipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _graphicsPipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _vertexShader);
    vkutil::destroy_shader_module(vkDevice, _fragmentShader);

    vkutil::destroy_pipeline(vkDevice, _binPipeline);
    vkutil::destroy_pipeline(vkDevice, _tilePipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _computePipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _binShader);
    vkutil::destroy_shader_module(vkDevice, _tileShader);
}
} // namespace vesta::render

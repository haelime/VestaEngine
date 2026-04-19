#include <vesta/render/passes/geometry_raster_pass.h>

#include <array>
#include <cstring>

#include <glm/glm.hpp>

#include <vesta/render/vulkan/vk_loader.h>
#include <vesta/render/vulkan/vk_pipelines.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

namespace vesta::render {
namespace {
// Geometry pass only needs a camera matrix because all mesh data already lives
// in world space after Scene::LoadFromFile flattens the node hierarchy.
struct GeometryPushConstants {
    glm::mat4 viewProjection{ 1.0f };
    uint32_t materialBufferIndex{ kInvalidResourceIndex };
};

constexpr VmaAllocationCreateFlags kMappedHostFlags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
} // namespace

void GeometryRasterPass::SetTargets(GraphTextureHandle albedo, GraphTextureHandle normal, GraphTextureHandle material, GraphTextureHandle depth)
{
    _albedoTarget = albedo;
    _normalTarget = normal;
    _materialTarget = material;
    _depthTarget = depth;
}

void GeometryRasterPass::SetScene(const vesta::scene::Scene* scene)
{
    _scene = scene;
}

void GeometryRasterPass::SetCamera(const Camera* camera)
{
    _camera = camera;
}

void GeometryRasterPass::SetVisibleSurfaceIndices(const std::vector<uint32_t>* visibleSurfaceIndices)
{
    _visibleSurfaceIndices = visibleSurfaceIndices;
}

void GeometryRasterPass::SetUseIndirectDraw(bool useIndirectDraw)
{
    _useIndirectDraw = useIndirectDraw;
}

void GeometryRasterPass::Initialize(RenderDevice& device)
{
    if (_pipeline != VK_NULL_HANDLE || device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice vkDevice = device.GetDevice();
    _vertexShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/geometry.vert.spv"));
    _fragmentShader = vkutil::load_shader_module(vkDevice, vkutil::resolve_runtime_path("shaders/geometry.frag.spv"));

    const std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts{ device.GetBindless().GetLayout() };
    const std::array<VkPushConstantRange, 1> pushConstants{
        VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(GeometryPushConstants),
        },
    };
    _pipelineLayout = vkutil::create_pipeline_layout(vkDevice, descriptorSetLayouts, pushConstants);

    // Vertex input matches SceneVertex exactly. The pass writes albedo and normal
    // into separate targets so later passes can choose how to consume them.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(vesta::scene::SceneVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 6> attributes{};
    attributes[0] = VkVertexInputAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, position),
    };
    attributes[1] = VkVertexInputAttributeDescription{
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, normal),
    };
    attributes[2] = VkVertexInputAttributeDescription{
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, tangent),
    };
    attributes[3] = VkVertexInputAttributeDescription{
        .location = 3,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, color),
    };
    attributes[4] = VkVertexInputAttributeDescription{
        .location = 4,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(vesta::scene::SceneVertex, texCoord),
    };
    attributes[5] = VkVertexInputAttributeDescription{
        .location = 5,
        .binding = 0,
        .format = VK_FORMAT_R32_UINT,
        .offset = offsetof(vesta::scene::SceneVertex, materialIndex),
    };

    vkutil::GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = _pipelineLayout;
    pipelineDesc.colorFormats = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
    };
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

void GeometryRasterPass::Setup(RenderGraphBuilder& builder)
{
    builder.Write(_albedoTarget, ResourceUsage::ColorAttachmentWrite);
    builder.Write(_normalTarget, ResourceUsage::ColorAttachmentWrite);
    builder.Write(_materialTarget, ResourceUsage::ColorAttachmentWrite);
    builder.Write(_depthTarget, ResourceUsage::DepthAttachmentWrite);
}

void GeometryRasterPass::Execute(const RenderGraphContext& context)
{
    if (_pipeline == VK_NULL_HANDLE) {
        return;
    }

    VkClearValue albedoClear{};
    albedoClear.color = { { 0.02f, 0.02f, 0.03f, 1.0f } };
    VkClearValue normalClear{};
    normalClear.color = { { 0.5f, 0.5f, 1.0f, 1.0f } };
    VkClearValue materialClear{};
    materialClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

    std::array<VkRenderingAttachmentInfo, 3> colorAttachments{};
    colorAttachments[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachments[0].imageView = context.GetTextureView(_albedoTarget);
    colorAttachments[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachments[0].clearValue = albedoClear;

    colorAttachments[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachments[1].imageView = context.GetTextureView(_normalTarget);
    colorAttachments[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachments[1].clearValue = normalClear;

    colorAttachments[2].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachments[2].imageView = context.GetTextureView(_materialTarget);
    colorAttachments[2].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachments[2].clearValue = materialClear;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = context.GetTextureView(_depthTarget);
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, context.GetRenderExtent() };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
    renderingInfo.pColorAttachments = colorAttachments.data();
    renderingInfo.pDepthAttachment = &depthAttachment;

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
    if (_scene != nullptr && _camera != nullptr && _scene->HasRasterGeometry()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);

        GeometryPushConstants pushConstants{
            .viewProjection = _camera->GetViewProjection(),
            .materialBufferIndex = context.GetDevice().GetBufferResource(_scene->GetMaterialBuffer()).bindless.storageBuffer,
        };
        vkCmdPushConstants(commandBuffer,
            _pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(GeometryPushConstants),
            &pushConstants);

        const VkBuffer vertexBuffer = context.GetDevice().GetBuffer(_scene->GetVertexBuffer());
        const VkBuffer indexBuffer = context.GetDevice().GetBuffer(_scene->GetIndexBuffer());
        constexpr VkDeviceSize vertexOffset = 0;
        const VkDescriptorSet bindlessSet = context.GetDevice().GetBindless().GetSet();
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Keeping surfaces separate leaves room for future material/state splits
        // and for CPU-side visibility culling to trim the draw list before it reaches Vulkan.
        if (_useIndirectDraw) {
            std::vector<VkDrawIndexedIndirectCommand> drawCommands;
            if (_visibleSurfaceIndices != nullptr) {
                drawCommands.reserve(_visibleSurfaceIndices->size());
                const auto& surfaces = _scene->GetSurfaces();
                for (uint32_t surfaceIndex : *_visibleSurfaceIndices) {
                    const vesta::scene::SceneSurface& surface = surfaces.at(surfaceIndex);
                    drawCommands.push_back(VkDrawIndexedIndirectCommand{
                        .indexCount = surface.indexCount,
                        .instanceCount = 1,
                        .firstIndex = surface.firstIndex,
                        .vertexOffset = 0,
                        .firstInstance = 0,
                    });
                }
            } else {
                drawCommands.reserve(_scene->GetSurfaces().size());
                for (const vesta::scene::SceneSurface& surface : _scene->GetSurfaces()) {
                    drawCommands.push_back(VkDrawIndexedIndirectCommand{
                        .indexCount = surface.indexCount,
                        .instanceCount = 1,
                        .firstIndex = surface.firstIndex,
                        .vertexOffset = 0,
                        .firstInstance = 0,
                    });
                }
            }

            if (!drawCommands.empty()) {
                const size_t requiredBytes = sizeof(VkDrawIndexedIndirectCommand) * drawCommands.size();
                if (!_indirectBuffer || _indirectBufferCapacity < requiredBytes) {
                    if (_indirectBuffer) {
                        context.GetDevice().DestroyBuffer(_indirectBuffer);
                    }
                    _indirectBuffer = context.GetDevice().CreateBuffer(BufferDesc{
                        .size = requiredBytes,
                        .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                        .allocationFlags = kMappedHostFlags,
                        .debugName = "GeometryIndirectCommands",
                    });
                    _indirectBufferCapacity = requiredBytes;
                }

                const AllocatedBuffer& indirectBuffer = context.GetDevice().GetBufferResource(_indirectBuffer);
                std::memcpy(indirectBuffer.allocationInfo.pMappedData, drawCommands.data(), requiredBytes);
                context.GetDevice().FlushBuffer(_indirectBuffer, 0, static_cast<VkDeviceSize>(requiredBytes));
                vkCmdDrawIndexedIndirect(commandBuffer,
                    context.GetDevice().GetBuffer(_indirectBuffer),
                    0,
                    static_cast<uint32_t>(drawCommands.size()),
                    sizeof(VkDrawIndexedIndirectCommand));
            }
        } else if (_visibleSurfaceIndices != nullptr) {
            const auto& surfaces = _scene->GetSurfaces();
            for (uint32_t surfaceIndex : *_visibleSurfaceIndices) {
                const vesta::scene::SceneSurface& surface = surfaces.at(surfaceIndex);
                vkCmdDrawIndexed(commandBuffer, surface.indexCount, 1, surface.firstIndex, 0, 0);
            }
        } else {
            for (const vesta::scene::SceneSurface& surface : _scene->GetSurfaces()) {
                vkCmdDrawIndexed(commandBuffer, surface.indexCount, 1, surface.firstIndex, 0, 0);
            }
        }
    }

    vkCmdEndRendering(commandBuffer);
}

void GeometryRasterPass::Shutdown(RenderDevice& device)
{
    VkDevice vkDevice = device.GetDevice();
    if (vkDevice == VK_NULL_HANDLE) {
        return;
    }

    vkutil::destroy_pipeline(vkDevice, _pipeline);
    vkutil::destroy_pipeline_layout(vkDevice, _pipelineLayout);
    vkutil::destroy_shader_module(vkDevice, _vertexShader);
    vkutil::destroy_shader_module(vkDevice, _fragmentShader);
    if (_indirectBuffer) {
        device.DestroyBuffer(_indirectBuffer);
        _indirectBuffer = {};
        _indirectBufferCapacity = 0;
    }
}
} // namespace vesta::render

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include <vesta/core/job_system.h>
#include <vesta/render/path_trace_backend.h>
#include <vesta/render/graph/render_graph.h>
#include <vesta/render/rhi/render_device.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

struct SDL_Window;
union SDL_Event;

namespace vesta::render {
inline constexpr uint32_t kMaxRenderGraphTimestampPasses = 32;

// High-level view modes exposed to the app and debug UI.
enum class RendererDisplayMode : uint32_t {
    Composite = 0,
    DeferredLighting = 1,
    Gaussian = 2,
    PathTrace = 3,
};

enum class RendererPreset : uint32_t {
    Recommended = 0,
    Performance = 1,
    Balanced = 2,
    Quality = 3,
};

enum class RendererDebugView : uint32_t {
    FinalColor = 0,
    Albedo = 1,
    Normal = 2,
    WorldPosition = 3,
    Depth = 4,
    UV = 5,
    MaterialId = 6,
    ObjectId = 7,
    Roughness = 8,
    Metallic = 9,
    Emissive = 10,
    AmbientOcclusion = 11,
    MotionVector = 12,
    DirectLighting = 13,
    IndirectLighting = 14,
    Reflection = 15,
    DenoisedResult = 16,
    DifferenceFromReference = 17,
    Wireframe = 18,
    MipLevel = 19,
    ShadowMap = 20,
    Overdraw = 21,
    TemporalHistoryColor = 22,
    TemporalHistoryDepth = 23,
    TemporalReprojection = 24,
    TemporalDisocclusion = 25,
    TemporalJitter = 26,
    ContactShadow = 27,
    ShadowCascade = 28,
    RayTracedGlobalIllumination = 29,
};

enum class GaussianDebugView : uint32_t {
    Final = 0,
    Alpha = 1,
    Revealage = 2,
    OverdrawHeatmap = 3,
    Depth = 4,
    TileOccupancy = 5,
    SplatRadius = 6,
    ContributionCount = 7,
    SplatId = 8,
    ShBand = 9,
    Covariance = 10,
    RasterDepth = 11,
    CompositionMask = 12,
    DepthDifference = 13,
};

enum class CompareMode : uint32_t {
    Off = 0,
    RasterPathSplit = 1,
    DifferenceHeatmap = 2,
};

enum class RasterPipelineMode : uint32_t {
    Forward = 0,
    Deferred = 1,
};

enum class ToneMappingMode : uint32_t {
    None = 0,
    Reinhard = 1,
    ACES = 2,
};

enum class SceneLoadState : uint32_t {
    Idle = 0,
    Parsing = 1,
    Preparing = 2,
    UploadingGeometry = 3,
    UploadingTextures = 4,
    BuildingBLAS = 5,
    BuildingTLAS = 6,
    ReadyToSwap = 7,
    Ready = 8,
    Failed = 9,
};

enum class SceneUploadMode : uint32_t {
    Synchronous = 0,
    AsyncParseSyncUpload = 1,
    Streaming = 2,
};

enum class SceneUploadContinuation : uint32_t {
    UploadTextures = 0,
    BuildBLAS = 1,
    ReadyToSwap = 2,
};

[[nodiscard]] constexpr bool IsValidSceneLoadTransition(SceneLoadState from, SceneLoadState to)
{
    if (from == to || to == SceneLoadState::Failed) {
        return true;
    }

    switch (from) {
    case SceneLoadState::Idle:
        return to == SceneLoadState::Parsing;
    case SceneLoadState::Parsing:
        return to == SceneLoadState::Preparing || to == SceneLoadState::UploadingGeometry;
    case SceneLoadState::Preparing:
        return to == SceneLoadState::UploadingGeometry;
    case SceneLoadState::UploadingGeometry:
        return to == SceneLoadState::UploadingTextures || to == SceneLoadState::BuildingBLAS
            || to == SceneLoadState::ReadyToSwap;
    case SceneLoadState::UploadingTextures:
        return to == SceneLoadState::BuildingBLAS || to == SceneLoadState::ReadyToSwap;
    case SceneLoadState::BuildingBLAS:
        return to == SceneLoadState::BuildingTLAS;
    case SceneLoadState::BuildingTLAS:
        return to == SceneLoadState::ReadyToSwap;
    case SceneLoadState::ReadyToSwap:
        return to == SceneLoadState::Ready;
    case SceneLoadState::Ready:
    case SceneLoadState::Failed:
    default:
        return to == SceneLoadState::Parsing;
    }
}

[[nodiscard]] constexpr SceneUploadContinuation DecideSceneUploadContinuation(
    bool textureStreamingEnabled, bool hasTextures, bool buildRayTracingStructuresOnLoad, bool rayTracingSupported, bool hasIndices)
{
    if (textureStreamingEnabled && hasTextures) {
        return SceneUploadContinuation::UploadTextures;
    }
    if (buildRayTracingStructuresOnLoad && rayTracingSupported && hasIndices) {
        return SceneUploadContinuation::BuildBLAS;
    }
    return SceneUploadContinuation::ReadyToSwap;
}

struct SceneLoadStatus {
    SceneLoadState state{ SceneLoadState::Idle };
    std::filesystem::path path;
    std::string message;
    std::string uploadStage;
    std::string lastBlockingWait;
    float parseMs{ 0.0f };
    float prepareMs{ 0.0f };
    float geometryUploadMs{ 0.0f };
    float textureUploadMs{ 0.0f };
    float blasMs{ 0.0f };
    float tlasMs{ 0.0f };
    uint64_t pendingUploadBytes{ 0 };
    uint32_t pendingUploadCopies{ 0 };
};

struct SceneUploadOptions {
    bool useDeviceLocalSceneBuffers{ true };
    bool buildRayTracingStructuresOnLoad{ true };
    bool textureStreamingEnabled{ true };
    bool useDeviceLocalTextures{ true };
};

struct MeshletClusterStats {
    uint32_t trianglesPerMeshlet{ 64 };
    uint32_t totalClusters{ 0 };
    uint32_t visibleClusters{ 0 };
    uint32_t culledClusters{ 0 };
    uint32_t totalMeshlets{ 0 };
    uint32_t visibleMeshlets{ 0 };
    uint32_t culledMeshlets{ 0 };
    uint32_t boundsAvailable{ 0 };
    bool visibilitySetValid{ false };
    bool coneCullingEnabled{ false };
    bool gpuDrivenBackend{ false };
};

struct GpuDrivenStats {
    uint32_t totalSurfaces{ 0 };
    uint32_t visibleSurfaces{ 0 };
    uint32_t culledSurfaces{ 0 };
    uint32_t indirectDrawEstimate{ 0 };
    bool visibilitySetValid{ false };
    bool indirectDrawEnabled{ false };
    bool gpuDrivenBackend{ false };
};

struct TemporalUpscalerStats {
    uint32_t inputWidth{ 0 };
    uint32_t inputHeight{ 0 };
    uint32_t outputWidth{ 0 };
    uint32_t outputHeight{ 0 };
    float scale{ 1.0f };
    float sharpness{ 0.0f };
    bool requested{ false };
    bool backendAvailable{ false };
    bool taaHistoryAvailable{ false };
    bool motionVectorsAvailable{ true };
    bool depthAvailable{ true };
    bool reactiveMaskAvailable{ false };
    bool materialReactiveMaskAvailable{ false };
    bool authoredAlphaReactiveMaskAvailable{ false };
    float reactiveMaskStrength{ 0.0f };
};

struct RestirStats {
    uint32_t activeLightCount{ 0 };
    uint32_t emissiveTriangleCount{ 0 };
    uint32_t localLightCount{ 0 };
    uint32_t candidateLightCount{ 0 };
    uint32_t reservoirCount{ 0 };
    uint64_t reservoirPixels{ 0 };
    uint64_t estimatedDiReservoirBytes{ 0 };
    uint64_t estimatedGiReservoirBytes{ 0 };
    uint64_t estimatedPtReservoirBytes{ 0 };
    uint64_t estimatedReservoirBytes{ 0 };
    bool requestedDi{ false };
    bool requestedGi{ false };
    bool requestedPt{ false };
    bool backendAvailable{ false };
    bool reservoirBuffersAvailable{ false };
    bool diReservoirBuffersAvailable{ false };
    bool giReservoirBuffersAvailable{ false };
    bool ptReservoirBuffersAvailable{ false };
    bool candidateSamplingAvailable{ false };
    bool temporalReusePassAvailable{ false };
    bool spatialReusePassAvailable{ false };
    bool lightingResolveAvailable{ false };
    bool giResolveAvailable{ false };
    bool ptResolveAvailable{ false };
    bool giCandidatePassAvailable{ false };
    bool ptCandidatePassAvailable{ false };
    bool giReservoirBackendAvailable{ false };
    bool ptReservoirBackendAvailable{ false };
    bool temporalReuse{ true };
    bool spatialReuse{ true };
    bool historyAvailable{ false };
};

struct DdgiStats {
    uint32_t probeCountX{ 0 };
    uint32_t probeCountY{ 0 };
    uint32_t probeCountZ{ 0 };
    uint32_t totalProbeCount{ 0 };
    uint32_t raysPerProbe{ 0 };
    uint64_t raysPerUpdate{ 0 };
    uint64_t estimatedIrradianceBytes{ 0 };
    uint64_t estimatedVisibilityBytes{ 0 };
    float probeSpacing{ 0.0f };
    float hysteresis{ 0.0f };
    float intensity{ 0.0f };
    bool requested{ false };
    bool backendAvailable{ false };
    bool probeStorageAvailable{ false };
    bool probeCompositeAvailable{ false };
    bool storageCompositeAvailable{ false };
    bool momentValidationAvailable{ false };
    bool spatialFilteringAvailable{ false };
    bool rayUpdateAvailable{ false };
    bool temporalBlendAvailable{ false };
    bool overlayEnabled{ false };
};

struct IblStats {
    uint32_t sourceWidth{ 0 };
    uint32_t sourceHeight{ 0 };
    uint32_t sourceChannels{ 0 };
    uint32_t diffuseCubemapResolution{ 32 };
    uint32_t specularCubemapResolution{ 128 };
    uint32_t specularMipCount{ 8 };
    uint32_t brdfLutResolution{ 256 };
    uint32_t environmentCubemapResolution{ 128 };
    uint64_t estimatedEnvironmentCubemapBytes{ 0 };
    uint64_t estimatedDiffuseBytes{ 0 };
    uint64_t estimatedSpecularBytes{ 0 };
    uint64_t estimatedBrdfLutBytes{ 0 };
    bool requested{ true };
    bool externalSourceAvailable{ false };
    bool environmentMapUploaded{ false };
    bool environmentCubemapAvailable{ false };
    bool sourceIsHdr{ false };
    bool proceduralSource{ true };
    bool diffuseBackendAvailable{ true };
    bool specularBackendAvailable{ false };
    bool diffuseIrradianceAvailable{ false };
    bool specularPrefilterAvailable{ false };
    bool brdfLutAvailable{ false };
};

struct RayEffectsStats {
    uint32_t inputWidth{ 0 };
    uint32_t inputHeight{ 0 };
    uint32_t shadowSamples{ 0 };
    uint32_t aoSamples{ 0 };
    uint32_t reflectionSamples{ 0 };
    uint32_t giSamples{ 0 };
    uint64_t estimatedShadowRays{ 0 };
    uint64_t estimatedAoRays{ 0 };
    uint64_t estimatedReflectionRays{ 0 };
    uint64_t estimatedGiRays{ 0 };
    bool shadowsRequested{ false };
    bool aoRequested{ false };
    bool reflectionsRequested{ false };
    bool giRequested{ false };
    bool rayQueryAvailable{ false };
    bool rtPipelineAvailable{ false };
    bool tlasAvailable{ false };
    bool backendAvailable{ false };
    bool halfResolution{ true };
    bool denoiserRequested{ true };
    bool giSpatialDenoiseAvailable{ false };
    bool temporalAccumulation{ true };
};

// RendererSettings collects the knobs that can safely change at runtime.
// When one of these changes in a way that affects history, accumulation resets.
struct RendererSettings {
    RendererDisplayMode displayMode{ RendererDisplayMode::DeferredLighting };
    bool enableRaster{ true };
    bool enableGaussian{ true };
    bool enablePathTracing{ true };
    bool optimizeInactivePasses{ true };
    bool preferAsyncSceneLoading{ true };
    bool useDeviceLocalSceneBuffers{ true };
    bool buildRayTracingStructuresOnLoad{ true };
    bool textureStreamingEnabled{ true };
    bool useDeviceLocalTextures{ true };
    bool deferOldSceneDestruction{ true };
    bool autoFocusSceneOnLoad{ false };
    bool frameTimingCapture{ false };
    bool benchmarkOverlay{ false };
    bool enableVSync{ true };
    bool enableFpsLimit{ false };
    uint32_t fpsLimit{ 60 };
    bool enableFrustumCulling{ true };
    bool enableDistanceCulling{ true };
    bool useIndirectDraw{ false };
    SceneUploadMode sceneUploadMode{ SceneUploadMode::AsyncParseSyncUpload };
    uint32_t maxUploadBytesPerFrame{ 4u * 1024u * 1024u };
    uint32_t maxTextureUploadBytesPerFrame{ 8u * 1024u * 1024u };
    float distanceCullScale{ 6.0f };
    float gaussianOpacity{ 1.0f };
    float gaussianMix{ 0.28f };
    uint32_t gaussianShDegree{ 0 };
    bool gaussianViewDependentColor{ true };
    bool gaussianAntialiasing{ true };
    bool gaussianFastCulling{ true };
    float pathTraceResolutionScale{ 0.5f };
    uint32_t pathTraceSamplesPerPixel{ 1 };
    uint32_t pathTraceMaxBounces{ 4 };
    bool pathTraceNextEventEstimation{ true };
    bool pathTraceRussianRoulette{ true };
    uint32_t pathTraceRussianRouletteDepth{ 3 };
    float pathTraceFireflyClamp{ 8.0f };
    PathTraceDebugView pathTraceDebugView{ PathTraceDebugView::Final };
    RendererDebugView debugView{ RendererDebugView::FinalColor };
    GaussianDebugView gaussianDebugView{ GaussianDebugView::Final };
    CompareMode compareMode{ CompareMode::Off };
    float compareSplitPosition{ 0.5f };
    float compareDifferenceScale{ 4.0f };
    RasterPipelineMode rasterPipelineMode{ RasterPipelineMode::Deferred };
    bool showGBufferPreview{ false };
    bool showShadowCascadeOverlay{ false };
    uint32_t shadowCascadeCount{ 4 };
    float shadowCascadeLambda{ 0.65f };
    bool showGiProbeOverlay{ false };
    bool gaussianShowTileGrid{ false };
    bool gaussianShowCovarianceEllipsoids{ false };
    bool gaussianShowSpatialBounds{ false };
    bool hybridDepthCompositeDebug{ false };
    bool enableRtShadows{ false };
    bool enableRtAmbientOcclusion{ false };
    bool enableRtReflections{ false };
    bool enableRtGlobalIllumination{ false };
    uint32_t rtShadowSamples{ 1 };
    uint32_t rtAoSamples{ 1 };
    uint32_t rtReflectionSamples{ 1 };
    uint32_t rtGiSamples{ 1 };
    float rtMaxRayDistance{ 100.0f };
    float rtAoRadius{ 2.0f };
    float rtReflectionRoughnessCutoff{ 0.8f };
    bool rtHalfResolution{ true };
    bool rtDenoiser{ true };
    bool rtTemporalAccumulation{ true };
    bool enablePathTracedGi{ true };
    bool enableDdgi{ false };
    bool enableVoxelGi{ false };
    bool enableRestirGi{ false };
    uint32_t ddgiProbeCountX{ 8 };
    uint32_t ddgiProbeCountY{ 4 };
    uint32_t ddgiProbeCountZ{ 8 };
    float ddgiProbeSpacing{ 2.0f };
    float ddgiHysteresis{ 0.95f };
    float ddgiIntensity{ 0.28f };
    uint32_t ddgiRaysPerProbe{ 128 };
    bool showGiIndirectOnly{ false };
    ToneMappingMode toneMappingMode{ ToneMappingMode::ACES };
    bool enableBloom{ true };
    bool enableColorGrading{ false };
    bool enableVignette{ false };
    bool enableMotionBlur{ false };
    bool enableFxaa{ true };
    float motionBlurStrength{ 0.35f };
    float bloomThreshold{ 1.0f };
    float bloomIntensity{ 0.1f };
    float vignetteStrength{ 0.0f };
    float colorGradingSaturation{ 1.0f };
    float colorGradingContrast{ 1.0f };
    bool enableRestirDi{ false };
    bool enableRestirPt{ false };
    uint32_t restirCandidateLights{ 8 };
    uint32_t restirReservoirCount{ 1 };
    uint32_t restirSpatialSamples{ 4 };
    float restirDirectLightingIntensity{ 0.18f };
    bool restirTemporalReuse{ true };
    bool restirSpatialReuse{ true };
    bool restirShowReservoirs{ false };
    bool restirShowSelectedLight{ false };
    bool enableGpuDrivenRendering{ false };
    bool enableAsyncCompute{ false };
    bool enableMeshletCulling{ false };
    bool showAsyncComputeTimeline{ false };
    bool enableTemporalUpscaler{ false };
    bool showTemporalUpscalerDebug{ false };
    float temporalUpscalerScale{ 0.67f };
    float temporalUpscalerSharpness{ 0.25f };
    bool temporalMaterialReactiveMask{ true };
    float temporalReactiveMaskStrength{ 0.65f };
    bool enablePathTraceDenoiser{ true };
    float pathTraceDenoiserStrength{ 0.65f };
    float pathTraceDenoiserTemporalBlend{ 0.88f };
    uint32_t pathTraceDenoiserIterations{ 3 };
    bool enableSsao{ true };
    float ssaoRadius{ 0.75f };
    float ssaoIntensity{ 1.35f };
    bool enableTaa{ true };
    float taaFeedback{ 0.88f };
    bool enableSsr{ true };
    float ssrMaxDistance{ 18.0f };
    float ssrThickness{ 0.18f };
    float ssrIntensity{ 0.65f };
    bool enableSsgi{ true };
    float ssgiRadius{ 1.4f };
    float ssgiIntensity{ 0.32f };
    uint32_t ssgiSampleCount{ 10 };
    bool enableShadowMap{ true };
    uint32_t shadowMapSize{ 2048 };
    float shadowBias{ 0.0015f };
    float shadowNormalBias{ 0.015f };
    float shadowStrength{ 0.82f };
    bool enablePcssShadows{ false };
    float shadowFilterRadius{ 1.0f };
    bool enableContactShadows{ true };
    float contactShadowLength{ 1.2f };
    float contactShadowIntensity{ 0.35f };
    bool animationPlaying{ false };
    float animationTimeScale{ 1.0f };
    float animationTimeSeconds{ 0.0f };
    bool animateDirectionalLight{ false };
    bool animateEnvironment{ false };
    glm::vec4 lightDirectionAndIntensity{ -0.4f, -1.0f, -0.3f, 2.0f };
    glm::vec4 directionalLightColor{ 1.0f, 1.0f, 1.0f, 0.0f };
    bool enablePointLight{ false };
    glm::vec4 pointLightPositionAndIntensity{ 0.0f, 2.0f, 0.0f, 4.0f };
    glm::vec4 pointLightColor{ 1.0f, 0.82f, 0.55f, 0.0f };
    bool enableSpotLight{ false };
    glm::vec4 spotLightPositionAndIntensity{ 0.0f, 3.0f, 2.5f, 10.0f };
    glm::vec4 spotLightDirectionAndAngle{ 0.0f, -0.8f, -0.6f, 28.0f };
    glm::vec4 spotLightColor{ 1.0f, 0.88f, 0.68f, 0.0f };
    bool enableAreaLight{ false };
    glm::vec4 areaLightPositionAndIntensity{ 0.0f, 3.2f, 0.0f, 5.0f };
    glm::vec4 areaLightNormalAndSize{ 0.0f, -1.0f, 0.0f, 2.0f };
    glm::vec4 areaLightColor{ 0.86f, 0.92f, 1.0f, 0.0f };
    float environmentIntensity{ 1.0f };
    float environmentRotationDegrees{ 0.0f };
    uint32_t environmentPreset{ 0 };
    std::filesystem::path externalHdriPath{};
    bool externalHdriAvailable{ false };
    bool externalHdriIsHdr{ false };
    uint32_t externalHdriWidth{ 0 };
    uint32_t externalHdriHeight{ 0 };
    uint32_t externalHdriChannels{ 0 };
    std::string externalHdriStatus{ "Procedural IBL" };
    float environmentDiffuseStrength{ 0.22f };
    float environmentSpecularStrength{ 0.45f };
    float cameraExposureEv{ 0.0f };
    float cameraApertureRadius{ 0.0f };
    float cameraFocalDistance{ 5.0f };
    PathTraceBackend pathTraceBackend{ PathTraceBackend::Auto };
};

// Each overlapping frame owns its own command buffer and sync objects so the CPU
// can prepare the next frame while the GPU is still finishing the previous one.
struct RendererFrameContext {
    VkCommandPool commandPool{ VK_NULL_HANDLE };
    VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
    VkSemaphore acquireSemaphore{ VK_NULL_HANDLE };
    VkFence renderFence{ VK_NULL_HANDLE };
    VkQueryPool renderGraphTimestampPool{ VK_NULL_HANDLE };
    bool renderGraphTimestampPending{ false };
    uint32_t renderGraphTimestampPassCount{ 0 };
    std::vector<std::string> renderGraphTimestampPassNames;
    std::vector<ImageHandle> acquiredTransientImages;
    std::vector<BufferHandle> transientBuffers;
    BufferHandle screenshotReadbackBuffer{};
    std::filesystem::path screenshotPath;
    VkExtent2D screenshotExtent{};
    VkFormat screenshotFormat{ VK_FORMAT_UNDEFINED };
    bool screenshotPending{ false };
};

// These are the logical edges between passes in the frame graph.
struct RendererGraphResources {
    GraphTextureHandle swapchainTarget{};
    GraphTextureHandle gbufferAlbedo{};
    GraphTextureHandle gbufferNormal{};
    GraphTextureHandle gbufferMaterial{};
    GraphTextureHandle gbufferDebug{};
    GraphTextureHandle gbufferMotion{};
    GraphTextureHandle gbufferReactive{};
    GraphTextureHandle sceneDepth{};
    GraphTextureHandle shadowMap{};
    GraphTextureHandle overdraw{};
    GraphTextureHandle rayEffects{};
    GraphTextureHandle rayReflection{};
    GraphTextureHandle rayGlobalIllumination{};
    GraphTextureHandle restirDirectLighting{};
    GraphTextureHandle deferredLighting{};
    GraphTextureHandle deferredLightingDebug{};
    GraphTextureHandle temporalLighting{};
    GraphTextureHandle pathTraceOutput{};
    GraphTextureHandle pathTraceNormalGuide{};
    GraphTextureHandle pathTraceDepthGuide{};
    GraphTextureHandle pathTraceDenoised{};
    GraphTextureHandle bloomHalf{};
    GraphTextureHandle bloomQuarter{};
    GraphTextureHandle bloomOutput{};
    GraphTextureHandle gaussianAccum{};
    GraphTextureHandle gaussianReveal{};
    GraphTextureHandle gaussianDebug{};
};

struct VisibleSet {
    std::shared_ptr<const vesta::scene::PreparedScene> scene;
    std::vector<uint32_t> surfaceIndices;
};

struct FrameSnapshot {
    VisibleSet visibleSet;
};

enum class SelectionKind : uint32_t {
    None = 0,
    Object = 1,
    DirectionalLight = 2,
    PointLight = 3,
    SpotLight = 4,
    AreaLight = 5,
};

struct EditorSelection {
    SelectionKind kind{ SelectionKind::None };
    uint32_t objectIndex{ 0 };
};

using RenderPassConfigureFn = std::function<void(IRenderPass&, const RendererGraphResources&)>;
using OverlayDrawFn = std::function<void(VkCommandBuffer)>;
using OverlaySwapchainCallback = std::function<void(uint32_t)>;

struct RenderPassRegistrationDesc {
    std::string id;
    std::unique_ptr<IRenderPass> pass;
    RenderPassConfigureFn configure;
    uint32_t order{ 0 };
    bool enabled{ true };
};

struct RenderPassDebugInfo {
    std::string id;
    std::string name;
    uint32_t order{ 0 };
    bool enabled{ false };
    uint32_t drawCount{ 0 };
    uint32_t dispatchCount{ 0 };
    uint64_t triangleCount{ 0 };
    uint64_t instanceCount{ 0 };
    uint64_t rayCount{ 0 };
    uint64_t primaryRayCount{ 0 };
    uint64_t shadowRayCount{ 0 };
    uint64_t diffuseRayCount{ 0 };
    uint64_t specularRayCount{ 0 };
    uint64_t splatCount{ 0 };
};

struct TransientImageKey {
    VkExtent3D extent{ 1, 1, 1 };
    VkFormat format{ VK_FORMAT_UNDEFINED };
    VkImageUsageFlags usage{ 0 };
    VkImageAspectFlags aspectFlags{ 0 };
    VkImageLayout initialLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
    VmaMemoryUsage memoryUsage{ VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    uint32_t mipLevels{ 1 };
    uint32_t arrayLayers{ 1 };

    [[nodiscard]] bool operator==(const TransientImageKey& other) const
    {
        return extent.width == other.extent.width && extent.height == other.extent.height && extent.depth == other.extent.depth
            && format == other.format && usage == other.usage && aspectFlags == other.aspectFlags
            && initialLayout == other.initialLayout && memoryUsage == other.memoryUsage && mipLevels == other.mipLevels
            && arrayLayers == other.arrayLayers;
    }
};

struct TransientImagePoolEntry {
    ImageHandle handle{};
    TransientImageKey key{};
    bool inUse{ false };
};

class TransientImagePool {
public:
    // Reuses images with identical descriptions to avoid creating temporary
    // Vulkan images every frame.
    [[nodiscard]] ImageHandle Acquire(RenderDevice& device, const ImageDesc& desc);
    void Release(ImageHandle handle);
    void Purge(RenderDevice& device);

private:
    [[nodiscard]] static TransientImageKey MakeKey(const ImageDesc& desc);

    std::vector<TransientImagePoolEntry> _entries;
};

class Renderer {
public:
    static constexpr uint32_t kFrameOverlap = 2;

    // Renderer drives the per-frame flow:
    // input -> camera update -> graph build -> pass execution -> presentation.
    bool Initialize(SDL_Window* window, VkExtent2D initialExtent, bool enableValidation);
    void Shutdown();
    void HandleEvent(const SDL_Event& event);
    void Update(float deltaSeconds);
    void RenderFrame();

    [[nodiscard]] const vesta::scene::Scene& GetScene() const { return _scene; }
    [[nodiscard]] const Camera& GetCamera() const { return _camera; }
    [[nodiscard]] Camera& GetCamera() { return _camera; }
    [[nodiscard]] vesta::core::JobSystem& GetJobSystem() { return _jobs; }
    [[nodiscard]] const RendererSettings& GetSettings() const { return _settings; }
    [[nodiscard]] RendererSettings& GetSettings() { return _settings; }
    [[nodiscard]] const SceneLoadStatus& GetSceneLoadStatus() const { return _sceneLoadStatus; }
    [[nodiscard]] uint32_t GetPathTraceFrameIndex() const { return _pathTraceFrameIndex; }
    [[nodiscard]] uint32_t GetFrameSlot() const { return static_cast<uint32_t>(_frameNumber % kFrameOverlap); }
    [[nodiscard]] float GetFrameTimeMs() const { return _frameTimeMs; }
    [[nodiscard]] float GetSmoothedFrameTimeMs() const { return _smoothedFrameTimeMs; }
    [[nodiscard]] const std::array<float, 240>& GetFrameTimeHistoryMs() const { return _frameTimeHistoryMs; }
    [[nodiscard]] size_t GetFrameTimeHistoryCount() const { return _frameTimeHistoryCount; }
    [[nodiscard]] const std::vector<RenderGraphPassTiming>& GetLastRenderGraphTimings() const { return _lastRenderGraphTimings; }
    [[nodiscard]] uint32_t GetVisibleSurfaceCount() const { return static_cast<uint32_t>(_visibleSurfaceIndices.size()); }
    [[nodiscard]] const std::vector<uint32_t>& GetVisibleSurfaceIndices() const { return _visibleSurfaceIndices; }
    [[nodiscard]] GpuDrivenStats GetGpuDrivenStats() const;
    [[nodiscard]] MeshletClusterStats GetMeshletClusterStats() const;
    [[nodiscard]] TemporalUpscalerStats GetTemporalUpscalerStats() const;
    [[nodiscard]] RestirStats GetRestirStats() const;
    [[nodiscard]] DdgiStats GetDdgiStats() const;
    [[nodiscard]] BufferHandle GetDdgiIrradianceBuffer() const { return _ddgiIrradianceBuffer; }
    [[nodiscard]] BufferHandle GetDdgiVisibilityBuffer() const { return _ddgiVisibilityBuffer; }
    [[nodiscard]] BufferHandle GetRestirReservoirBuffer() const { return _restirReservoirBuffer; }
    [[nodiscard]] BufferHandle GetRestirHistoryReservoirBuffer() const { return _restirHistoryReservoirBuffer; }
    [[nodiscard]] BufferHandle GetRestirGiReservoirBuffer() const { return _restirGiReservoirBuffer; }
    [[nodiscard]] BufferHandle GetRestirGiHistoryReservoirBuffer() const { return _restirGiHistoryReservoirBuffer; }
    [[nodiscard]] BufferHandle GetRestirPtReservoirBuffer() const { return _restirPtReservoirBuffer; }
    [[nodiscard]] BufferHandle GetRestirPtHistoryReservoirBuffer() const { return _restirPtHistoryReservoirBuffer; }
    [[nodiscard]] IblStats GetIblStats() const;
    [[nodiscard]] RayEffectsStats GetRayEffectsStats() const;
    [[nodiscard]] std::vector<RenderPassDebugInfo> GetRenderPassDebugInfo() const;
    [[nodiscard]] const std::string& GetLastShaderReloadMessage() const { return _lastShaderReloadMessage; }
    [[nodiscard]] bool HasValidVisibilitySet() const { return _visibleSceneToken != nullptr && _visibleSceneToken == _scene.GetPreparedScene(); }
    [[nodiscard]] uint32_t GetResidentTextureCount() const { return static_cast<uint32_t>(_scene.GetResidentTextureCount()); }
    [[nodiscard]] size_t GetRetiredSceneCount() const { return _retiredScenes.size(); }
    [[nodiscard]] uint32_t GetWorkerThreadCount() const { return _jobs.GetWorkerCount(); }
    [[nodiscard]] size_t GetPendingJobCount() const { return _jobs.GetPendingJobCount(); }
    [[nodiscard]] uint32_t GetOfficialGaussianProjectedCount() const;
    [[nodiscard]] uint32_t GetOfficialGaussianDuplicateCount() const;
    [[nodiscard]] uint32_t GetOfficialGaussianPaddedDuplicateCount() const;
    [[nodiscard]] uint32_t GetOfficialGaussianTileCount() const;
    [[nodiscard]] float GetOfficialGaussianAverageTilesTouched() const;
    [[nodiscard]] uint64_t GetOfficialGaussianRebuildCount() const;
    [[nodiscard]] float GetOfficialGaussianPreprocessMs() const;
    [[nodiscard]] float GetOfficialGaussianScanMs() const;
    [[nodiscard]] float GetOfficialGaussianDuplicateMs() const;
    [[nodiscard]] float GetOfficialGaussianSortMs() const;
    [[nodiscard]] float GetOfficialGaussianRangeMs() const;
    [[nodiscard]] float GetOfficialGaussianRasterMs() const;
    [[nodiscard]] float GetOfficialGaussianTotalBuildMs() const;
    [[nodiscard]] bool IsGaussianInteractivePreviewActive() const { return _scene.HasTrainedGaussians() && _gaussianInteractivePreviewFramesRemaining > 0; }
    [[nodiscard]] RenderDevice& GetRenderDevice() { return _device; }
    [[nodiscard]] const RenderDevice& GetRenderDevice() const { return _device; }
    [[nodiscard]] PathTraceBackend GetActivePathTraceBackend() const;
    [[nodiscard]] RendererPreset GetRecommendedPreset() const;
    [[nodiscard]] bool IsSceneLoadInProgress() const { return _sceneLoadInProgress; }
    [[nodiscard]] const std::filesystem::path& GetPendingScenePath() const { return _sceneLoadStatus.path; }
    [[nodiscard]] const std::string& GetSceneLoadStatusMessage() const { return _sceneLoadStatus.message; }
    [[nodiscard]] bool IsStartupSafeModeActive() const { return _startupSafeModeActive; }
    [[nodiscard]] vesta::scene::SceneKind GetRecommendedSceneKind() const;
    [[nodiscard]] RendererDisplayMode GetRecommendedDisplayModeForScene() const;
    [[nodiscard]] const EditorSelection& GetSelection() const { return _selection; }
    [[nodiscard]] std::string GetSelectionLabel() const;
    [[nodiscard]] uint32_t GetEnvironmentSampledImageIndex() const { return _environmentSampledImageIndex; }
    [[nodiscard]] uint32_t GetIblBrdfLutSampledImageIndex() const { return _iblBrdfLutSampledImageIndex; }
    [[nodiscard]] uint32_t GetIblEnvironmentCubemapSampledImageIndex() const { return _iblEnvironmentCubemapSampledImageIndex; }
    [[nodiscard]] uint32_t GetIblDiffuseIrradianceSampledImageIndex() const { return _iblDiffuseIrradianceSampledImageIndex; }
    [[nodiscard]] uint32_t GetIblSpecularPrefilterSampledImageIndex() const { return _iblSpecularPrefilterSampledImageIndex; }
    [[nodiscard]] ImageHandle GetExternalEnvironmentImage() const { return _externalEnvironmentImage; }
    [[nodiscard]] ImageHandle GetIblEnvironmentCubemapImage() const { return _iblEnvironmentCubemapImage; }
    [[nodiscard]] ImageHandle GetIblBrdfLutImage() const { return _iblBrdfLutImage; }
    [[nodiscard]] ImageHandle GetIblDiffuseIrradianceImage() const { return _iblDiffuseIrradianceImage; }
    [[nodiscard]] ImageHandle GetIblSpecularPrefilterImage() const { return _iblSpecularPrefilterImage; }

    void ResetAccumulation() { _pathTraceFrameIndex = 0; }
    bool ReloadShaders();
    bool RequestScreenshot(const std::filesystem::path& path);
    void SetVSyncEnabled(bool enabled);
    void ApplyPreset(RendererPreset preset);
    bool LoadScene(const std::filesystem::path& path);
    bool LoadSceneAsync(const std::filesystem::path& path);
    bool ReloadSceneAsync();
    bool LoadExternalEnvironmentMap(const std::filesystem::path& path);
    void ClearExternalEnvironmentMap();
    bool EnsureRayTracingScene();
    void SetStartupSafeModeActive(bool active) { _startupSafeModeActive = active; }
    void SelectDirectionalLight();
    void SelectPointLight();
    void SelectSpotLight();
    void SelectAreaLight();
    bool SelectObject(uint32_t objectIndex);
    bool SetSelectedObjectPosition(glm::vec3 position);
    bool RotateSelectedObject(glm::vec3 eulerDeltaDegrees);
    bool ScaleSelectedObject(float uniformScale);
    bool UpdateMaterial(uint32_t materialIndex, const vesta::scene::SceneMaterial& material);
    void ClearSelection();
    bool OrbitCameraAroundSelection();
    void OrbitCameraAroundScene();
    bool DollyCameraAroundSelection();
    void DollyCameraAroundScene();
    void DisableCameraOrbit();
    [[nodiscard]] bool IsTrackingSelectionOrbit() const { return _trackSelectedObjectOrbit; }
    void SetOverlayCallbacks(OverlayDrawFn drawFn, OverlaySwapchainCallback swapchainCallback = {});
    void ClearOverlayCallbacks();

    bool RegisterPass(RenderPassRegistrationDesc desc);
    bool UnregisterPass(std::string_view id);
    bool SetPassEnabled(std::string_view id, bool enabled);
    bool SetPassOrder(std::string_view id, uint32_t order);
    [[nodiscard]] IRenderPass* FindPass(std::string_view id);
    [[nodiscard]] const IRenderPass* FindPass(std::string_view id) const;

    template <typename TPass>
    [[nodiscard]] TPass* FindPass(std::string_view id)
    {
        return dynamic_cast<TPass*>(FindPass(id));
    }

    template <typename TPass>
    [[nodiscard]] const TPass* FindPass(std::string_view id) const
    {
        return dynamic_cast<const TPass*>(FindPass(id));
    }

private:
    struct RegisteredPassEntry {
        std::string id;
        std::unique_ptr<IRenderPass> pass;
        RenderPassConfigureFn configure;
        uint32_t order{ 0 };
        bool enabled{ true };
    };

    struct AsyncSceneLoadResult {
        std::filesystem::path path;
        vesta::scene::Scene scene;
        std::string errorMessage;
        float parseMs{ 0.0f };
        float prepareMs{ 0.0f };
        bool success{ false };
    };

    struct RetiredSceneEntry {
        vesta::scene::Scene scene;
        uint64_t safeFrameNumber{ 0 };
    };

    struct VisibilityCullResult {
        std::shared_ptr<const vesta::scene::PreparedScene> scene;
        std::vector<uint32_t> visibleSurfaceIndices;
    };

    enum class PendingSceneUploadStage : uint32_t {
        Idle = 0,
        AllocateBuffers = 1,
        UploadVertices = 2,
        UploadGaussians = 3,
        UploadMaterials = 4,
        UploadIndices = 5,
        UploadTriangles = 6,
        UploadTextures = 7,
        BuildBLAS = 8,
        BuildTLAS = 9,
        SwapScene = 10,
    };

    struct PendingSceneUpload {
        vesta::scene::Scene scene;
        std::filesystem::path path;
        float parseMs{ 0.0f };
        float prepareMs{ 0.0f };
        float uploadMs{ 0.0f };
        float textureUploadMs{ 0.0f };
        PendingSceneUploadStage stage{ PendingSceneUploadStage::Idle };
        size_t vertexOffsetBytes{ 0 };
        size_t gaussianOffsetBytes{ 0 };
        size_t materialOffsetBytes{ 0 };
        size_t indexOffsetBytes{ 0 };
        size_t triangleOffsetBytes{ 0 };
        size_t textureIndex{ 0 };
        bool active{ false };
    };

    void InitializeCommands();
    void InitializeSyncStructures();
    void InitializeDefaultPasses();
    void DestroyFrameResources();
    void ReleaseTransientResources(RendererFrameContext& frameContext);
    void ProcessCompletedFrameReadback(RendererFrameContext& frameContext);
    void RecordScreenshotReadback(VkCommandBuffer commandBuffer, RendererFrameContext& frameContext, uint32_t swapchainImageIndex);
    void RecordOverlay(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex);
    void RecreateSwapchain();
    void ClearPassRegistry();
    void RebuildPassExecutionPlan();
    void PumpSceneLoadRequests();
    void PumpPendingSceneUpload();
    void PumpVisibilityResults();
    void DispatchVisibilityCullIfNeeded();
    void ReleaseRetiredScenes();
    void OnSceneEdited(bool rebuildRayTracing);
    void UpdateSceneEditDrag(const glm::vec2& mousePosition);
    void EndSceneEditDrag();
    [[nodiscard]] std::pair<glm::vec3, glm::vec3> ComputeMouseRay(glm::vec2 mousePosition) const;
    [[nodiscard]] EditorSelection PickSelection(glm::vec2 mousePosition) const;
    [[nodiscard]] SceneUploadOptions GetSceneUploadOptions() const;
    bool LoadSceneResolved(const std::filesystem::path& resolvedPath);
    void StartPendingSceneUpload(vesta::scene::Scene&& scene, float parseMs, float prepareMs);
    void ApplyLoadedScene(vesta::scene::Scene&& scene);
    [[nodiscard]] RendererFrameContext& GetCurrentFrame();
    [[nodiscard]] RenderGraph BuildFrameGraph(uint32_t swapchainImageIndex);
    [[nodiscard]] RegisteredPassEntry* FindPassEntry(std::string_view id);
    [[nodiscard]] const RegisteredPassEntry* FindPassEntry(std::string_view id) const;
    void CreateIblBrdfLut();
    void CreateEnvironmentCubemapImage(std::span<const float> rgbaPixels, uint32_t width, uint32_t height);
    void CreateDiffuseIrradianceEquirect(std::span<const float> rgbaPixels, uint32_t width, uint32_t height);
    void CreateSpecularPrefilterEquirectAtlas(std::span<const float> rgbaPixels, uint32_t width, uint32_t height);
    void DestroyIblResources();
    void EnsureDdgiResources();
    void DestroyDdgiResources();
    void EnsureRestirResources();
    void DestroyRestirResources();

    RenderDevice _device;
    vesta::core::JobSystem _jobs;
    std::array<RendererFrameContext, kFrameOverlap> _frames{};
    std::vector<VkSemaphore> _swapchainImageRenderSemaphores;
    uint64_t _frameNumber{ 0 };
    std::vector<RegisteredPassEntry> _passRegistry;
    std::vector<RegisteredPassEntry*> _passExecutionPlan;
    bool _passExecutionPlanDirty{ true };
    bool _renderGraphTimestampsSupported{ false };
    float _timestampPeriodNs{ 0.0f };
    TransientImagePool _transientImagePool;
    SDL_Window* _window{ nullptr };
    vesta::scene::Scene _scene;
    Camera _camera;
    RendererSettings _settings;
    uint32_t _pathTraceFrameIndex{ 0 };
    float _frameTimeMs{ 0.0f };
    float _smoothedFrameTimeMs{ 0.0f };
    std::array<float, 240> _frameTimeHistoryMs{};
    size_t _frameTimeHistoryHead{ 0 };
    size_t _frameTimeHistoryCount{ 0 };
    std::vector<RenderGraphPassTiming> _lastRenderGraphTimings;
    std::string _lastShaderReloadMessage;
    std::future<AsyncSceneLoadResult> _sceneLoadFuture;
    std::future<VisibilityCullResult> _visibilityFuture;
    SceneLoadStatus _sceneLoadStatus;
    PendingSceneUpload _pendingSceneUpload;
    std::deque<RetiredSceneEntry> _retiredScenes;
    std::vector<uint32_t> _visibleSurfaceIndices;
    FrameSnapshot _frameSnapshot;
    std::shared_ptr<const vesta::scene::PreparedScene> _visibleSceneToken;
    bool _sceneLoadInProgress{ false };
    bool _visibilityCullInProgress{ false };
    bool _visibilityDirty{ true };
    OverlayDrawFn _overlayDrawFn;
    OverlaySwapchainCallback _overlaySwapchainCallback;
    bool _startupSafeModeActive{ false };
    EditorSelection _selection{};
    bool _selectionDragging{ false };
    bool _selectionEditedSinceDragStart{ false };
    bool _trackSelectedObjectOrbit{ false };
    std::filesystem::path _pendingScreenshotPath;
    ImageHandle _externalEnvironmentImage{};
    uint32_t _environmentSampledImageIndex{ kInvalidResourceIndex };
    ImageHandle _iblEnvironmentCubemapImage{};
    uint32_t _iblEnvironmentCubemapSampledImageIndex{ kInvalidResourceIndex };
    ImageHandle _iblDiffuseIrradianceImage{};
    uint32_t _iblDiffuseIrradianceSampledImageIndex{ kInvalidResourceIndex };
    ImageHandle _iblSpecularPrefilterImage{};
    uint32_t _iblSpecularPrefilterSampledImageIndex{ kInvalidResourceIndex };
    ImageHandle _iblBrdfLutImage{};
    uint32_t _iblBrdfLutSampledImageIndex{ kInvalidResourceIndex };
    BufferHandle _ddgiIrradianceBuffer{};
    BufferHandle _ddgiVisibilityBuffer{};
    uint64_t _ddgiIrradianceBufferBytes{ 0 };
    uint64_t _ddgiVisibilityBufferBytes{ 0 };
    BufferHandle _restirReservoirBuffer{};
    BufferHandle _restirHistoryReservoirBuffer{};
    BufferHandle _restirGiReservoirBuffer{};
    BufferHandle _restirGiHistoryReservoirBuffer{};
    BufferHandle _restirPtReservoirBuffer{};
    BufferHandle _restirPtHistoryReservoirBuffer{};
    uint64_t _restirReservoirBufferBytes{ 0 };
    uint64_t _restirHistoryReservoirBufferBytes{ 0 };
    uint64_t _restirGiReservoirBufferBytes{ 0 };
    uint64_t _restirGiHistoryReservoirBufferBytes{ 0 };
    uint64_t _restirPtReservoirBufferBytes{ 0 };
    uint64_t _restirPtHistoryReservoirBufferBytes{ 0 };
    glm::vec2 _lastDragMousePosition{ 0.0f };
    glm::vec3 _dragPlaneOrigin{ 0.0f };
    glm::vec3 _dragPlaneNormal{ 0.0f, 1.0f, 0.0f };
    glm::vec3 _dragGrabOffset{ 0.0f };
    uint32_t _gaussianInteractivePreviewFramesRemaining{ 0 };
};
} // namespace vesta::render

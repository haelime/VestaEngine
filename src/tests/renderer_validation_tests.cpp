#include <cassert>

#include <vesta/render/renderer.h>
#include <vesta/render/vulkan/vk_engine.h>

namespace {
void TestSceneLoadTransitions()
{
    using namespace vesta::render;

    assert(IsValidSceneLoadTransition(SceneLoadState::Idle, SceneLoadState::Parsing));
    assert(IsValidSceneLoadTransition(SceneLoadState::Parsing, SceneLoadState::Preparing));
    assert(IsValidSceneLoadTransition(SceneLoadState::Preparing, SceneLoadState::UploadingGeometry));
    assert(IsValidSceneLoadTransition(SceneLoadState::UploadingGeometry, SceneLoadState::UploadingTextures));
    assert(IsValidSceneLoadTransition(SceneLoadState::UploadingTextures, SceneLoadState::BuildingBLAS));
    assert(IsValidSceneLoadTransition(SceneLoadState::BuildingBLAS, SceneLoadState::BuildingTLAS));
    assert(IsValidSceneLoadTransition(SceneLoadState::BuildingTLAS, SceneLoadState::ReadyToSwap));
    assert(IsValidSceneLoadTransition(SceneLoadState::ReadyToSwap, SceneLoadState::Ready));

    assert(!IsValidSceneLoadTransition(SceneLoadState::Idle, SceneLoadState::Ready));
    assert(!IsValidSceneLoadTransition(SceneLoadState::BuildingTLAS, SceneLoadState::UploadingGeometry));
    assert(!IsValidSceneLoadTransition(SceneLoadState::Ready, SceneLoadState::BuildingBLAS));
}

void TestSceneUploadContinuation()
{
    using namespace vesta::render;

    assert(DecideSceneUploadContinuation(true, true, true, true, true) == SceneUploadContinuation::UploadTextures);
    assert(DecideSceneUploadContinuation(false, false, true, true, true) == SceneUploadContinuation::BuildBLAS);
    assert(DecideSceneUploadContinuation(false, false, true, false, true) == SceneUploadContinuation::ReadyToSwap);
    assert(DecideSceneUploadContinuation(false, false, true, true, false) == SceneUploadContinuation::ReadyToSwap);
}

void TestStartupSafeOverrides()
{
    EngineLaunchOptions options;
    options.safeStartupMode = true;
    options.deferRayTracingBuildUntilAfterFirstPresent = true;

    vesta::render::RendererSettings settings;
    settings.displayMode = vesta::render::RendererDisplayMode::Composite;
    settings.enableGaussian = true;
    settings.enablePathTracing = true;
    settings.buildRayTracingStructuresOnLoad = true;
    settings.textureStreamingEnabled = true;
    settings.enableDistanceCulling = true;
    settings.useIndirectDraw = true;
    settings.sceneUploadMode = vesta::render::SceneUploadMode::AsyncParseSyncUpload;
    settings.preferAsyncSceneLoading = false;

    const vesta::render::RendererSettings safe = ApplyStartupSafeRendererSettings(settings, options);
    assert(safe.displayMode == vesta::render::RendererDisplayMode::DeferredLighting);
    assert(!safe.enableGaussian);
    assert(!safe.enablePathTracing);
    assert(!safe.buildRayTracingStructuresOnLoad);
    assert(!safe.textureStreamingEnabled);
    assert(!safe.enableDistanceCulling);
    assert(!safe.useIndirectDraw);
    assert(safe.sceneUploadMode == vesta::render::SceneUploadMode::Streaming);
    assert(safe.preferAsyncSceneLoading);

    options.safeStartupMode = false;
    const vesta::render::RendererSettings unchanged = ApplyStartupSafeRendererSettings(settings, options);
    assert(unchanged.displayMode == settings.displayMode);
    assert(unchanged.enableGaussian == settings.enableGaussian);
    assert(unchanged.sceneUploadMode == settings.sceneUploadMode);
}
} // namespace

int main()
{
    TestSceneLoadTransitions();
    TestSceneUploadContinuation();
    TestStartupSafeOverrides();
    return 0;
}

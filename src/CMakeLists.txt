# Add source to this project's executable.
add_executable(engine
  app/main.cpp
  render/renderer.cpp
  render/graph/render_graph.cpp
  render/passes/composite_pass.cpp
  render/passes/deferred_lighting_pass.cpp
  render/passes/gaussian_splat_pass.cpp
  render/passes/path_tracer_pass.cpp
  render/passes/geometry_raster_pass.cpp
  render/rhi/render_device.cpp
  render/rhi/vma_impl.cpp
  scene/camera.cpp
  scene/scene.cpp
  render/vulkan/vk_context.cpp
  render/vulkan/vk_descriptors.cpp
  render/vulkan/vk_engine.cpp
  render/vulkan/vk_images.cpp
  render/vulkan/vk_initializers.cpp
  render/vulkan/vk_loader.cpp
  render/vulkan/vk_pipelines.cpp
  ../include/vesta/render/renderer.h
  ../include/vesta/render/graph/render_graph.h
  ../include/vesta/render/passes/composite_pass.h
  ../include/vesta/render/passes/deferred_lighting_pass.h
  ../include/vesta/render/passes/gaussian_splat_pass.h
  ../include/vesta/render/passes/geometry_raster_pass.h
  ../include/vesta/render/passes/irender_pass.h
  ../include/vesta/render/passes/path_tracer_pass.h
  ../include/vesta/render/resources/resource_handles.h
  ../include/vesta/render/rhi/render_device.h
  ../include/vesta/scene/camera.h
  ../include/vesta/scene/scene.h
  ../include/vesta/render/vulkan/vk_context.h
  ../include/vesta/render/vulkan/vk_descriptors.h
  ../include/vesta/render/vulkan/vk_engine.h
  ../include/vesta/render/vulkan/vk_images.h
  ../include/vesta/render/vulkan/vk_initializers.h
  ../include/vesta/render/vulkan/vk_loader.h
  ../include/vesta/render/vulkan/vk_pipelines.h
  ../include/vesta/render/vulkan/vk_types.h
)

add_dependencies(engine Shaders)

set_property(TARGET engine PROPERTY CXX_STANDARD 20)
target_compile_definitions(engine PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_include_directories(engine PUBLIC "${PROJECT_SOURCE_DIR}/include")

if(MSVC)
  # Parallel compilation can contend on the shared engine PDB under Ninja.
  target_compile_options(engine PRIVATE /FS)
endif()

target_link_libraries(engine PUBLIC vma Vulkan::Vulkan fmt::fmt SDL2::SDL2 vkbootstrap glm fastgltf::fastgltf)
target_link_libraries(engine PUBLIC imgui)

target_precompile_headers(engine PUBLIC <optional> <vector> <memory> <string> <vector> <unordered_map> <vulkan/vulkan.h>)

if(WIN32)
  add_custom_command(TARGET engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> $<TARGET_FILE_DIR:engine>
    COMMAND_EXPAND_LISTS
    )
endif()

# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.21)

project(VestaEngine LANGUAGES C CXX)


find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(src)

find_program(GLSL_VALIDATOR
  NAMES glslangValidator
  HINTS
    "$ENV{VULKAN_SDK}/Bin"
    "$ENV{VULKAN_SDK}/Bin32"
)

if(NOT GLSL_VALIDATOR)
  message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK or set VULKAN_SDK.")
endif()

# Avoid CONFIGURE_DEPENDS here: in this environment Ninja/CMake can hang
# indefinitely in "Re-checking globbed directories..." before compilation.
# Re-run CMake manually when adding or removing shader files.
file(GLOB_RECURSE GLSL_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/shaders/*.frag"
  "${PROJECT_SOURCE_DIR}/shaders/*.vert"
  "${PROJECT_SOURCE_DIR}/shaders/*.comp"
  "${PROJECT_SOURCE_DIR}/shaders/*.rgen"
  "${PROJECT_SOURCE_DIR}/shaders/*.rmiss"
  "${PROJECT_SOURCE_DIR}/shaders/*.rchit"
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(SPIRV_BINARY_FILES)

foreach(GLSL IN LISTS GLSL_SOURCE_FILES)
  file(RELATIVE_PATH REL_SHADER "${PROJECT_SOURCE_DIR}/shaders" "${GLSL}")
  set(SPIRV "${SHADER_OUTPUT_DIR}/${REL_SHADER}.spv")
  get_filename_component(SPIRV_DIR "${SPIRV}" DIRECTORY)

  add_custom_command(
    OUTPUT "${SPIRV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_DIR}"
    COMMAND "${GLSL_VALIDATOR}" -V "${GLSL}" -o "${SPIRV}"
    DEPENDS "${GLSL}"
    VERBATIM
  )
  list(APPEND SPIRV_BINARY_FILES "${SPIRV}")
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPIRV_BINARY_FILES})

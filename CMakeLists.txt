cmake_minimum_required(VERSION 3.20)

project(VulkanLearning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

add_executable(VulkanLearning
    main.cpp
    app/src/Runtime/VulkanApplication.cpp
    app/src/Engine/Camera/Camera.cpp
    app/src/ECS/component/Component.cpp
    app/src/ECS/entity/Entity.cpp
    app/src/ECS/core/Scene.cpp
    app/src/ECS/system/CullingSystem.cpp
    app/src/Engine/Events/EventBus.cpp
    app/src/Rendering/RHI/Vulkan/VulkanContext.cpp
    app/src/Rendering/RHI/Vulkan/RayTracingContext.cpp
    app/src/Rendering/RHI/Vulkan/SwapChain.cpp
    app/src/Rendering/RHI/Vulkan/VulkanResourceCreator.cpp
    app/src/Resource/core/Resource.cpp
    app/src/Resource/core/ResourceManager.cpp
    app/src/Resource/model/Model.cpp
    app/src/Resource/model/loaders/GltfModelLoader.cpp
    app/src/Resource/model/loaders/ObjModelLoader.cpp
    app/src/Resource/texture/Texture.cpp
    app/src/Resource/shader/Shader.cpp
    app/src/Rendering/pipeline/GraphicsPipeline.cpp
    app/src/Rendering/core/FrameManager.cpp
    app/src/Rendering/core/Rendergraph.cpp
    app/src/Rendering/renderer/Renderer.cpp
    app/src/Rendering/core/RenderPass.cpp
    app/src/Rendering/core/RenderTarget.cpp
    app/src/Rendering/pass/ForwardPass.cpp
    app/src/Rendering/mesh/GpuMesh.cpp
)

target_include_directories(VulkanLearning PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/app/include
)

# ---- tinygltf (header-only) ----
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.3
)
FetchContent_MakeAvailable(tinygltf)
FetchContent_GetProperties(tinygltf)
target_include_directories(VulkanLearning PRIVATE
    ${tinygltf_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/_deps/tinygltf-src
)

# ---- Vulkan ----
find_package(Vulkan REQUIRED)
target_link_libraries(VulkanLearning PRIVATE Vulkan::Vulkan)

# ---- GLFW (use the one bundled in Vulkan SDK on Windows) ----
if (WIN32)
    if (NOT DEFINED ENV{VULKAN_SDK})
        message(FATAL_ERROR "VULKAN_SDK environment variable is not set. Install Vulkan SDK and set VULKAN_SDK.")
    endif()

    set(_VULKAN_SDK "$ENV{VULKAN_SDK}")
    set(_GLFW_HINT_DIR "${_VULKAN_SDK}/Third-Parties/glfw-3.4.bin.WIN64")

    # stb_image.h (bundled with Vulkan SDK)
    target_include_directories(VulkanLearning PRIVATE
        "${_VULKAN_SDK}/Third-Parties"
    )

    # KTX-Software (bundled with Vulkan SDK)
    target_include_directories(VulkanLearning PRIVATE
        "${_VULKAN_SDK}/Third-Parties/KTX-Software/include"
    )

    find_path(GLFW_INCLUDE_DIR
        NAMES GLFW/glfw3.h
        HINTS "${_GLFW_HINT_DIR}/include"
        REQUIRED
    )

    find_library(GLFW_LIBRARY
        NAMES glfw3
        HINTS "${_GLFW_HINT_DIR}/lib-vc2022"
        REQUIRED
    )

    add_library(GLFW::GLFW STATIC IMPORTED GLOBAL)
    set_target_properties(GLFW::GLFW PROPERTIES
        IMPORTED_LOCATION "${GLFW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
    )

    target_link_libraries(VulkanLearning PRIVATE GLFW::GLFW)
else()
    message(FATAL_ERROR "This CMake setup currently targets Windows (WIN32) only.")
endif()


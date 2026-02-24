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
    app/src/Resource/texture/HdrTextureLoader.cpp
    app/src/Resource/texture/KtxTextureLoader.cpp
    app/src/Resource/texture/Texture.cpp
    app/src/Resource/shader/Shader.cpp
    app/src/Rendering/pipeline/GraphicsPipeline.cpp
    app/src/Rendering/pipeline/DepthPrepassPipeline.cpp
    app/src/Rendering/pipeline/RtaoComputePipeline.cpp
    app/src/Rendering/pipeline/PostProcessPipeline.cpp
    app/src/Rendering/core/FrameManager.cpp
    app/src/Rendering/core/Rendergraph.cpp
    app/src/Rendering/animation/AnimationPlayer.cpp
    app/src/Rendering/renderer/Renderer.cpp
    app/src/Rendering/core/RenderPass.cpp
    app/src/Rendering/core/RenderTarget.cpp
    app/src/Rendering/pass/DepthPrepass.cpp
    app/src/Rendering/pass/ForwardPass.cpp
    app/src/Rendering/pass/BloomExtractPass.cpp
    app/src/Rendering/pass/BloomBlurPass.cpp
    app/src/Rendering/pass/RtaoComputePass.cpp
    app/src/Rendering/pass/TonemapBloomPass.cpp
    app/src/Rendering/mesh/GpuMesh.cpp
    app/src/Rendering/mesh/GlobalMeshBuffer.cpp
    app/src/Rendering/ibl/EquirectToCubemap.cpp
    app/src/Rendering/ibl/IblPrecompute.cpp
    app/src/Rendering/pipeline/SkyboxPipeline.cpp
    app/src/Rendering/pass/SkyboxPass.cpp
    app/src/ImGuiIntegration/ImGuiContext.cpp
)

# Run from project root so assets/ and assets/models/ resolve correctly
set_property(TARGET VulkanLearning PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(VulkanLearning PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/app/include
)

if (MSVC)
    # Ensure UTF-8 source parsing (prevents CP936 multibyte issues with Chinese comments/strings).
    target_compile_options(VulkanLearning PRIVATE /utf-8)
endif()

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

    # ImGui (bundled with Vulkan SDK)
    set(_IMGUI_DIR "${_VULKAN_SDK}/Third-Parties/imgui")
    target_sources(VulkanLearning PRIVATE
        "${_IMGUI_DIR}/imgui.cpp"
        "${_IMGUI_DIR}/imgui_draw.cpp"
        "${_IMGUI_DIR}/imgui_tables.cpp"
        "${_IMGUI_DIR}/imgui_widgets.cpp"
        "${_IMGUI_DIR}/imgui_demo.cpp"
        "${_IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
        "${_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    )
    target_include_directories(VulkanLearning PRIVATE
        "${_IMGUI_DIR}"
        "${_IMGUI_DIR}/backends"
    )

    # KTX-Software (bundled with Vulkan SDK)
    target_include_directories(VulkanLearning PRIVATE
        "${_VULKAN_SDK}/Third-Parties/KTX-Software/include"
    )
    find_library(KTX_LIBRARY
        NAMES ktx
        HINTS "${_VULKAN_SDK}/Third-Parties/KTX-Software/lib"
        REQUIRED
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
    target_link_libraries(VulkanLearning PRIVATE "${KTX_LIBRARY}")
else()
    message(FATAL_ERROR "This CMake setup currently targets Windows (WIN32) only.")
endif()

# ---- Shaders (compile to SPIR-V) ----
find_program(GLSLC_EXECUTABLE glslc)
if (GLSLC_EXECUTABLE)
    set(SHADER_SOURCES
        assets/shaders/VertShaders/pbr.vert
        assets/shaders/FragShaders/pbr.frag
        assets/shaders/VertShaders/depth_prepass.vert
        assets/shaders/FragShaders/depth_only.frag
        assets/shaders/VertShaders/occlusion_bounds.vert
        assets/shaders/VertShaders/cubemap_capture.vert
        assets/shaders/FragShaders/equirect_to_cubemap.frag
        assets/shaders/VertShaders/skybox.vert
        assets/shaders/FragShaders/skybox.frag
        assets/shaders/FragShaders/irradiance_convolution.frag
        assets/shaders/VertShaders/prefilter_capture.vert
        assets/shaders/FragShaders/prefilter.frag
        assets/shaders/VertShaders/brdf_quad.vert
        assets/shaders/VertShaders/fullscreen.vert
        assets/shaders/FragShaders/brdf_integrate.frag
        assets/shaders/FragShaders/bloom_extract.frag
        assets/shaders/FragShaders/bloom_blur.frag
        assets/shaders/FragShaders/tonemap_bloom.frag
        assets/shaders/CompShaders/rtao_trace_half.comp
        assets/shaders/CompShaders/rtao_atrous.comp
        assets/shaders/CompShaders/rtao_upsample.comp
    )

    set(SHADER_SPV)
    foreach(SHADER IN LISTS SHADER_SOURCES)
        set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}")
        set(SPV "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}.spv")
        add_custom_command(
            OUTPUT "${SPV}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.2 "${SRC}" -o "${SPV}"
            DEPENDS "${SRC}"
            COMMENT "Compiling shader ${SHADER}"
            VERBATIM
        )
        list(APPEND SHADER_SPV "${SPV}")
    endforeach()

    add_custom_target(Shaders DEPENDS ${SHADER_SPV})
    add_dependencies(VulkanLearning Shaders)
else()
    message(WARNING "glslc not found; shaders must be precompiled into assets/shaders/*.spv")
endif()


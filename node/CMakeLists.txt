cmake_minimum_required(VERSION 3.20)
project(clap_cpp_node VERSION 0.1.0 LANGUAGES C CXX)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../include/clap.h")
    get_filename_component(CLAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
else()
    set(CLAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/native")
endif()
foreach(source ggml/CMakeLists.txt thirdparty/json/CMakeLists.txt thirdparty/miniaudio/miniaudio.h)
    if(NOT EXISTS "${CLAP_ROOT}/${source}")
        message(FATAL_ERROR "Native sources missing: run git submodule update --init --recursive")
    endif()
endforeach()

# Build a self-contained CPU extension, never CLI executables, GPU backends,
# external OpenMP runtimes, or shared libraries requiring a loader search path.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
foreach(backend METAL CUDA HIP VULKAN SYCL OPENCL OPENVINO RPC MUSA WEBGPU ZDNN
                VIRTGPU VIRTGPU_BACKEND ET HEXAGON ZENDNN BLAS ACCELERATE)
    set(GGML_${backend} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(feature BACKEND_DL OPENMP BUILD_TESTS BUILD_EXAMPLES)
    set(GGML_${feature} OFF CACHE BOOL "" FORCE)
endforeach()
# Avoid host-specific x86 ISA requirements in redistributed wheels.
foreach(isa SSE42 AVX AVX2 AVX_VNNI BMI2 FMA F16C AVX512 AVX512_VBMI
            AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8 AMX_BF16)
    set(GGML_${isa} OFF CACHE BOOL "" FORCE)
endforeach()
set(CLAP_USE_SYSTEM_ICU OFF CACHE BOOL "" FORCE)
add_subdirectory("${CLAP_ROOT}/ggml" ggml EXCLUDE_FROM_ALL)
add_subdirectory("${CLAP_ROOT}/thirdparty/json" json EXCLUDE_FROM_ALL)
include("${CLAP_ROOT}/cmake/icu.cmake")
add_subdirectory("${CLAP_ROOT}/src" core EXCLUDE_FROM_ALL)

# cmake-js supplies Node headers and the import library on Windows.
execute_process(COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE NODE_ADDON_API_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\"" "" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")
add_library(clap_node MODULE bindings.cpp ${CMAKE_JS_SRC})
target_include_directories(clap_node PRIVATE ${CMAKE_JS_INC} "${NODE_ADDON_API_DIR}")
target_link_libraries(clap_node PRIVATE clap ${CMAKE_JS_LIB})
if(APPLE)
    target_link_options(clap_node PRIVATE "-undefined" "dynamic_lookup")
endif()
target_compile_features(clap_node PRIVATE cxx_std_20)
target_compile_definitions(clap_node PRIVATE NAPI_VERSION=8 NAPI_CPP_EXCEPTIONS)
set_target_properties(clap_node PROPERTIES PREFIX "" SUFFIX ".node" CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(clap_node PRIVATE /EHsc)
    # Import the Node-API stubs for Windows when requested by cmake-js.
    if(CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
        execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
            /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
    endif()
else()
    target_compile_options(clap_node PRIVATE -fexceptions)
endif()

cmake_minimum_required(VERSION 3.20)
project(clap_cpp_rust VERSION 0.1.0 LANGUAGES C CXX)

if(NOT DEFINED CLAP_ROOT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/include/clap.h")
        get_filename_component(CLAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../vendor" ABSOLUTE)
    else()
        get_filename_component(CLAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    endif()
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
# Avoid host-specific x86 ISA requirements in downstream executables.
foreach(isa SSE42 AVX AVX2 AVX_VNNI BMI2 FMA F16C AVX512 AVX512_VBMI
            AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8 AMX_BF16)
    set(GGML_${isa} OFF CACHE BOOL "" FORCE)
endforeach()
set(CLAP_USE_SYSTEM_ICU OFF CACHE BOOL "" FORCE)
add_subdirectory("${CLAP_ROOT}/ggml" ggml EXCLUDE_FROM_ALL)
add_subdirectory("${CLAP_ROOT}/thirdparty/json" json EXCLUDE_FROM_ALL)
include("${CLAP_ROOT}/cmake/icu.cmake")
add_subdirectory("${CLAP_ROOT}/src" core EXCLUDE_FROM_ALL)

add_library(clap-rust-native STATIC bridge.cpp)
target_link_libraries(clap-rust-native PRIVATE clap)
target_compile_features(clap-rust-native PRIVATE cxx_std_17)
install(TARGETS clap-rust-native clap ggml ggml-base ggml-cpu clap-icu-i18n clap-icu-uc clap-icu-data
        ARCHIVE DESTINATION lib COMPONENT rust)

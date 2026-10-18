cmake_minimum_required(VERSION 3.20)

project(clap_cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(GGML_NATIVE "Optimize ggml for the build machine" OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule is missing. Run: git submodule update --init --recursive")
endif()

add_subdirectory(ggml EXCLUDE_FROM_ALL)

add_executable(clap src/main.cpp)
target_compile_features(clap PRIVATE cxx_std_17)
target_link_libraries(clap PRIVATE ggml)
set_target_properties(clap PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(clap PRIVATE /W4)
else()
    target_compile_options(clap PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(ggml-smoke tests/ggml-smoke.cpp)
    target_compile_features(ggml-smoke PRIVATE cxx_std_17)
    target_link_libraries(ggml-smoke PRIVATE ggml)
    add_test(NAME ggml_smoke COMMAND ggml-smoke)

    add_test(NAME clap_smoke COMMAND clap)
    set_tests_properties(clap_smoke PROPERTIES PASS_REGULAR_EXPRESSION "clap.cpp")
endif()

install(TARGETS clap RUNTIME DESTINATION bin)

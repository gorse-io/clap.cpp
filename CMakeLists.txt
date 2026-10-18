cmake_minimum_required(VERSION 3.20)

project(clap_cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(GGML_NATIVE "Optimize ggml for the build machine" OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule is missing. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CLI11/CMakeLists.txt")
    message(FATAL_ERROR
        "CLI11 submodule is missing. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cpp-httplib/CMakeLists.txt")
    message(FATAL_ERROR
        "cpp-httplib submodule is missing. Run: git submodule update --init --recursive")
endif()

add_subdirectory(ggml EXCLUDE_FROM_ALL)
add_subdirectory(thirdparty/CLI11 EXCLUDE_FROM_ALL)
add_subdirectory(thirdparty/cpp-httplib EXCLUDE_FROM_ALL)

add_executable(clap src/clap.cpp)
target_compile_features(clap PRIVATE cxx_std_17)
target_link_libraries(clap PRIVATE ggml CLI11::CLI11 httplib::httplib)
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

    add_executable(httplib-smoke tests/httplib-smoke.cpp)
    target_compile_features(httplib-smoke PRIVATE cxx_std_17)
    target_link_libraries(httplib-smoke PRIVATE httplib::httplib)
    add_test(NAME httplib_smoke COMMAND httplib-smoke)

    add_test(NAME clap_smoke COMMAND clap)
    set_tests_properties(clap_smoke PROPERTIES PASS_REGULAR_EXPRESSION "clap.cpp")

    add_test(NAME clap_serve COMMAND clap serve)
    set_tests_properties(clap_serve PROPERTIES PASS_REGULAR_EXPRESSION "hello world")
endif()

install(TARGETS clap RUNTIME DESTINATION bin)

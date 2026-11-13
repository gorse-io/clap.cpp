cmake_minimum_required(VERSION 3.20)

project(clap_cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(CLAP_BUILD_CLI "Build the clap command-line executable" ON)
option(CLAP_STATIC_DEPENDENCIES "Build bundled dependencies as static PIC libraries" OFF)
option(CLAP_GENERATE_LINK_METADATA "Generate a never-built native consumer link probe for File API" OFF)
include(GNUInstallDirs)
include(CTest)

option(GGML_NATIVE "Optimize ggml for the build machine" OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule is missing. Run: git submodule update --init --recursive")
endif()

if(CLAP_BUILD_CLI)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CLI11/CMakeLists.txt")
        message(FATAL_ERROR
            "CLI11 submodule is missing. Run: git submodule update --init --recursive")
    endif()
endif()

if(CLAP_BUILD_CLI OR BUILD_TESTING)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cpp-httplib/CMakeLists.txt")
        message(FATAL_ERROR
            "cpp-httplib submodule is missing. Run: git submodule update --init --recursive")
    endif()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/CMakeLists.txt")
    message(FATAL_ERROR
        "json submodule is missing. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/miniaudio/miniaudio.h")
    message(FATAL_ERROR
        "miniaudio submodule is missing. Run: git submodule update --init --recursive")
endif()


# Scope BUILD_SHARED_LIBS to dependency creation: the core still honors the
# caller's setting, without forcing cache values or duplicating its build graph.
function(clap_add_dependencies)
    if(CLAP_STATIC_DEPENDENCIES)
        set(BUILD_SHARED_LIBS OFF)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    endif()
    add_subdirectory(ggml EXCLUDE_FROM_ALL)
    add_subdirectory(thirdparty/json EXCLUDE_FROM_ALL)
    include(cmake/icu.cmake)
    if(CLAP_BUILD_CLI)
        add_subdirectory(thirdparty/CLI11 EXCLUDE_FROM_ALL)
    endif()
    if(CLAP_BUILD_CLI OR BUILD_TESTING)
        if(BUILD_TESTING)
            set(HTTPLIB_REQUIRE_OPENSSL ON)
        endif()
        add_subdirectory(thirdparty/cpp-httplib EXCLUDE_FROM_ALL)
    endif()
endfunction()
clap_add_dependencies()
add_subdirectory(src)
include(cmake/static-development.cmake)
clap_static_development()

if(CLAP_BUILD_CLI)
    # The CLI target differs from the core library target, while its executable
    # keeps the existing "clap" name and command-line behavior.
    add_executable(clap-cli src/cli.cpp src/embedding.cpp src/search.cpp src/server.cpp)
    target_compile_features(clap-cli PRIVATE cxx_std_20)
    target_link_libraries(clap-cli PRIVATE clap CLI11::CLI11 httplib::httplib nlohmann_json::nlohmann_json)
    set_target_properties(clap-cli PROPERTIES OUTPUT_NAME clap CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(clap-cli PRIVATE /W4)
    else()
        target_compile_options(clap-cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(CLAP_BUILD_CLI)
    install(TARGETS clap-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# The runtime component is also usable by standalone C-library consumers.
# Keep development archives and headers out of binary-only distributions.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/clap.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT development)
foreach(dependency IN ITEMS ggml miniaudio icu json)
    if(dependency STREQUAL "ggml")
        set(license "ggml/LICENSE")
    elseif(dependency STREQUAL "json")
        set(license "thirdparty/json/LICENSE.MIT")
    else()
        set(license "thirdparty/${dependency}/LICENSE")
    endif()
    foreach(component IN ITEMS runtime development)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${license}"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/licenses
            RENAME ${dependency}.txt COMPONENT ${component})
    endforeach()
endforeach()

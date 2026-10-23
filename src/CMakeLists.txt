# One core library contains both shared runtime and architecture implementations,
# following llama.cpp/src/CMakeLists.txt. Models are not a separate library.
file(GLOB CLAP_MODELS_SOURCES CONFIGURE_DEPENDS "models/*.cpp")

set(CLAP_CORE_SOURCES
    clap-api.cpp
    model.cpp
    model-loader.cpp
    model-ops.cpp
    audio.cpp
    vocab.cpp)

add_library(clap
    ../include/clap.h
    ${CLAP_CORE_SOURCES}
    ${CLAP_MODELS_SOURCES})

target_compile_features(clap PRIVATE cxx_std_17)
target_include_directories(clap
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(clap SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/miniaudio")
target_link_libraries(clap PRIVATE
    "$<BUILD_INTERFACE:ggml>" "$<BUILD_INTERFACE:ggml-cpu>"
    "$<INSTALL_INTERFACE:clap::ggml-runtime>"
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>" Threads::Threads ICU::uc ICU::i18n)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(clap PUBLIC CLAP_SHARED PRIVATE CLAP_BUILD)
    # Internal C++ test runners also link to the shared implementation on Windows.
    set_target_properties(clap PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
set_target_properties(clap PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)
if(NOT BUILD_SHARED_LIBS)
    set(CLAP_EXPORT_ARGUMENTS EXPORT clap-targets)
endif()
install(TARGETS clap ${CLAP_EXPORT_ARGUMENTS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ../include/clap.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(GGML_VULKAN)
    target_link_libraries(clap PRIVATE "$<BUILD_INTERFACE:ggml-vulkan>")
    target_compile_definitions(clap PRIVATE CLAP_USE_VULKAN)
endif()
set_target_properties(clap PROPERTIES CXX_EXTENSIONS OFF)
if(NOT MSVC)
    target_compile_options(clap PRIVATE -Wall -Wextra -Wpedantic)
endif()

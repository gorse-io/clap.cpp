# One core library contains both shared runtime and architecture implementations,
# following llama.cpp/src/CMakeLists.txt. Models are not a separate library.
file(GLOB CLAP_MODELS_SOURCES CONFIGURE_DEPENDS "models/*.cpp")

set(CLAP_CORE_SOURCES
    clap.cpp
    model.h
    model.cpp
    model-loader.cpp
    quant.h
    quant.cpp
    graph.cpp
    audio.cpp
    vocab.cpp)

add_library(clap
    ../include/clap.h
    ${CLAP_CORE_SOURCES}
    ${CLAP_MODELS_SOURCES})

target_compile_features(clap PRIVATE cxx_std_20)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(clap PUBLIC CLAP_SHARED PRIVATE CLAP_BUILD)
    set_target_properties(clap PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_include_directories(clap
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(clap SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/miniaudio")
target_link_libraries(clap PUBLIC ggml
    PRIVATE ggml-cpu nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
set_target_properties(clap PROPERTIES CXX_EXTENSIONS OFF)
if(NOT MSVC)
    target_compile_options(clap PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS clap
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT development)

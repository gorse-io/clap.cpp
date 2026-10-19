# One core library contains both shared runtime and architecture implementations,
# following llama.cpp/src/CMakeLists.txt. Models are not a separate library.
file(GLOB CLAP_MODELS_SOURCES CONFIGURE_DEPENDS "models/*.cpp")

set(CLAP_CORE_SOURCES
    model.cpp
    model-loader.cpp
    model-ops.cpp
    audio.cpp
    audio-decoder.cpp
    resample.cpp
    vocab.cpp)

add_library(clap
    ../include/clap.h
    ${CLAP_CORE_SOURCES}
    ${CLAP_MODELS_SOURCES})

target_compile_features(clap PUBLIC cxx_std_17)
target_include_directories(clap
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(clap SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/dr_libs")
target_link_libraries(clap PUBLIC ggml
    PRIVATE ggml-cpu nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
set_target_properties(clap PROPERTIES CXX_EXTENSIONS OFF)
if(NOT MSVC)
    target_compile_options(clap PRIVATE -Wall -Wextra -Wpedantic)
endif()

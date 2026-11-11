# One core library contains both shared runtime and architecture implementations,
# following llama.cpp/src/CMakeLists.txt. Models are not a separate library.
file(GLOB CLAP_MODELS_SOURCES CONFIGURE_DEPENDS "models/*.cpp")

set(CLAP_CORE_SOURCES
    clap.cpp
    model.h
    model.cpp
    model-loader.cpp
    graph.cpp
    tensor-data.cpp
    audio.cpp
    vocab.cpp)

add_library(clap
    ../include/clap.h
    ${CLAP_CORE_SOURCES}
    ${CLAP_MODELS_SOURCES})

# The operation-local CPU specialization must use the same SIMD/rounding policy
# as the pinned GGML CPU backend, without changing its global F16 dot traits.
add_library(clap-qwen3-attention OBJECT qwen3-attention.cpp)
set_target_properties(clap-qwen3-attention PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(clap-qwen3-attention PRIVATE cxx_std_17)
target_compile_options(clap-qwen3-attention PRIVATE "$<TARGET_PROPERTY:ggml-cpu,COMPILE_OPTIONS>")
# Copy backend policy, but never its DLL producer/export ownership. Linking the
# object target as a consumer supplies GGML_SHARED/GGML_BACKEND_SHARED correctly.
target_compile_definitions(clap-qwen3-attention PRIVATE
    "$<FILTER:$<TARGET_PROPERTY:ggml-cpu,COMPILE_DEFINITIONS>,EXCLUDE,^GGML_(BACKEND_)?BUILD(=|$)>")
target_link_libraries(clap-qwen3-attention PRIVATE ggml-cpu ggml-base)
target_include_directories(clap-qwen3-attention SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu")
target_sources(clap PRIVATE $<TARGET_OBJECTS:clap-qwen3-attention>)

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

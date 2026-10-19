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

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/CMakeLists.txt")
    message(FATAL_ERROR
        "json submodule is missing. Run: git submodule update --init --recursive")
endif()

foreach(header dr_wav.h dr_mp3.h dr_flac.h)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/dr_libs/${header}")
        message(FATAL_ERROR
            "dr_libs submodule is missing. Run: git submodule update --init --recursive")
    endif()
endforeach()

add_subdirectory(ggml EXCLUDE_FROM_ALL)
add_subdirectory(thirdparty/CLI11 EXCLUDE_FROM_ALL)
add_subdirectory(thirdparty/cpp-httplib EXCLUDE_FROM_ALL)
add_subdirectory(thirdparty/json EXCLUDE_FROM_ALL)

include(cmake/icu.cmake)
add_subdirectory(src)

add_executable(clap-quantize src/quantize.cpp)
target_compile_features(clap-quantize PRIVATE cxx_std_17)
target_link_libraries(clap-quantize PRIVATE clap)
set_target_properties(clap-quantize PROPERTIES CXX_EXTENSIONS OFF)

add_executable(clap-model examples/embedding/embedding.cpp)
target_link_libraries(clap-model PRIVATE clap CLI11::CLI11 nlohmann_json::nlohmann_json)
set_target_properties(clap-model PROPERTIES CXX_EXTENSIONS OFF)

# The CLI target differs from the core library target, while its executable
# keeps the existing "clap" name and command-line behavior.
add_executable(clap-cli src/clap.cpp src/embedding.cpp src/server.cpp)
target_compile_features(clap-cli PRIVATE cxx_std_17)
target_link_libraries(clap-cli PRIVATE clap CLI11::CLI11 httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(clap-cli PROPERTIES OUTPUT_NAME clap CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(clap-cli PRIVATE /W4)
else()
    target_compile_options(clap-cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(FFMPEG_EXECUTABLE NAMES ffmpeg REQUIRED)
    set(CLAP_AUDIO_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests/audio")
    set(CLAP_AUDIO_FIXTURES
        mono.wav stereo.wav mono.flac stereo.flac mono.mp3 stereo.mp3
        pcm24.wav float.wav 44100.flac audio.dat invalid.bin empty.wav)
    list(TRANSFORM CLAP_AUDIO_FIXTURES PREPEND "${CLAP_AUDIO_FIXTURE_DIR}/")
    add_custom_command(
        OUTPUT ${CLAP_AUDIO_FIXTURES}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio/generate.py"
                "${CLAP_AUDIO_FIXTURE_DIR}" --ffmpeg "${FFMPEG_EXECUTABLE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio/generate.py"
        COMMENT "Generating audio test fixtures"
        VERBATIM)
    add_custom_target(clap-audio-fixtures ALL DEPENDS ${CLAP_AUDIO_FIXTURES})
    add_executable(clap-resample-test tests/resample-test.cpp)
    target_link_libraries(clap-resample-test PRIVATE clap)
    add_test(NAME clap_resample COMMAND clap-resample-test)
    add_executable(clap-audio-test tests/audio-test.cpp)
    target_link_libraries(clap-audio-test PRIVATE clap)
    add_dependencies(clap-audio-test clap-audio-fixtures)
    add_test(NAME clap_audio COMMAND clap-audio-test "${CLAP_AUDIO_FIXTURE_DIR}")
    add_executable(clap-quantize-output-test tests/quantize-output-test.cpp)
    target_compile_features(clap-quantize-output-test PRIVATE cxx_std_17)
    target_include_directories(clap-quantize-output-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME clap_quantize_publication
        COMMAND clap-quantize-output-test "${CMAKE_CURRENT_BINARY_DIR}/quantize-publication-fixture")
    add_executable(clap-q8-test tests/q8-test.cpp)
    target_include_directories(clap-q8-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(clap-q8-test PRIVATE clap)
    add_test(NAME clap_quantize
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize.py"
                "$<TARGET_FILE:clap-q8-test>" "$<TARGET_FILE:clap-quantize>")
    set_tests_properties(clap_quantize PROPERTIES TIMEOUT 60)
    foreach(regression alignment symlink)
        add_test(NAME clap_quantize_${regression}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize.py"
                    "$<TARGET_FILE:clap-q8-test>" "$<TARGET_FILE:clap-quantize>" "${regression}")
        set_tests_properties(clap_quantize_${regression} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    endforeach()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_library(clap-quantize-race SHARED tests/quantize-race.cpp)
        target_link_libraries(clap-quantize-race PRIVATE ${CMAKE_DL_LIBS})
        add_test(NAME clap_quantize_concurrent
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize.py"
                    "$<TARGET_FILE:clap-q8-test>" "$<TARGET_FILE:clap-quantize>" concurrent
                    "$<TARGET_FILE:clap-quantize-race>")
        set_tests_properties(clap_quantize_concurrent PROPERTIES TIMEOUT 60)
    endif()
    add_executable(clap-model-test tests/models/model-test.cpp)
    target_link_libraries(clap-model-test PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME clap_model_unit COMMAND clap-model-test)
    add_executable(ggml-smoke tests/ggml-smoke.cpp)
    target_compile_features(ggml-smoke PRIVATE cxx_std_17)
    target_link_libraries(ggml-smoke PRIVATE ggml)
    add_test(NAME ggml_smoke COMMAND ggml-smoke)

    add_test(NAME clap_smoke COMMAND clap-cli)
    set_tests_properties(clap_smoke PROPERTIES PASS_REGULAR_EXPRESSION "clap.cpp")

    add_test(NAME clap_audio_cli
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio_cli.py" "$<TARGET_FILE:clap-model>"
                "${CLAP_AUDIO_FIXTURE_DIR}")
    set_tests_properties(clap_audio_cli PROPERTIES TIMEOUT 30)
    add_test(NAME clap_embeddings
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/embeddings.py" "$<TARGET_FILE:clap-cli>")
    set_tests_properties(clap_embeddings PROPERTIES TIMEOUT 30)
    option(CLAP_MODEL_PARITY "Run optional Transformers oracle tests (torch/transformers/numpy required)" OFF)
    if(CLAP_MODEL_PARITY)
        add_test(NAME clap_model_parity
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/parity.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/model-fixture")
        set_tests_properties(clap_model_parity PROPERTIES TIMEOUT 180)
        add_test(NAME clap_model_cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/cli.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/model-fixture")
        set_tests_properties(clap_model_cli PROPERTIES DEPENDS clap_model_parity TIMEOUT 60)
    endif()
    option(CLAMP3_MODEL_PARITY "Run optional CLaMP3 Transformers oracle tests (torch/transformers/numpy/tokenizers required)" OFF)
    if(CLAMP3_MODEL_PARITY)
        add_executable(clamp3-model-test tests/models/clamp3-model-test.cpp)
        target_link_libraries(clamp3-model-test PRIVATE clap nlohmann_json::nlohmann_json)
        add_executable(clap-unigram-test tests/models/unigram-test.cpp)
        target_include_directories(clap-unigram-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(clap-unigram-test PRIVATE clap nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
        add_test(NAME clamp3_tokenizer
            COMMAND clap-unigram-test "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/tokenizer.json"
                    "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/tokenizer-cases.json")
        set_tests_properties(clamp3_tokenizer PROPERTIES DEPENDS clamp3_model_parity TIMEOUT 120)
        add_test(NAME clamp3_model_parity
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_parity.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture")
        set_tests_properties(clamp3_model_parity PROPERTIES TIMEOUT 300)
        add_test(NAME clamp3_model_native
            COMMAND clamp3-model-test
                    "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/model.gguf"
                    "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/fixture.json")
        set_tests_properties(clamp3_model_native PROPERTIES DEPENDS clamp3_model_parity TIMEOUT 120)
        add_test(NAME clamp3_model_native_f16
            COMMAND clamp3-model-test
                    "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/model-f16.gguf"
                    "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/fixture-f16.json")
        set_tests_properties(clamp3_model_native_f16 PROPERTIES DEPENDS clamp3_model_parity TIMEOUT 120)
        add_test(NAME clamp3_model_cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_cli.py"
                    "$<TARGET_FILE:clap-model>" "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture")
        set_tests_properties(clamp3_model_cli PROPERTIES DEPENDS clamp3_model_parity TIMEOUT 120)
        add_test(NAME clamp3_invalid_metadata
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_invalid.py"
                    "$<TARGET_FILE:clap-model>" "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture")
        set_tests_properties(clamp3_invalid_metadata PROPERTIES DEPENDS clamp3_model_parity TIMEOUT 120)
    endif()
    add_test(NAME clap_embedding_cli
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/embedding_cli.py" "$<TARGET_FILE:clap-cli>")
    set_tests_properties(clap_embedding_cli PROPERTIES TIMEOUT 30)
endif()

install(TARGETS clap-cli clap-model clap-quantize RUNTIME DESTINATION bin)

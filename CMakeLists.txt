cmake_minimum_required(VERSION 3.20)

project(clap_cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(CLAP_BUILD_CLI "Build the clap command-line executable" ON)
option(CLAP_STATIC_DEPENDENCIES "Build bundled dependencies as static PIC libraries" OFF)
option(CLAP_GENERATE_LINK_METADATA "Generate a never-built native consumer link probe for File API" OFF)
include(GNUInstallDirs)

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
    add_executable(clap-cli src/cli.cpp src/embedding.cpp src/search.cpp src/server.cpp src/quantize.cpp)
    target_compile_features(clap-cli PRIVATE cxx_std_20)
    target_link_libraries(clap-cli PRIVATE clap CLI11::CLI11 httplib::httplib nlohmann_json::nlohmann_json)
    set_target_properties(clap-cli PROPERTIES OUTPUT_NAME clap CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(clap-cli PRIVATE /W4)
    else()
        target_compile_options(clap-cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/CMakeLists.txt")
        message(FATAL_ERROR
            "GoogleTest submodule is missing. Run: git submodule update --init --recursive")
    endif()
    set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use the shared MSVC runtime" FORCE)
    add_subdirectory(thirdparty/googletest EXCLUDE_FROM_ALL)
    if(BUILD_SHARED_LIBS)
        # GoogleTest only sets this for installed consumers, not build-tree users.
        target_compile_definitions(gtest INTERFACE GTEST_LINKED_AS_SHARED_LIBRARY=1)
    endif()

    if(NOT BUILD_SHARED_LIBS)
        add_executable(test-abi tests/test-abi.cpp)
        target_link_libraries(test-abi PRIVATE clap GTest::gtest_main)
        add_test(NAME test-clap-c-api-allocation COMMAND test-abi)
    endif()
    add_executable(test-byte-bpe tests/models/test-byte-bpe.cpp)
    target_include_directories(test-byte-bpe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test-byte-bpe PRIVATE clap nlohmann_json::nlohmann_json ICU::i18n)
    add_test(NAME test-clap-byte-bpe COMMAND test-byte-bpe)
    add_test(NAME test-clap-conversion-imports
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/conversion.py")
    set_tests_properties(test-clap-conversion-imports PROPERTIES TIMEOUT 30)
    find_program(FFMPEG_EXECUTABLE NAMES ffmpeg REQUIRED)
    set(CLAP_AUDIO_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests/audio")
    set(CLAP_AUDIO_FIXTURES
        mono.wav stereo.wav mono.flac stereo.flac mono.mp3 stereo.mp3
        pcm24.wav float.wav 44100.flac audio.dat invalid.bin empty.wav
        truncated.wav truncated.flac nonfinite.wav surround.wav unknown-length.flac)
    list(TRANSFORM CLAP_AUDIO_FIXTURES PREPEND "${CLAP_AUDIO_FIXTURE_DIR}/")
    add_custom_command(
        OUTPUT ${CLAP_AUDIO_FIXTURES}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio/generate.py"
                "${CLAP_AUDIO_FIXTURE_DIR}" --ffmpeg "${FFMPEG_EXECUTABLE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio/generate.py"
        COMMENT "Generating audio test fixtures"
        VERBATIM)
    add_custom_target(clap-audio-fixtures ALL DEPENDS ${CLAP_AUDIO_FIXTURES})
    add_executable(clap-embeddinggemma2-test tests/models/embeddinggemma2-test.cpp)
    target_link_libraries(clap-embeddinggemma2-test PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME embeddinggemma2_model
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/embeddinggemma2.py"
                --test "$<TARGET_FILE:clap-embeddinggemma2-test>")
    set_tests_properties(embeddinggemma2_model PROPERTIES TIMEOUT 120)
    if(CLAP_BUILD_CLI)
        add_test(NAME embeddinggemma2_cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/embeddinggemma2_cli.py"
                    "$<TARGET_FILE:clap-cli>")
        set_tests_properties(embeddinggemma2_cli PROPERTIES TIMEOUT 120)
    endif()
    add_executable(clap-gemma-tokenizer-test tests/models/gemma-tokenizer-test.cpp)
    target_include_directories(clap-gemma-tokenizer-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(clap-gemma-tokenizer-test PRIVATE clap nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
    add_test(NAME gemma_tokenizer COMMAND clap-gemma-tokenizer-test)
    set(CLAP_GEMMA_TOKENIZER_JSON "" CACHE FILEPATH "Pinned EmbeddingGemma2 tokenizer.json for optional HF parity (Python tokenizers required)")
    if(CLAP_GEMMA_TOKENIZER_JSON)
        if(NOT EXISTS "${CLAP_GEMMA_TOKENIZER_JSON}")
            message(FATAL_ERROR "CLAP_GEMMA_TOKENIZER_JSON does not exist")
        endif()
        set(GEMMA_TOKENIZER_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/gemma-tokenizer-parity")
        add_test(NAME gemma_tokenizer_hf_fixture
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/gemma_tokenizer_parity.py"
                    "${CLAP_GEMMA_TOKENIZER_JSON}" "${GEMMA_TOKENIZER_FIXTURE}")
        set_tests_properties(gemma_tokenizer_hf_fixture PROPERTIES FIXTURES_SETUP gemma_tokenizer_hf TIMEOUT 120)
        add_test(NAME gemma_tokenizer_hf
            COMMAND clap-gemma-tokenizer-test "${GEMMA_TOKENIZER_FIXTURE}/metadata.json" "${GEMMA_TOKENIZER_FIXTURE}/cases.json")
        set_tests_properties(gemma_tokenizer_hf PROPERTIES FIXTURES_REQUIRED gemma_tokenizer_hf TIMEOUT 120)
    endif()
    add_executable(clap-gemma-storage-test tests/gemma-storage-test.cpp)
    target_include_directories(clap-gemma-storage-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(clap-gemma-storage-test PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME gemma_storage COMMAND clap-gemma-storage-test "${CMAKE_CURRENT_BINARY_DIR}/gemma-storage-")
    add_executable(clap-metadata-test tests/metadata-test.cpp)
    target_include_directories(clap-metadata-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(clap-metadata-test PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME clap_metadata COMMAND clap-metadata-test "${CMAKE_CURRENT_BINARY_DIR}/metadata.gguf")
    add_executable(test-audio tests/test-audio.cpp)
    target_link_libraries(test-audio PRIVATE clap GTest::gtest)
    if(WIN32 AND BUILD_SHARED_LIBS)
        add_custom_command(TARGET test-audio POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:gtest>" "$<TARGET_FILE_DIR:test-audio>"
            VERBATIM)
    endif()
    add_dependencies(test-audio clap-audio-fixtures)
    add_test(NAME test-clap-audio COMMAND test-audio "${CLAP_AUDIO_FIXTURE_DIR}")
    add_executable(test-audio-path tests/test-audio-path.cpp)
    target_link_libraries(test-audio-path PRIVATE clap)
    add_dependencies(test-audio-path clap-audio-fixtures)
    add_test(NAME test-clap-audio-path
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio_path.py"
                "$<TARGET_FILE:test-audio-path>" "${CLAP_AUDIO_FIXTURE_DIR}")
    set_tests_properties(test-clap-audio-path PROPERTIES TIMEOUT 120)
    add_executable(test-quantize-output tests/test-quantize-output.cpp)
    target_compile_features(test-quantize-output PRIVATE cxx_std_20)
    target_include_directories(test-quantize-output PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME test-clap-quantize-publication
        COMMAND test-quantize-output "${CMAKE_CURRENT_BINARY_DIR}/quantize-publication-fixture")
    add_executable(test-bf16 tests/test-bf16.cpp src/quantize.cpp)
    target_include_directories(test-bf16 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test-bf16 PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME test-clap-bf16 COMMAND test-bf16 "${CMAKE_CURRENT_BINARY_DIR}/bf16-fixture")
    set_tests_properties(test-clap-bf16 PROPERTIES TIMEOUT 120)
    add_executable(test-all-quantization tests/test-all-quantization.cpp src/quantize.cpp)
    target_include_directories(test-all-quantization PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test-all-quantization PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME test-all-architecture-quantization COMMAND test-all-quantization "${CMAKE_CURRENT_BINARY_DIR}/all-quantization-fixture")
    set_tests_properties(test-all-architecture-quantization PROPERTIES TIMEOUT 120)
    add_executable(test-q8 tests/test-q8.cpp)
    target_include_directories(test-q8 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test-q8 PRIVATE clap nlohmann_json::nlohmann_json)
    if(CLAP_BUILD_CLI)
        add_test(NAME test-clap-bf16-cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/bf16_cli.py"
                    "$<TARGET_FILE:clap-cli>" "$<TARGET_FILE:test-q8>")
        set_tests_properties(test-clap-bf16-cli PROPERTIES TIMEOUT 120)
        add_test(NAME test-clap-quantize-cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize_cli.py" "$<TARGET_FILE:clap-cli>")
        set_tests_properties(test-clap-quantize-cli PROPERTIES TIMEOUT 30)
        add_test(NAME test-clap-quantize
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize.py"
                    "$<TARGET_FILE:test-q8>" "$<TARGET_FILE:clap-cli>")
        set_tests_properties(test-clap-quantize PROPERTIES TIMEOUT 60)
        foreach(regression alignment symlink k)
            add_test(NAME test-clap-quantize-${regression}
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize.py"
                        "$<TARGET_FILE:test-q8>" "$<TARGET_FILE:clap-cli>" "${regression}")
            set_tests_properties(test-clap-quantize-${regression} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
        endforeach()
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_library(test-quantize-race SHARED tests/test-quantize-race.cpp)
            target_link_libraries(test-quantize-race PRIVATE ${CMAKE_DL_LIBS})
            add_test(NAME test-clap-quantize-concurrent
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantize.py"
                        "$<TARGET_FILE:test-q8>" "$<TARGET_FILE:clap-cli>" concurrent
                        "$<TARGET_FILE:test-quantize-race>")
            set_tests_properties(test-clap-quantize-concurrent PROPERTIES TIMEOUT 60)
        endif()
    endif()
    add_executable(test-feature-input tests/models/test-feature-input.cpp)
    target_link_libraries(test-feature-input PRIVATE clap nlohmann_json::nlohmann_json)
    target_include_directories(test-feature-input PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(CLAP_BUILD_CLI)
        add_executable(test-clap2-model tests/models/test-clap2-model.cpp)
        target_link_libraries(test-clap2-model PRIVATE clap nlohmann_json::nlohmann_json)
        add_test(NAME test-clap2-contract
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clap2_invalid.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/clap2-smoke")
        set_tests_properties(test-clap2-contract PROPERTIES TIMEOUT 180 FIXTURES_SETUP clap2_smoke)
        add_executable(test-clap2-rope tests/models/test-clap2-rope.cpp)
        target_include_directories(test-clap2-rope PRIVATE src)
        target_link_libraries(test-clap2-rope PRIVATE clap ggml::ggml nlohmann_json::nlohmann_json)
        add_test(NAME test-clap2-rope
            COMMAND test-clap2-rope "${CMAKE_CURRENT_BINARY_DIR}/clap2-smoke/model-f32.gguf")
        set_tests_properties(test-clap2-rope PROPERTIES
            FIXTURES_REQUIRED clap2_smoke TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=2")
        add_test(NAME test-clap2-oracle-config
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clap2_oracle_test.py")
        add_test(NAME test-clap2-sizes
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clap2_sizes.py")
        option(CLAP2_MODEL_PARITY "Run optional pinned open_clip CLAP2 numerical oracles" OFF)
        set(CLAP2_REFERENCE_SOURCE "" CACHE PATH "Pinned official open_clip source directory")
        set(CLAP2_REFERENCE_ASSETS "" CACHE PATH "Pinned CLAP2 config/tokenizer/checkpoint-header assets")
        if(CLAP2_MODEL_PARITY)
            if(NOT EXISTS "${CLAP2_REFERENCE_SOURCE}/src/open_clip/audio/naflex_audio.py" OR
               NOT EXISTS "${CLAP2_REFERENCE_ASSETS}/config.json")
                message(FATAL_ERROR "CLAP2_REFERENCE_SOURCE and CLAP2_REFERENCE_ASSETS must contain pinned official sources/assets")
            endif()
            add_test(NAME test-clap2-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clap2_parity.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/clap2-fixture"
                        "${CLAP2_REFERENCE_SOURCE}" "${CLAP2_REFERENCE_ASSETS}")
            set_tests_properties(test-clap2-model-parity PROPERTIES TIMEOUT 600 RESOURCE_LOCK clap2_oracles)
            foreach(profile IN ITEMS xxs-head56 asymmetric-head64)
                add_test(NAME test-clap2-model-parity-${profile}
                    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clap2_parity.py"
                            "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/clap2-fixture-${profile}"
                            "${CLAP2_REFERENCE_SOURCE}" "${CLAP2_REFERENCE_ASSETS}" --profile "${profile}")
                set_tests_properties(test-clap2-model-parity-${profile} PROPERTIES
                    TIMEOUT 600 RESOURCE_LOCK clap2_oracles)
            endforeach()
        endif()
        add_test(NAME test-clamp3-bundle-contract
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_bundle_contract.py" "$<TARGET_FILE:clap-cli>")
    endif()
    add_executable(test-model tests/models/test-model.cpp)
    target_link_libraries(test-model PRIVATE clap nlohmann_json::nlohmann_json)
    target_include_directories(test-model PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME test-clap-model-unit COMMAND test-model)
    add_executable(test-graph-pipeline tests/test-graph-pipeline.cpp)
    target_include_directories(test-graph-pipeline PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test-graph-pipeline PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME test-clap-graph-pipeline
        COMMAND test-graph-pipeline "${CMAKE_CURRENT_BINARY_DIR}/graph-pipeline.gguf")
    add_executable(test-tensor-ops tests/test-tensor-ops.cpp)
    target_include_directories(test-tensor-ops PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test-tensor-ops PRIVATE clap nlohmann_json::nlohmann_json)
    add_test(NAME test-clap-tensor-ops
        COMMAND test-tensor-ops "${CMAKE_CURRENT_BINARY_DIR}/tensor-ops-auto.gguf")
    add_test(NAME test-clap-tensor-ops-cpu
        COMMAND test-tensor-ops "${CMAKE_CURRENT_BINARY_DIR}/tensor-ops-cpu.gguf" --require-cpu)
    set_tests_properties(test-clap-tensor-ops-cpu PROPERTIES ENVIRONMENT "CLAP_DEVICE=cpu")
    if(GGML_VULKAN)
        add_test(NAME test-clap-tensor-ops-no-device
            COMMAND test-tensor-ops "${CMAKE_CURRENT_BINARY_DIR}/tensor-ops-no-device.gguf" --require-cpu)
        set_tests_properties(test-clap-tensor-ops-no-device PROPERTIES ENVIRONMENT
            "VK_DRIVER_FILES=${CMAKE_CURRENT_BINARY_DIR}/missing-vulkan-icd.json;VK_ICD_FILENAMES=${CMAKE_CURRENT_BINARY_DIR}/missing-vulkan-icd.json")
    endif()
    option(CLAP_VULKAN_TEST_REQUIRE_DEVICE "Require actual Vulkan execution in tensor tests" OFF)
    if(CLAP_VULKAN_TEST_REQUIRE_DEVICE)
        add_test(NAME test-clap-tensor-ops-vulkan
            COMMAND test-tensor-ops "${CMAKE_CURRENT_BINARY_DIR}/tensor-ops-vulkan.gguf" --require-vulkan)
    endif()
    add_executable(test-mert2-fft tests/models/test-mert2-fft.cpp)
    target_compile_features(test-mert2-fft PRIVATE cxx_std_20)
    target_link_libraries(test-mert2-fft PRIVATE clap)
    add_test(NAME test-mert2-fft-precision COMMAND test-mert2-fft)

    if(CLAP_BUILD_CLI)
        add_test(NAME test-clap-smoke COMMAND clap-cli)
        set_tests_properties(test-clap-smoke PROPERTIES PASS_REGULAR_EXPRESSION "clap.cpp")

        add_test(NAME test-clap-audio-cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio_cli.py" "$<TARGET_FILE:clap-cli>"
                    "${CLAP_AUDIO_FIXTURE_DIR}")
        set_tests_properties(test-clap-audio-cli PROPERTIES TIMEOUT 30)
        add_test(NAME test-clap-embeddings
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/embeddings.py" "$<TARGET_FILE:clap-cli>")
        set_tests_properties(test-clap-embeddings PROPERTIES TIMEOUT 30)
        add_test(NAME test-clap-audio-embeddings
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio_embeddings.py"
                    "$<TARGET_FILE:clap-cli>" "${CLAP_AUDIO_FIXTURE_DIR}")
        set_tests_properties(test-clap-audio-embeddings PROPERTIES TIMEOUT 120)
        option(CLAP_MODEL_PARITY "Run optional Transformers oracle tests (torch/transformers/numpy required)" OFF)
        if(CLAP_MODEL_PARITY)
            add_test(NAME test-clap-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/parity.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/model-fixture")
            set_tests_properties(test-clap-model-parity PROPERTIES TIMEOUT 600)
            add_test(NAME test-clap-model-cli
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/cli.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/model-fixture")
            set_tests_properties(test-clap-model-cli PROPERTIES DEPENDS test-clap-model-parity TIMEOUT 60)
        endif()
        option(CLAMP3_MODEL_PARITY "Run optional CLaMP3 Transformers oracle tests (torch/transformers/numpy/tokenizers required)" OFF)
        if(CLAMP3_MODEL_PARITY)
            add_executable(test-clamp3-model tests/models/test-clamp3-model.cpp)
            target_link_libraries(test-clamp3-model PRIVATE clap nlohmann_json::nlohmann_json)
            target_include_directories(test-clamp3-model PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
            add_executable(test-unigram tests/models/test-unigram.cpp)
            target_include_directories(test-unigram PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
            target_link_libraries(test-unigram PRIVATE clap nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
            add_test(NAME test-clamp3-tokenizer
                COMMAND test-unigram "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/tokenizer.json"
                        "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/tokenizer-cases.json")
            set_tests_properties(test-clamp3-tokenizer PROPERTIES DEPENDS test-clamp3-model-parity TIMEOUT 120)
            add_test(NAME test-clamp3-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_parity.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture")
            set_tests_properties(test-clamp3-model-parity PROPERTIES TIMEOUT 600)
            add_test(NAME test-clamp3-model-native
                COMMAND test-clamp3-model
                        "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/model.gguf"
                        "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/fixture.json")
            set_tests_properties(test-clamp3-model-native PROPERTIES DEPENDS test-clamp3-model-parity TIMEOUT 120)
            add_test(NAME test-clamp3-model-native-f16
                COMMAND test-clamp3-model
                        "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/model-f16.gguf"
                        "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture/fixture-f16.json")
            set_tests_properties(test-clamp3-model-native-f16 PROPERTIES DEPENDS test-clamp3-model-parity TIMEOUT 120)
            add_test(NAME test-clamp3-model-cli
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_cli.py"
                        "$<TARGET_FILE:clap-cli>" "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture")
            set_tests_properties(test-clamp3-model-cli PROPERTIES DEPENDS test-clamp3-model-parity TIMEOUT 120)
            add_test(NAME test-clamp3-invalid-metadata
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_invalid.py"
                        "$<TARGET_FILE:clap-cli>" "${CMAKE_CURRENT_BINARY_DIR}/clamp3-fixture")
            set_tests_properties(test-clamp3-invalid-metadata PROPERTIES DEPENDS test-clamp3-model-parity TIMEOUT 120)
        endif()
        option(MERT2_MODEL_PARITY "Run optional official MERT-v2-FullSong oracle tests (torch/torchaudio/transformers/safetensors required)" OFF)
        add_test(NAME test-mert-invalid-metadata
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/mert_invalid.py"
                    "$<TARGET_FILE:clap-cli>")
        set_tests_properties(test-mert-invalid-metadata PROPERTIES TIMEOUT 30)
        option(MERT_MODEL_PARITY "Run optional reviewed official MERT-v1 numerical oracles" OFF)
        set(MERT_REFERENCE_SOURCE "" CACHE PATH "Reviewed pinned MERT-v1 source (modeling_MERT.py/configuration_MERT.py)")
        if(MERT_MODEL_PARITY)
            if(NOT EXISTS "${MERT_REFERENCE_SOURCE}/modeling_MERT.py")
                message(FATAL_ERROR "MERT_REFERENCE_SOURCE must contain reviewed pinned official source")
            endif()
            add_executable(test-mert-model tests/models/test-mert-model.cpp)
            target_link_libraries(test-mert-model PRIVATE clap nlohmann_json::nlohmann_json)
            target_include_directories(test-mert-model PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
            # Private MERT feature bridge for tiny and published-checkpoint Python oracles.
            add_executable(test-feature-extract tests/models/test-feature-extract.cpp)
            target_link_libraries(test-feature-extract PRIVATE clap nlohmann_json::nlohmann_json)
            target_include_directories(test-feature-extract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
            add_test(NAME test-mert-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/mert_parity.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mert-fixture" "${MERT_REFERENCE_SOURCE}")
            set_tests_properties(test-mert-model-parity PROPERTIES TIMEOUT 600)
            if(CLAMP3_MODEL_PARITY)
                add_executable(test-clamp3-bundled tests/models/test-clamp3-bundled.cpp)
                target_link_libraries(test-clamp3-bundled PRIVATE clap nlohmann_json::nlohmann_json)
                target_include_directories(test-clamp3-bundled PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
                add_test(NAME test-clamp3-auto-bundle
                    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_auto_bundle.py"
                            "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mert-fixture")
                set_tests_properties(test-clamp3-auto-bundle PROPERTIES DEPENDS test-mert-clamp3-pipeline TIMEOUT 120)
                add_test(NAME test-clamp3-bundled
                    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/clamp3_bundled.py"
                            "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mert-fixture")
                set_tests_properties(test-clamp3-bundled PROPERTIES DEPENDS test-mert-clamp3-pipeline TIMEOUT 300)
                add_test(NAME test-mert-clamp3-pipeline
                    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/mert_pipeline.py"
                            "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mert-fixture")
                set_tests_properties(test-mert-clamp3-pipeline PROPERTIES DEPENDS "test-mert-model-parity;test-clamp3-model-parity" TIMEOUT 300)
            endif()
        endif()
        set(MERT2_REFERENCE_SOURCE "" CACHE PATH "Reviewed pinned MERT-v2-FullSong source directory")
        if(MERT2_MODEL_PARITY)
            if(NOT EXISTS "${MERT2_REFERENCE_SOURCE}/modeling_mert2.py")
                message(FATAL_ERROR "MERT2_REFERENCE_SOURCE must contain the reviewed pinned official source")
            endif()
            add_executable(test-mert2-model tests/models/test-mert2-model.cpp)
            target_link_libraries(test-mert2-model PRIVATE clap nlohmann_json::nlohmann_json)
            add_test(NAME test-mert2-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/mert2_parity.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mert2-fixture"
                        "${MERT2_REFERENCE_SOURCE}")
            set_tests_properties(test-mert2-model-parity PROPERTIES TIMEOUT 600)
            add_test(NAME test-mert2-model-cli
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/mert2_cli.py"
                        "$<TARGET_FILE:clap-cli>" "${CMAKE_CURRENT_BINARY_DIR}/mert2-fixture")
            set_tests_properties(test-mert2-model-cli PROPERTIES DEPENDS test-mert2-model-parity TIMEOUT 180)
        endif()
        add_test(NAME test-muq-conversion-contract
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_conversion_test.py")
        add_test(NAME test-muq-quantized-rejection
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_quantized.py"
                    "$<TARGET_FILE:clap-cli>" "$<TARGET_FILE:clap-cli>")
        set_tests_properties(test-muq-quantized-rejection PROPERTIES TIMEOUT 120)
        option(MUQ_MODEL_PARITY "Run optional MuQ official-component waveform oracle tests" OFF)
        set(MUQ_REFERENCE_SOURCE "" CACHE PATH "Pinned official MuQ conv.py/features.py directory")
        if(MUQ_MODEL_PARITY)
            if(NOT EXISTS "${MUQ_REFERENCE_SOURCE}/conv.py" OR NOT EXISTS "${MUQ_REFERENCE_SOURCE}/features.py")
                message(FATAL_ERROR "MUQ_REFERENCE_SOURCE must contain pinned official conv.py and features.py")
            endif()
            add_executable(test-muq-model tests/models/test-muq-model.cpp)
            target_link_libraries(test-muq-model PRIVATE clap nlohmann_json::nlohmann_json)
            add_test(NAME test-muq-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_parity.py"
                        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/muq-fixture"
                        "${MUQ_REFERENCE_SOURCE}")
            set_tests_properties(test-muq-model-parity PROPERTIES TIMEOUT 600)
        endif()
        add_test(NAME test-muq-mulan-conversion-contract
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_mulan_conversion_test.py")
        add_test(NAME test-muq-mulan-invalid
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_mulan_invalid.py" "$<TARGET_FILE:clap-cli>")
        option(MUQ_MULAN_MODEL_PARITY "Run tiny pinned official MuLan component oracles" OFF)
        set(MUQ_MULAN_REFERENCE_SOURCE "" CACHE PATH "Pinned MuLan source directory")
        if(MUQ_MULAN_MODEL_PARITY)
            add_executable(test-muq-mulan-model tests/models/test-muq-mulan-model.cpp)
            target_link_libraries(test-muq-mulan-model PRIVATE clap nlohmann_json::nlohmann_json)
            add_test(NAME test-muq-mulan-model-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_mulan_parity.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/muq-mulan-fixture"
                    "${MUQ_MULAN_REFERENCE_SOURCE}" "${MUQ_REFERENCE_SOURCE}")
            set_tests_properties(test-muq-mulan-model-parity PROPERTIES TIMEOUT 600)
            add_test(NAME test-muq-mulan-oracle-pipeline
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/muq_mulan_oracle_pipeline.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/muq-mulan-fixture"
                    "${MUQ_MULAN_REFERENCE_SOURCE}" "${MUQ_REFERENCE_SOURCE}")
            set_tests_properties(test-muq-mulan-oracle-pipeline PROPERTIES DEPENDS test-muq-mulan-model-parity TIMEOUT 600)
        endif()
        if(MERT_MODEL_PARITY AND MERT2_MODEL_PARITY AND MUQ_MODEL_PARITY AND MUQ_MULAN_MODEL_PARITY)
            add_test(NAME test-encoder-quantized-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/models/encoder_quantized.py"
                    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/encoder-quantized-fixture"
                    "${MERT_REFERENCE_SOURCE}" "${MERT2_REFERENCE_SOURCE}" "${MUQ_REFERENCE_SOURCE}" "${MUQ_MULAN_REFERENCE_SOURCE}")
            set_tests_properties(test-encoder-quantized-parity PROPERTIES TIMEOUT 600)
        endif()
        add_test(NAME test-clap-embedding-cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/embedding_cli.py" "$<TARGET_FILE:clap-cli>")
        set_tests_properties(test-clap-embedding-cli PROPERTIES TIMEOUT 30)
    endif()
    # Audio preprocessing helpers are private to the core and its tests.
    foreach(target IN ITEMS test-audio test-audio-path
            test-mert2-fft test-mert-model test-clamp3-bundled test-muq-model test-muq-mulan-model)
        if(TARGET ${target})
            target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        endif()
    endforeach()
endif()

# C++ test tools opt into language features, not blanket private-header access.
# Private implementation tests declare their src include directory above.
get_property(CLAP_LOCAL_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(target IN LISTS CLAP_LOCAL_TARGETS)
    get_target_property(kind ${target} TYPE)
    if(kind STREQUAL "EXECUTABLE")
        target_compile_features(${target} PRIVATE cxx_std_20)
    endif()
endforeach()

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

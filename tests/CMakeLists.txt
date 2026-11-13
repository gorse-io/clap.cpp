if(NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/googletest/CMakeLists.txt")
    message(FATAL_ERROR
        "GoogleTest submodule is missing. Run: git submodule update --init --recursive")
endif()
set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "Use the shared MSVC runtime" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/thirdparty/googletest"
    "${PROJECT_BINARY_DIR}/thirdparty/googletest" EXCLUDE_FROM_ALL)

set(CLAP_TEST_MODEL "${PROJECT_SOURCE_DIR}/models/larger_clap_general-q4_k_m.gguf")
add_executable(test-clap test-clap.cpp)
target_compile_features(test-clap PRIVATE cxx_std_20)
target_compile_definitions(test-clap PRIVATE
    CLAP_TEST_MODEL="${CLAP_TEST_MODEL}"
    CLAP_TEST_AUDIO="${PROJECT_SOURCE_DIR}/assets/dog.wav")
target_link_libraries(test-clap PRIVATE clap GTest::gtest_main httplib::httplib)
set_target_properties(test-clap PROPERTIES CXX_EXTENSIONS OFF)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(test-clap PRIVATE GTEST_LINKED_AS_SHARED_LIBRARY=1)
endif()
if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(TARGET test-clap POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:gtest>" "$<TARGET_FILE:gtest_main>" "$<TARGET_FILE_DIR:test-clap>"
        VERBATIM)
endif()

add_test(NAME test-clap COMMAND test-clap)
set_tests_properties(test-clap PROPERTIES TIMEOUT 1800)

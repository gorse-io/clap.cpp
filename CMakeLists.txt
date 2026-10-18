cmake_minimum_required(VERSION 3.20)

project(clap_cpp VERSION 0.1.0 LANGUAGES CXX)

add_executable(clap src/main.cpp)
target_compile_features(clap PRIVATE cxx_std_17)
set_target_properties(clap PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(clap PRIVATE /W4)
else()
    target_compile_options(clap PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME clap_smoke COMMAND clap)
    set_tests_properties(clap_smoke PROPERTIES PASS_REGULAR_EXPRESSION "clap.cpp")
endif()

install(TARGETS clap RUNTIME DESTINATION bin)

cmake_minimum_required(VERSION 3.20)
# Source is ISO C99. Enable CXX only for the static library's implementation
# runtime and transitive ggml dependencies; no C++ headers/features reach main.c.
project(clap_installed_c_consumer LANGUAGES C CXX)
find_package(clap CONFIG REQUIRED)
add_executable(clap-installed-c ../c-api-test.c)
set_target_properties(clap-installed-c PROPERTIES
    C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO LINKER_LANGUAGE CXX)
if(MSVC)
    target_compile_options(clap-installed-c PRIVATE /W4 /WX)
else()
    target_compile_options(clap-installed-c PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
target_link_libraries(clap-installed-c PRIVATE clap::clap)
get_target_property(features clap::clap INTERFACE_COMPILE_FEATURES)
if(features MATCHES "cxx_")
    message(FATAL_ERROR "Public C target leaks C++ compile features: ${features}")
endif()

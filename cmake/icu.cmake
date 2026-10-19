include_guard(GLOBAL)

option(CLAP_USE_SYSTEM_ICU "Use system ICU instead of the bundled static Unicode/regex build" OFF)

if(CLAP_USE_SYSTEM_ICU)
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    return()
endif()

# Pinned by the thirdparty/icu gitlink (ICU 78.3). ICU's supported uconfig.h
# feature switches allow a Unicode/regex-only build without Autotools, MSBuild
# projects, generated locale packages, or host-side data-generation tools.
set(CLAP_ICU_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../thirdparty/icu/icu4c/source")
if(NOT EXISTS "${CLAP_ICU_SOURCE}/common/unicode/uvernum.h")
    message(FATAL_ERROR "ICU submodule is missing. Run: git submodule update --init --recursive")
endif()
message(STATUS "Using bundled ICU 78.3 (static Unicode/regex-only build)")

add_library(clap-icu-options INTERFACE)
target_compile_features(clap-icu-options INTERFACE cxx_std_17)
target_include_directories(clap-icu-options SYSTEM INTERFACE
    "${CLAP_ICU_SOURCE}/common" "${CLAP_ICU_SOURCE}/i18n")
target_compile_definitions(clap-icu-options INTERFACE
    U_STATIC_IMPLEMENTATION
    U_CHARSET_IS_UTF8=1
    UCONFIG_NO_CONVERSION=1
    UCONFIG_NO_NORMALIZATION=1
    UCONFIG_NO_FORMATTING=1
    UCONFIG_NO_SERVICE=1
    UCONFIG_NO_FILE_IO=1
    U_ENABLE_DYLOAD=0)

# Unicode character/case/property tables are compiled into common's generated
# *_data.h files in upstream source. The model's regex does not require locale,
# collation, normalization or converter data. Stubdata satisfies the ICU data
# symbol without adding a runtime .dat file or data DLL dependency.
add_library(clap-icu-data STATIC "${CLAP_ICU_SOURCE}/stubdata/stubdata.cpp")
target_link_libraries(clap-icu-data PUBLIC clap-icu-options)

file(GLOB CLAP_ICU_COMMON CONFIGURE_DEPENDS "${CLAP_ICU_SOURCE}/common/*.cpp")
set(CLAP_ICU_I18N
    "${CLAP_ICU_SOURCE}/i18n/regexcmp.cpp"
    "${CLAP_ICU_SOURCE}/i18n/regeximp.cpp"
    "${CLAP_ICU_SOURCE}/i18n/regexst.cpp"
    "${CLAP_ICU_SOURCE}/i18n/regextxt.cpp"
    "${CLAP_ICU_SOURCE}/i18n/rematch.cpp"
    "${CLAP_ICU_SOURCE}/i18n/repattrn.cpp"
    "${CLAP_ICU_SOURCE}/i18n/uregex.cpp"
    "${CLAP_ICU_SOURCE}/i18n/uregexc.cpp"
    "${CLAP_ICU_SOURCE}/i18n/ucln_in.cpp")
add_library(clap-icu-uc STATIC ${CLAP_ICU_COMMON})
add_library(clap-icu-i18n STATIC ${CLAP_ICU_I18N})
find_package(Threads REQUIRED)
target_link_libraries(clap-icu-uc PUBLIC clap-icu-options clap-icu-data Threads::Threads)
target_link_libraries(clap-icu-i18n PUBLIC clap-icu-uc)
target_compile_definitions(clap-icu-uc PRIVATE U_COMMON_IMPLEMENTATION)
target_compile_definitions(clap-icu-i18n PRIVATE U_I18N_IMPLEMENTATION)
foreach(target IN ITEMS clap-icu-data clap-icu-uc clap-icu-i18n)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /bigobj)
    endif()
endforeach()

add_library(ICU::data ALIAS clap-icu-data)
add_library(ICU::uc ALIAS clap-icu-uc)
add_library(ICU::i18n ALIAS clap-icu-i18n)

install(FILES "${CLAP_ICU_SOURCE}/../../LICENSE" DESTINATION share/licenses/clap/icu)

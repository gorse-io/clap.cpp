include(CMakePackageConfigHelpers)
set(CLAP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/clap")
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clap-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/clap-config.cmake"
    INSTALL_DESTINATION "${CLAP_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/clap-config-version.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
if(NOT BUILD_SHARED_LIBS)
    install(EXPORT clap-targets NAMESPACE clap:: DESTINATION "${CLAP_INSTALL_CMAKEDIR}")
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/clap-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/clap-config-version.cmake"
    DESTINATION "${CLAP_INSTALL_CMAKEDIR}")

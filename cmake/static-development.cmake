include_guard(GLOBAL)

# Collect the actual configured archives, not a backend-specific name list.
# Call before test/tool targets are added so only the core/dependency graph is
# included. Imported SDK libraries remain external and are resolved by File API.
function(clap_collect_static_archives directory output)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    set(archives "")
    foreach(target IN LISTS targets)
        get_target_property(kind ${target} TYPE)
        if(kind STREQUAL "STATIC_LIBRARY")
            list(APPEND archives ${target})
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        clap_collect_static_archives("${child}" child_archives)
        list(APPEND archives ${child_archives})
    endforeach()
    set(${output} "${archives}" PARENT_SCOPE)
endfunction()

function(clap_static_development)
    get_target_property(kind clap TYPE)
    if(NOT kind STREQUAL "STATIC_LIBRARY")
        if(CLAP_GENERATE_LINK_METADATA)
            message(FATAL_ERROR "CLAP_GENERATE_LINK_METADATA requires a static clap library")
        endif()
        return()
    endif()
    clap_collect_static_archives("${CMAKE_CURRENT_SOURCE_DIR}" archives)
    # clap's own archive is already installed by src/CMakeLists.txt.
    set(dependencies ${archives})
    list(REMOVE_ITEM dependencies clap)
    if(dependencies)
        install(TARGETS ${dependencies}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT development)
    endif()
    if(CLAP_GENERATE_LINK_METADATA)
        # Never built or installed. CMake resolves the same target graph a real
        # C++ consumer would link, including imported GPU SDK dependencies.
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/clap-link-probe.cpp"
            "int main() { return 0; }\n")
        add_executable(clap-link-probe EXCLUDE_FROM_ALL
            "${CMAKE_CURRENT_BINARY_DIR}/clap-link-probe.cpp")
        target_link_libraries(clap-link-probe PRIVATE clap)
        set(archive_names "")
        foreach(archive IN LISTS archives)
            list(APPEND archive_names "$<TARGET_FILE_BASE_NAME:${archive}>")
        endforeach()
        file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/clap-link-config.cmake" CONTENT
            "set(CLAP_LINK_ARCHIVES \"${archive_names}\")\nset(CLAP_LINK_WINDOWS \"${WIN32}\")\nset(CLAP_LINK_APPLE \"${APPLE}\")\n")
    endif()
endfunction()

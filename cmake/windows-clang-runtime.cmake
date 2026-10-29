# CMake's Clang/MSVC static-CRT spelling uses -Xclang
# -flto-visibility-public-std, which sccache 0.18 cannot parse. Let the Clang
# driver preserve the required visibility option, _MT macro and libcmt dependency.
# Keep CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded (not the DLL CRT).
foreach(lang C CXX)
    set(CMAKE_${lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded
        -fms-runtime-lib=static)
endforeach()

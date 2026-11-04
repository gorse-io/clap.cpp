# Standalone core runtime

The root project builds the CLI and tests by default. To build and install only
its shared C API runtime, with bundled ggml and ICU linked as static PIC libraries:

```sh
cmake -S . -B build-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLAP_BUILD_CLI=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON -DCLAP_STATIC_DEPENDENCIES=ON
cmake --build build-core --parallel 4
cmake --install build-core --prefix /path/to/prefix --component runtime
```

This configuration does not need CLI11, cpp-httplib, Python, FFmpeg, or the CLI
and test sources. It requires the root CMakeLists.txt, cmake/, src/, include/,
ggml/, and thirdparty/{json,miniaudio,icu}/ sources. It uses the same core target
as the default build, without a separate packaging project.

`CLAP_STATIC_DEPENDENCIES` changes bundled dependency library types, not backend
selection. Set normal `GGML_*` options for the desired backends, OpenMP, and CPU
instruction baseline. Dynamically loaded backends (`GGML_BACKEND_DL`) require
shared ggml libraries and cannot be used with static bundled dependencies.
`CLAP_USE_SYSTEM_ICU` retains its normal behavior and is not made static by this
option; leave it off for a self-contained bundled ICU runtime.

GNUInstallDirs controls the install layout:

- Shared core libraries go to `CMAKE_INSTALL_LIBDIR` (normally `lib` or `lib64`).
- The core DLL on Windows goes to `CMAKE_INSTALL_BINDIR` (normally `bin`).
- Bundled dependency license notices go to `<CMAKE_INSTALL_LIBDIR>/licenses/`.
- These files use component `runtime`; static/import archives and `clap.h` use
  component `development` and are not included in a runtime-only install.

For a namespaced distribution, set both `CMAKE_INSTALL_LIBDIR` and
`CMAKE_INSTALL_BINDIR` to the desired relative namespace. A normal install without
`--component` also retains the CLI executable name `clap` when it is enabled.
A runtime-only install with shared ggml dependencies does not bundle those
libraries; use static bundled dependencies or arrange their runtime deployment
separately. Vendor GPU libraries and enabled system dependencies also remain the
consumer's deployment responsibility.

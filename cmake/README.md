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
- Bundled dependency license notices go to `<CMAKE_INSTALL_LIBDIR>/licenses/`
  in both `runtime` and `development` installs.
- Shared runtime files use component `runtime`; static/import archives and
  `clap.h` use component `development` and are not in a runtime-only install.
- A static core's `development` component also installs all configured bundled
  static dependency archives (ggml backends and bundled ICU included). Imported
  SDK/system libraries remain external; a shared core does not install its
  private bundled static archives.

For a namespaced distribution, set both `CMAKE_INSTALL_LIBDIR` and
`CMAKE_INSTALL_BINDIR` to the desired relative namespace. A normal install without
`--component` also retains the CLI executable name `clap` when it is enabled.
A runtime-only install with shared ggml dependencies does not bundle those
libraries; use static bundled dependencies or arrange their runtime deployment
separately. Vendor GPU libraries and enabled system dependencies also remain the
consumer's deployment responsibility.

## Static consumers and link metadata

A static consumer uses the same root graph, not a binding-specific CMake project:

```sh
cmake -S . -B build-static \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLAP_BUILD_CLI=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCLAP_STATIC_DEPENDENCIES=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build-static --target clap --parallel 2
cmake --install build-static --prefix /path/to/prefix --component development
```

`static-development.cmake` collects configured, non-imported static targets before
CLI and test targets are created and installs their archive closure. Backend
selection stays in the ordinary `GGML_*` options. Static archives do not imply
static GPU SDK runtimes: keep `GGML_STATIC=OFF` when the SDK should choose its
normal linkage (HIP requires this).

For tooling that needs resolved native link requirements, request the CMake File
API `codemodel-v2` before configuring and set `CLAP_GENERATE_LINK_METADATA=ON`.
This adds only an `EXCLUDE_FROM_ALL` executable `clap-link-probe` linked to `clap`;
it is never built or installed. Its File API link fragments contain CMake's
ordered transitive closure, including imported SDK targets and framework paths.
`clap-link-config.cmake` records platform and configured archive names for the
Cargo adapter. After generation, Rust invokes:

```sh
cmake -DCLAP_LINK_BUILD=/absolute/build-static \
  -DCLAP_LINK_MANIFEST=/absolute/cargo-link.txt -P cmake/cargo-link.cmake
```

The adapter emits ordered `cargo:rustc-link-lib` / `cargo:rustc-link-search`
directives, uses installed archives for bundled targets, and preserves resolved
external SDK paths, versioned shared filenames, Windows import-library prefixes,
and Apple frameworks. It selects Release in multi-configuration generators.
Build-tree rpaths and compiler-driver flags are intentionally not exported;
shared SDK deployment remains the application's responsibility. `build.rs`
selects the portable CPU baseline, optional backends, install prefix, and isolated
build directory using root options; no Rust-specific native target graph exists.

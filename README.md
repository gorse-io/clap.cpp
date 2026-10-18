# clap.cpp

Minimal C++17/CMake project scaffold. Model loading and inference are not implemented yet.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler

## Clone

```sh
git clone --recurse-submodules https://github.com/gorse-io/clap.cpp.git
cd clap.cpp
```

For an existing checkout:

```sh
git submodule update --init --recursive
```

The `ggml/` submodule uses upstream `ggml-org/ggml`, pinned to a commit.
Like stable-diffusion.cpp, it is built with `add_subdirectory` and linked via
its `ggml` CMake target; no system ggml installation is required.

CLI11 is included as a submodule under `thirdparty/CLI11`, pinned to v2.7.2.
The `clap` target links `CLI11::CLI11` so the command-line entry point can use
its headers without a system installation.

cpp-httplib is included as a submodule under `thirdparty/cpp-httplib`, pinned to
v0.61.0. The `clap` target links `httplib::httplib` for HTTP server/client support.
The `serve` subcommand remains a placeholder that prints `hello world`; it does
not start an HTTP server yet.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Run the `serve` subcommand to print `hello world`:

```sh
./build/clap serve
```

Running `clap` without a subcommand still prints `clap.cpp`.
The executable is located at `build/clap` for single-configuration generators or
`build/Release/clap.exe` for Visual Studio Release builds.

To skip tests, configure with `-DBUILD_TESTING=OFF`.
Tests include a ggml context/tensor smoke test to verify headers and linking.
Dependencies are static by default. `GGML_NATIVE` defaults to `OFF` for portable
builds; configure with `-DGGML_NATIVE=ON` to optimize for the build machine.
Other backend options can be passed directly, for example `-DGGML_CUDA=ON`
(requires the CUDA toolkit).

## Install

```sh
cmake --install build --config Release --prefix ./install
```

The executable is installed under `bin/` in the selected prefix.

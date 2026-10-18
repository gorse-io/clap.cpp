# clap.cpp

Minimal C++17/CMake project scaffold. Model loading and inference are not implemented yet.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

The `clap` executable currently prints `clap.cpp` as a smoke test.
It is located at `build/clap` for single-configuration generators or
`build/Release/clap.exe` for Visual Studio Release builds.

To skip tests, configure with `-DBUILD_TESTING=OFF`.

## Install

```sh
cmake --install build --config Release --prefix ./install
```

The executable is installed under `bin/` in the selected prefix.

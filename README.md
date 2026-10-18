# clap.cpp

Minimal C++17/CMake project scaffold. Model loading and inference are not implemented yet.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- Python 3 (only when building tests)

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
nlohmann/json is included as a submodule under `thirdparty/json`, pinned to
v3.12.0. Request parsing and response serialization use `nlohmann_json::nlohmann_json`.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

## Embeddings server

Start the server (defaults to `127.0.0.1:8080`):

```sh
./build/clap serve --host 127.0.0.1 --port 8080
```

Send a request following the llama.cpp `/v1/embeddings` JSON response shape:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"dummy","input":["hello","world"],"encoding_format":"float"}'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
    {"object": "embedding", "index": 1, "embedding": [1.0, 0.0, 0.0]}
  ],
  "model": "dummy",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

This is a stub: every input gets the same fixed 3-dimensional vector; no model
is loaded and no tokenization or inference takes place. Usage counts are zero.
`input` accepts one non-empty string or an array of 1–1024 non-empty strings.
`model` is optional (defaults to `dummy`) and is echoed in the response.
Only `encoding_format: "float"` (the default) and `dimensions: 3` are supported;
token IDs, multimodal inputs and base64 encoding are not implemented.
Invalid JSON or unsupported parameter values return HTTP 400 with a JSON
`error` object. Request bodies are limited to 1 MiB.

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

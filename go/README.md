# clap.cpp Go bindings

Requires Go 1.22+, CGO and the clap shared C library. Building the native library
requires CMake 3.20+, Ninja and a C++20 compiler.

## Build

From a recursive checkout of the repository, build the shared runtime used by
other bindings:

```sh
cmake --preset libraries
cmake --build build/libraries --target clap --parallel 4
export CGO_LDFLAGS="-L$PWD/build/libraries/src"
export LD_LIBRARY_PATH="$PWD/build/libraries/src"   # Linux
export DYLD_LIBRARY_PATH="$PWD/build/libraries/src" # macOS
```

For an installed runtime, set `CGO_CFLAGS` and `CGO_LDFLAGS` to its include and
library directories, and configure the matching runtime loader path.

## Usage

```go
import "github.com/gorse-io/clap.cpp/go"

model, err := clap.New("larger_clap_general-q4_k_m.gguf", 0)
if err != nil {
    return err
}
defer model.Close()
textEmbedding, err := model.EncodeText("Sound of a dog")
if err != nil {
    return err
}
audioEmbedding, err := model.EncodeAudio("dog.wav", 0)
if err != nil {
    return err
}
```

## Test

```sh
cd go
go test ./...
```

`TestCLAP` downloads the model on first use and checks audio/text similarity
against `assets/dog.wav`, matching the C++ integration test.

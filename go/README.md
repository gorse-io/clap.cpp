# Go bindings (CGO)

Module: `github.com/gorse-io/clap.cpp/go`, package `clap`. Requires Go 1.22+,
CGO enabled, a C/C++17 toolchain, CMake 3.20+ and pkg-config. Linux and macOS
are the initial targets; Windows is not supported by this build entrypoint.
The default is portable CPU-only: no GPU SDK, OpenMP, BLAS or system ICU needed.

## Build the native library

From a recursive checkout of this repository (not a Go module-cache directory):

```sh
git submodule update --init --recursive
# Use an absolute prefix: pkg-config records it at configure time.
PREFIX="$PWD/go/build/install"
cmake -S go/native -B go/build/native \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build go/build/native --parallel 4
cmake --install go/build/native --component go
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
# Linux:
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# macOS (instead of LD_LIBRARY_PATH):
# export DYLD_LIBRARY_PATH="$PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Install with the same prefix used at configuration (do not override it with
`cmake --install --prefix`). This builds a shared `clap-go` bridge containing
static PIC clap/ggml/bundled ICU. Only that bridge and platform runtime libraries
are needed at execution; no sibling Rust, Python or Node build is used. Deploy
the installed library with your application and configure its loader path (or
install it in a system loader directory). `PKG_CONFIG_PATH` is needed at Go
**build** time; the loader variable is needed at **run** time too.

Until a revision containing this module is published, use a local replacement
from your consumer module:

```sh
go mod edit -require=github.com/gorse-io/clap.cpp/go@v0.0.0
go mod edit -replace=github.com/gorse-io/clap.cpp/go=/absolute/path/to/clap.cpp/go
go mod tidy
```

The native build requires repository sources/submodules. A downloaded Go module
alone does not supply them or automatically build the bridge; install it from a
matching recursive repository revision first.

## API

```go
import clap "github.com/gorse-io/clap.cpp/go"

model, err := clap.New("model.gguf", 0) // 0 = native default thread count
if err != nil { return err }
defer model.Close()
text, err := model.EncodeText("A jazz quartet.")
if err != nil { return err }
audio, err := model.EncodeAudio("music.flac", 0)
if err != nil { return err }
_ = text
_ = audio
```

Each call embeds one item; results are Go-owned `[]float32`. Audio decoding,
multichannel downmix and model-rate resampling happen internally. `cropOffset`
is a sample offset at the **model's** rate, not the input rate or byte offset;
CLAP uses it to select a ten-second crop. Other architectures may ignore or
reject nonzero offsets. Model-specific modality restrictions propagate as errors.

Calls on the same model, including `Close`, are serialized. Copies of `CLAP`
share the same lock and native lifetime: closing any copy closes them all.
`Close` is idempotent, including on nil/zero values; subsequent inference returns
`ErrClosed` (`errors.Is` works). Explicitly close models; the GC finalizer is a
fallback, not deterministic resource management. Paths/text must be valid UTF-8
without embedded NUL. Negative or overflowing thread counts and crop offsets
outside native signed-index range are rejected. Native exceptions become errors;
this does not sandbox untrusted GGUF files or recover native process aborts.

## Tests and runnable example

After native installation and environment setup, from the repository root:

```sh
python3 go/tests/generate.py
cd go
go test -race -count=1 ./...
go vet ./...
go run ./examples/embed build/fixtures/模型.gguf Hi build/fixtures/音频.wav
```

Python 3 is needed only to generate test fixtures, using the repository's
stdlib-only tiny GGUF generator and independent scalar text oracle. Tests run
real text/audio inference (including MERT), Unicode paths/text, 24 kHz stereo
downmix/resampling, deterministic nonzero cropping, invalid inputs, ownership
and concurrent close/inference. Generated models/audio/native builds remain in
ignored `go/build`; no model downloads are required. The Go race detector checks
Go synchronization, not unsanitized C++ internals. CI also builds an external
consumer module against the installed bridge.

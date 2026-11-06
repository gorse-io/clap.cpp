# Java bindings

Java 11+ bindings to the public `include/clap.h` C API using JNA. There is no
project-specific JNI library or Java native compilation step. The Maven JAR
contains Java classes only; build and distribute the clap.cpp shared library
separately. JNA is a normal transitive Maven dependency (not shaded into the JAR).

## Prerequisites and build

- JDK 11 or newer and Maven 3.6.3+.
- For the native library: CMake 3.20+, a C++20 compiler, and the repository's
  initialized submodules. See the root build documentation for platform details.
- For native integration tests only: Python 3.8+ (standard library only) and a
  complete repository checkout containing `python/tests/tiny_models.py`.

From the repository root, build the native shared library independently:

```sh
git submodule update --init --recursive
cmake -S . -B build/java-native \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCLAP_STATIC_DEPENDENCIES=ON -DCLAP_BUILD_CLI=OFF -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_LIBDIR=. -DCMAKE_INSTALL_BINDIR=.
cmake --build build/java-native --config Release --target clap -j2
cmake --install build/java-native --config Release --component runtime \
  --prefix "$PWD/build/java-native/runtime"
```

`CLAP_STATIC_DEPENDENCIES=ON` embeds bundled dependencies as static PIC libraries
in the shared clap core. Without it, their runtime shared libraries must also be
available to the OS loader. For multi-configuration generators select
`--config Release`; Windows outputs a DLL and macOS a dylib instead of `.so`.

Build/install the Java artifact from `java/`:

```sh
mvn verify       # unit tests; native integration tests explicitly skipped
mvn install      # install io.gorse:clap:0.1.0-SNAPSHOT in your local Maven repository
```

Both the ordinary and `-sources.jar` artifacts are produced at package time.
The version above is a local development coordinate, not a claim of publication
to Maven Central. Consumers can use:

```xml
<dependency>
  <groupId>io.gorse</groupId>
  <artifactId>clap</artifactId>
  <version>0.1.0-SNAPSHOT</version>
</dependency>
```

## Native loading

Before constructing the first model, set `clap.library.path` to an **absolute
shared-library filename**, not a directory, static archive, executable, or JNI
shim. Use the clap library built from the same source revision as the bindings.
For example:

```sh
java -Dclap.library.path=/absolute/path/to/libclap.so -cp 'your-application-classpath' YourApplication
```

Windows uses an absolute `clap.dll` path; macOS uses `libclap.dylib`.
Dependencies of the library still follow the operating system's loading rules.
The Java process and native library must have matching architectures. On JDKs
that require native-access permission, add `--enable-native-access=ALL-UNNAMED`
for classpath applications.

When the property is absent, the binding uses JNA's `Native.load("clap", ...)`
search (for example, `-Djna.library.path=/absolute/library/directory`). Explicit
`clap.library.path` is recommended for reproducible deployment. Loading is lazy
and cached once per classloader: configure it before the first constructor;
changing the property afterwards does not reload the library. Missing native
libraries or dependencies produce JNA `UnsatisfiedLinkError` diagnostics.

## Usage

```java
import io.gorse.clap.CLAP;
import java.nio.file.Path;

CLAP.Options options = new CLAP.Options(
    2, "cpu", CLAP.ActivationPrecision.F32);
try (CLAP model = new CLAP(Path.of("models/clap.gguf"), options)) {
    float[] text = model.encodeText("A dog barking 🐕");
    float[] audio = model.encodeAudio(Path.of("recording.wav"));
    float[] cropped = model.encodeAudio(Path.of("recording.wav"), 0L);

    float[] mono = new float[24000]; // Replace with your mono PCM samples.
    float[] resampled = model.encodeAudio(mono, 24000);
    float[] withCrop = model.encodeAudio(mono, 24000, 0L);
    float[] nativeRatePcm = new float[model.audioSampleRate()];
    float[] atModelRate = model.encodeAudio(nativeRatePcm);
    int dimension = model.projectionDimension();
}
```

`new CLAP(path)` uses `new CLAP.Options()` (automatic threads, native device
selection, model-default activation precision). `threads` must be nonnegative;
zero selects the native default. A null/empty `device` honors native `CLAP_DEVICE`
or auto-selection; `"cpu"` forces CPU. Activation precision is `MODEL_DEFAULT` or
`F32`. Options are immutable and may be reused between models.

Text, paths and device names use UTF-8 regardless of the JVM default encoding.
Embedded NUL and malformed UTF-16 strings are rejected. PCM must be nonempty,
finite, mono `float[]`, normally in `[-1, 1]`, with a positive input sample rate.
Inputs outside that amplitude range are not artificially clipped by Java.
Do not mutate a PCM array while an inference call uses it. WAV/MP3/FLAC files are
decoded by contents in native code; no Java audio decoder is involved.

`cropOffset` is a nonnegative `long` measured in **model-rate samples**, even when
input PCM is resampled. On 32-bit platforms it must fit unsigned `size_t`.
Non-CLAP architectures may reject nonzero crop offsets. Unsupported modalities,
invalid model files, decoder errors and other native failures throw
`IllegalStateException` with the operation, status and immediately copied native
diagnostic. Java input validation throws `IllegalArgumentException` or
`NullPointerException`; every operation after close throws `IllegalStateException`.

The Java API intentionally exposes only text/audio embeddings and the model's
audio rate/projection dimension, not low-level tokens or audio frames. Output
normalization follows each model's native semantics; not all architectures
produce unit vectors. Each returned array is a fresh Java-owned copy that remains
valid after closing the model; native output buffers are freed in `finally`.

## Lifetime and threading

Always use try-with-resources or explicit `close()`. Close is idempotent. A
`Cleaner` is only a fallback for unreachable models; its action holds the native
handle and library, never the owning Java object. Every instance operation is
synchronized, so inference, property reads and close on the same model cannot
race. Separate instances may run concurrently. Account for native thread counts
when running multiple instances. The library remains loaded for the classloader
lifetime; this wrapper does not unload it while models may be alive.

## Tests

From `java/`, after the separate native build above:

```sh
mvn -Dclap.library.path="$(pwd)/../build/java-native/runtime/libclap.so" verify
python3 tests/verify_package.py \
  --library "$(pwd)/../build/java-native/runtime/libclap.so" \
  --model "target/fixtures/tiny-clap.gguf"
```

For CI from the repository root, explicitly generate the reusable fixtures first
(the JUnit suite also generates them automatically when absent):

```sh
python3 java/tests/generate_fixtures.py
mvn -B -f java/pom.xml -Dclap.library.path=/absolute/path/to/shared-library verify
python3 java/tests/verify_package.py --library /absolute/path/to/shared-library \
  --model java/target/fixtures/tiny-clap.gguf
```

Setting `clap.library.path` enables the native tests; no additional opt-in property
is required. Without it the native tests are explicitly skipped. A configured
but missing/broken native library is a hard failure. Fixtures default to
`java/target/fixtures` regardless of where the generator is launched. An optional
positional argument selects a different generator output directory; the JUnit
suite itself always reads `java/target/fixtures`.

The package check compiles independent consumers against the binary JAR and the
recompiled source JAR using `javac --release 11`, then runs both on the real
library when `--library`/`--model` are supplied. It does not access the network;
JNA must already be in the local Maven repository (or passed with `--jna`).
Omit those two native arguments for a compile-only offline packaging check.

Set `-Dclap.test.python=/absolute/path/to/python3` if necessary. Opting in makes
missing/broken native libraries and fixture generation failures hard failures,
not skips. Tests generate tiny valid CLAP/MERT GGUF and WAV files under
`java/target/` using the shared stdlib Python fixture generator (no downloads,
Torch, NumPy, or committed model/audio blobs). They cover an independent scalar
text oracle, Unicode strings/paths, real PCM/file equivalence, resampling,
validation and native diagnostics, output ownership, idempotent close, and
concurrent inference/close. These tiny fixtures test the ABI and inference
contract; they are not a replacement for published-checkpoint accuracy tests.

# clap.cpp Java bindings

Requires Java 11+ and the clap.cpp shared library. Uses JNA; native binaries are
not bundled in the JAR.

## Install

From the repository root, install the development snapshot locally:

```sh
mvn -f java/pom.xml -DskipTests install
```

Add the Maven dependency (not yet published to Maven Central):

```xml
<dependency>
  <groupId>io.gorse</groupId>
  <artifactId>clap-cpp</artifactId>
  <version>0.1.0-SNAPSHOT</version>
</dependency>
```

Set `-Dclap.library.path=/absolute/path/to/libclap.so` when running Java.
Use `libclap.dylib` on macOS or `clap.dll` on Windows.

## Usage

```java
import io.gorse.clap.CLAP;
import java.nio.file.Path;

try (CLAP model = new CLAP(Path.of("larger_clap_general-q4_k_m.gguf"))) {
    float[] textEmbedding = model.encodeText("Sound of a dog");
    float[] audioEmbedding = model.encodeAudio(Path.of("dog.wav"));
}
```

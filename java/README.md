# clap.cpp Java bindings

Requires Java 11+. The JAR bundles staged native libraries; JNA loads the matching
library automatically.

## Install

Stage native runtimes under `libraries/<platform>/clap_cpp` using the shared
native build, then install the development snapshot from the repository root:

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

The JAR includes whichever Linux x64/ARM64, macOS ARM64, and Windows x64/ARM64
runtimes are staged when packaging. No library path is needed for bundled
platforms; `-Dclap.library.path=/absolute/path/to/library` overrides loading.

## Usage

```java
import io.gorse.clap.CLAP;
import java.nio.file.Path;

try (CLAP model = new CLAP(Path.of("larger_clap_general-q4_k_m.gguf"))) {
    float[] textEmbedding = model.encodeText("Sound of a dog");
    float[] audioEmbedding = model.encodeAudio(Path.of("dog.wav"));
}
```

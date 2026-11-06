"""Offline JAR consumer/source-JAR check after `mvn package` (no downloads)."""
import argparse
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", type=Path, help="also execute both consumers with this absolute native library")
parser.add_argument("--model", type=Path, help="valid CLAP GGUF for native execution")
parser.add_argument("--jna", type=Path, help="JNA JAR (default: Maven local repository)")
args = parser.parse_args()
if bool(args.library) != bool(args.model):
    parser.error("--library and --model must be specified together")
root = Path(__file__).resolve().parents[1]
ns = {"m": "http://maven.apache.org/POM/4.0.0"}
pom = ET.parse(root / "pom.xml")
version = pom.findtext("m:version", namespaces=ns)
jna_version = pom.findtext("m:properties/m:jna.version", namespaces=ns)
jna = args.jna or Path.home() / f".m2/repository/net/java/dev/jna/jna/{jna_version}/jna-{jna_version}.jar"
assert jna.is_file(), f"JNA not cached: {jna}; run Maven once first or pass --jna"
jar = root / f"target/clap-{version}.jar"
sources = root / f"target/clap-{version}-sources.jar"
assert jar.is_file() and sources.is_file(), "Run mvn package first"
work = root / "target/package-check"
work.mkdir(parents=True, exist_ok=True)
consumer = work / "Consumer.java"
consumer.write_text('''import io.gorse.clap.CLAP;
import java.nio.file.Path;
public final class Consumer {
    public static void main(String[] args) {
        CLAP.Options options = new CLAP.Options(1, "cpu", CLAP.ActivationPrecision.F32);
        try (CLAP model = new CLAP(Path.of(args[0]), options)) {
            float[] result = model.encodeText("Hello, 世界 🎵");
            if (result.length != model.projectionDimension()) throw new AssertionError("dimension");
            for (float value : result) if (!Float.isFinite(value)) throw new AssertionError("nonfinite");
            System.out.println("consumer embedding dimension=" + result.length);
        }
    }
}
''', encoding="utf-8")
for kind in ("binary", "sources"):
    classes = work / f"{kind}-classes"
    classes.mkdir(exist_ok=True)
    inputs = [str(consumer)]
    classpath = str(jna)
    if kind == "binary":
        classpath += os.pathsep + str(jar)
    else:
        source_dir = work / "source-content"
        with zipfile.ZipFile(sources) as archive:
            names = [name for name in archive.namelist() if name.endswith(".java")]
            assert set(names) == {"io/gorse/clap/CLAP.java", "io/gorse/clap/NativeBindings.java"}, names
            for name in names:
                destination = source_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                inputs.append(str(destination))
    subprocess.run(["javac", "--release", "11", "-encoding", "UTF-8", "-cp", classpath,
                    "-d", str(classes), *inputs], check=True)
    print(f"{kind} consumer compiled offline for Java 11", flush=True)
    if args.library:
        assert args.library.is_absolute(), "--library must be absolute"
        subprocess.run(["java", f"-Dclap.library.path={args.library}", "-cp",
                        str(classes) + os.pathsep + classpath, "Consumer", str(args.model.resolve())], check=True)
        print(f"{kind} consumer native execution passed", flush=True)

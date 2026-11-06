package io.gorse.clap;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class LibraryLoadingTest {
    @Test void explicitLibraryMustBeExistingAbsoluteFilename() throws Exception {
        for (String location : new String[] {"relative-clap.so", "", Path.of("target", "absent-library.so").toAbsolutePath().toString(),
                Path.of("target").toAbsolutePath().toString()}) {
            Process child = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-Dclap.library.path=" + location, "-cp", System.getProperty("java.class.path"),
                    Probe.class.getName()).redirectErrorStream(true).start();
            assertTrue(child.waitFor(30, TimeUnit.SECONDS), "loader subprocess timed out");
            String output = new String(child.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(0, child.exitValue(), output);
            assertTrue(output.contains("invalid library path rejected"), output);
        }
    }

    public static final class Probe {
        public static void main(String[] args) {
            try {
                new CLAP(Path.of("unused-model.gguf"));
                throw new AssertionError("invalid library path accepted");
            } catch (IllegalArgumentException expected) {
                if (!expected.getMessage().contains("clap.library.path")) throw expected;
                System.out.println("invalid library path rejected");
            }
        }
    }
}

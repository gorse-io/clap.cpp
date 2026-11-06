package io.gorse.clap;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** A configured native library enables integration tests; a broken one is a failure. */
@EnabledIfSystemProperty(named = "clap.library.path", matches = ".*")
class NativeIntegrationTest {
    static Path fixtures;
    static final CLAP.Options OPTIONS = new CLAP.Options(1, "cpu", CLAP.ActivationPrecision.F32);

    @BeforeAll static void generate() throws Exception {
        fixtures = Path.of("target", "fixtures").toAbsolutePath();
        if (Files.isRegularFile(fixtures.resolve("tiny-clap.gguf"))
                && Files.isRegularFile(fixtures.resolve("tiny-mert.gguf"))
                && Files.isRegularFile(fixtures.resolve("audio 世界.wav"))
                && Files.isRegularFile(fixtures.resolve("text-reference.txt"))
                && Files.isRegularFile(fixtures.resolve("bad.wav"))) return;
        Process process = new ProcessBuilder(System.getProperty("clap.test.python", "python3"),
                "tests/generate_fixtures.py", fixtures.toString()).inheritIO().start();
        assertTrue(process.waitFor(60, TimeUnit.SECONDS), "fixture generator timed out");
        assertEquals(0, process.exitValue(), "fixture generator failed");
    }

    static CLAP clap() { return new CLAP(fixtures.resolve("tiny-clap.gguf"), OPTIONS); }
    static CLAP mert() { return new CLAP(fixtures.resolve("tiny-mert.gguf"), OPTIONS); }
    static float[] pcm() {
        float[] values = new float[24000];
        for (int i = 0; i < values.length; i++) values[i] = ((i % 97) - 48) / 256.0f;
        return values;
    }
    static void finite(float[] values, int dimension) {
        assertEquals(dimension, values.length);
        for (float value : values) assertTrue(Float.isFinite(value));
    }

    @Test void defaultsAndUnicodeModelPathWork() throws Exception {
        Path unicodeModel = fixtures.resolve("model 世界.gguf");
        Files.copy(fixtures.resolve("tiny-clap.gguf"), unicodeModel,
                java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        try (CLAP model = new CLAP(unicodeModel)) {
            finite(model.encodeText(""), 3);
            finite(model.encodeText("世界 🎵"), 3);
        }
    }

    @Test void actualUnicodeTextMatchesIndependentOracle() throws Exception {
        List<String> lines = Files.readAllLines(fixtures.resolve("text-reference.txt"));
        float[] reference = new float[lines.size()];
        for (int i = 0; i < reference.length; i++) reference[i] = Float.parseFloat(lines.get(i));
        float[] embedding;
        try (CLAP model = clap()) {
            assertEquals(48000, model.audioSampleRate());
            assertEquals(3, model.projectionDimension());
            embedding = model.encodeText("Hello, 世界 🎵");
            finite(embedding, 3);
            assertArrayEquals(reference, embedding, 3e-5f);
            assertArrayEquals(embedding, model.encodeText("Hello, 世界 🎵"), 1e-6f);
            embedding[0] = 123;
            assertArrayEquals(reference, model.encodeText("Hello, 世界 🎵"), 3e-5f);
        }
        assertEquals(123, embedding[0]); // The returned Java array is independently owned.
    }

    @Test void audioPcmAndUnicodeFileAreEquivalentForBothModels() {
        for (CLAP model : new CLAP[] {clap(), mert()}) {
            try (CLAP closeable = model) {
                float[] result = model.encodeAudio(pcm(), 24000, 0);
                finite(result, model.projectionDimension());
                assertArrayEquals(result, model.encodeAudio(pcm(), 24000), 1e-6f);
                assertArrayEquals(result, model.encodeAudio(fixtures.resolve("audio 世界.wav"), 0), 2e-5f);
                assertArrayEquals(result, model.encodeAudio(fixtures.resolve("audio 世界.wav")), 2e-5f);
                float[] nativeRate = new float[model.audioSampleRate()];
                for (int i = 0; i < nativeRate.length; i++) nativeRate[i] = ((i % 97) - 48) / 256.0f;
                assertArrayEquals(model.encodeAudio(nativeRate, model.audioSampleRate(), 0),
                        model.encodeAudio(nativeRate), 1e-6f);
            }
        }
    }

    @Test void inputValidationAndNativeErrorsLeaveModelUsable() {
        try (CLAP model = mert()) {
            assertThrows(NullPointerException.class, () -> model.encodeText(null));
            assertThrows(IllegalArgumentException.class, () -> model.encodeText("a\0b"));
            assertThrows(IllegalArgumentException.class, () -> model.encodeText("\ud800"));
            assertThrows(IllegalArgumentException.class, () -> model.encodeText("\udc00"));
            assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(new float[0], 24000));
            assertThrows(NullPointerException.class, () -> model.encodeAudio((float[]) null));
            assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(pcm(), 0));
            assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(pcm(), -1));
            for (float bad : new float[] {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
                assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(new float[] {bad}, 24000));
            }
            assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(pcm(), 24000, -1));
            assertThrows(NullPointerException.class, () -> model.encodeAudio((Path) null, 0));
            assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(Path.of(""), 0));
            assertThrows(IllegalArgumentException.class, () -> model.encodeAudio(fixtures.resolve("audio 世界.wav"), -1));
            IllegalStateException unsupported = assertThrows(IllegalStateException.class, () -> model.encodeText("hello"));
            String diagnostic = unsupported.getMessage();
            assertTrue(diagnostic.contains("clap_encode_text"));
            assertFalse(diagnostic.endsWith(": "));
            assertThrows(IllegalStateException.class, () -> model.encodeAudio(pcm(), 24000, 1));
            assertThrows(IllegalStateException.class, () -> model.encodeAudio(fixtures.resolve("absent.wav")));
            assertThrows(IllegalStateException.class, () -> model.encodeAudio(fixtures.resolve("bad.wav")));
            finite(model.encodeAudio(pcm()), 4);
            assertEquals(diagnostic, unsupported.getMessage());
        }
        assertThrows(IllegalStateException.class, () -> new CLAP(fixtures.resolve("missing.gguf"), OPTIONS));
    }

    @Test void closeIsIdempotentAndAllCallsAfterCloseFail() {
        CLAP model = clap();
        model.close();
        model.close();
        assertThrows(IllegalStateException.class, () -> model.encodeText("hello"));
        assertThrows(IllegalStateException.class, () -> model.encodeAudio(pcm()));
        assertThrows(IllegalStateException.class, () -> model.encodeAudio(pcm(), 24000));
        assertThrows(IllegalStateException.class, () -> model.encodeAudio(pcm(), 24000, 0));
        assertThrows(IllegalStateException.class, () -> model.encodeAudio(fixtures.resolve("audio 世界.wav")));
        assertThrows(IllegalStateException.class, () -> model.encodeAudio(fixtures.resolve("audio 世界.wav"), 0));
        assertThrows(IllegalStateException.class, model::audioSampleRate);
        assertThrows(IllegalStateException.class, model::projectionDimension);
    }

    @Test void concurrentInferenceAndCloseNeverUseFreedHandle() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(4);
        CLAP model = clap();
        try {
            float[] reference = model.encodeText("hello");
            List<Future<float[]>> results = new ArrayList<>();
            for (int i = 0; i < 8; i++) results.add(pool.submit(() -> model.encodeText("hello")));
            for (Future<float[]> result : results) assertArrayEquals(reference, result.get(30, TimeUnit.SECONDS), 1e-6f);
            CountDownLatch start = new CountDownLatch(1);
            Future<?> encode = pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < 5; i++) model.encodeText("hello");
                } catch (IllegalStateException closed) {
                    assertEquals("CLAP model is closed", closed.getMessage());
                } catch (InterruptedException e) { throw new AssertionError(e); }
            });
            Future<?> close = pool.submit(() -> {
                try { start.await(); model.close(); }
                catch (InterruptedException e) { throw new AssertionError(e); }
            });
            start.countDown();
            encode.get(30, TimeUnit.SECONDS);
            close.get(30, TimeUnit.SECONDS);
        } finally {
            model.close();
            pool.shutdownNow();
        }
    }
}

package io.gorse.clap;

import static org.junit.jupiter.api.Assertions.*;

import com.aresstack.huggingface.hub.HuggingFaceHub;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import org.junit.jupiter.api.Test;

/** Opt-in published weights smoke test, not a semantic accuracy benchmark. */
class PublishedModelIT {
    static final String REPO = "gorse-io/larger_clap_music-GGUF";
    static final String REVISION = "25401a7cae09f916a395b9f686ee1fa43628d815";
    static final String FILE = "larger_clap_music-q4_k_m.gguf";
    // HF /api/models/{repo}/revision/{revision}?blobs=true: siblings[].lfs.
    static final long SIZE = 128926752L;
    static final String SHA256 = "5f5cb986d8729930cbd681aa4d3dffbe4b1634d5e62a2e9823f8fa8f9e9ea247";

    @Test void publishedClapEncodesTextAndAudio() throws Exception {
        String library = System.getProperty("clap.library.path");
        assertNotNull(library, "published-model requires -Dclap.library.path=/absolute/path/to/libclap.so");
        assertTrue(Path.of(library).isAbsolute() && Files.isRegularFile(Path.of(library)),
                "configured native library must be an existing absolute filename");
        Path cache = Path.of(System.getProperty("clap.test.modelCache", "target/published-models"));
        Path weights = cache.resolve(REVISION).resolve(FILE).toAbsolutePath();
        // Download through huggingface4j, not a shell downloader or a substitute client.
        // Cached files are verified below too: overwrite(false) alone does NOT verify them.
        HuggingFaceHub.standard().anonymous().build().models().model(REPO).file(FILE)
                .revision(REVISION).downloadTo(weights).resume(true).overwrite(false)
                .verifySize(SIZE).verifySha256(SHA256).execute();
        assertEquals(SIZE, Files.size(weights), "cached/downloaded model size");
        assertEquals(SHA256, sha256(weights), "cached/downloaded model SHA-256; delete corrupt cache to retry");

        try (CLAP model = new CLAP(weights, new CLAP.Options(2, "cpu", CLAP.ActivationPrecision.F32))) {
            assertEquals(512, model.projectionDimension());
            assertEquals(48000, model.audioSampleRate());
            float[] music = model.encodeText("A recording of instrumental music.");
            float[] speech = model.encodeText("A person speaking in a quiet room.");
            float[] repeated = model.encodeText("A recording of instrumental music.");
            // Deterministic, non-silent one-second mono tone; no external audio fixture required.
            float[] pcm = new float[48000];
            for (int i = 0; i < pcm.length; i++) {
                pcm[i] = (float) (0.2 * Math.sin(2 * Math.PI * 440 * i / 48000.0));
            }
            float[] audio = model.encodeAudio(pcm, 48000);
            for (float[] vector : new float[][] {music, speech, repeated, audio}) unitEmbedding(vector);
            assertArrayEquals(music, repeated, 1e-5f, "repeat text inference");
            double textSimilarity = cosine(music, speech);
            double crossModalSimilarity = cosine(music, audio);
            assertTrue(textSimilarity < 0.9999, "different prompts must not collapse to an identical embedding");
            assertEquals(1.0, cosine(audio, audio), 1e-6, "audio self similarity");
            System.out.printf("Published CLAP %s@%s/%s: verified %d bytes SHA-256=%s; "
                            + "dimension=512, text/text cosine=%.6f, text/audio cosine=%.6f%n",
                    REPO, REVISION, FILE, SIZE, SHA256, textSimilarity, crossModalSimilarity);
        }
    }

    private static void unitEmbedding(float[] vector) {
        assertEquals(512, vector.length);
        double squared = 0;
        for (float value : vector) {
            assertTrue(Float.isFinite(value), "embedding values must be finite");
            squared += (double) value * value;
        }
        assertEquals(1.0, Math.sqrt(squared), 1e-3, "CLAP L2 norm");
    }

    private static double cosine(float[] left, float[] right) {
        double dot = 0, leftSquared = 0, rightSquared = 0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            leftSquared += (double) left[i] * left[i];
            rightSquared += (double) right[i] * right[i];
        }
        double result = dot / Math.sqrt(leftSquared * rightSquared);
        assertTrue(Double.isFinite(result) && result >= -1.000001 && result <= 1.000001,
                "cosine must be finite and within [-1, 1]");
        return result;
    }

    private static String sha256(Path file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(file)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder hex = new StringBuilder();
        for (byte value : digest.digest()) {
            hex.append(Character.forDigit((value >>> 4) & 15, 16));
            hex.append(Character.forDigit(value & 15, 16));
        }
        return hex.toString();
    }
}

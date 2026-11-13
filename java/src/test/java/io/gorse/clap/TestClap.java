package io.gorse.clap;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.aresstack.huggingface.hub.HuggingFaceHub;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TestClap {
    @Test
    void testClap() throws Exception {
        Path modelPath = Path.of(System.getProperty("clap.test.modelCache", "target/models"))
                .resolve("larger_clap_general-q4_k_m.gguf").toAbsolutePath();
        HuggingFaceHub.standard().anonymous().build().models()
                .model("gorse-io/larger_clap_general-GGUF")
                .file("larger_clap_general-q4_k_m.gguf")
                .downloadTo(modelPath).resume(true).overwrite(false).execute();

        Path audioPath = Path.of("..").toAbsolutePath().normalize().resolve("assets/dog.wav");
        float[] audio;
        float[] dog;
        float[] vacuum;
        try (CLAP model = new CLAP(modelPath)) {
            audio = model.encodeAudio(audioPath, 0);
            dog = model.encodeText("Sound of a dog");
            vacuum = model.encodeText("Sound of vaccum cleaner");
        }

        // Learned logit_scale_a from laion/larger_clap_general's published weights.
        double logitScale = Math.exp(3.6549275);
        double dogLogit = cosine(audio, dog) * logitScale;
        double vacuumLogit = cosine(audio, vacuum) * logitScale;
        double maxLogit = Math.max(dogLogit, vacuumLogit);
        double dogExp = Math.exp(dogLogit - maxLogit);
        double vacuumExp = Math.exp(vacuumLogit - maxLogit);
        double dogScore = dogExp / (dogExp + vacuumExp);
        double vacuumScore = vacuumExp / (dogExp + vacuumExp);
        assertTrue(dogScore > 0.99, "dog score: " + dogScore);
        assertTrue(vacuumScore < 0.01, "vacuum cleaner score: " + vacuumScore);
    }

    private static double cosine(float[] left, float[] right) {
        double dot = 0;
        double leftSquared = 0;
        double rightSquared = 0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            leftSquared += (double) left[i] * left[i];
            rightSquared += (double) right[i] * right[i];
        }
        return dot / Math.sqrt(leftSquared * rightSquared);
    }
}

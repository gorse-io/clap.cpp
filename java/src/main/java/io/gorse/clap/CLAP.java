package io.gorse.clap;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Owning model handle backed by clap.cpp's public C API. Use try-with-resources.
 * Calls on one instance, including close, are serialized. Different instances
 * can run concurrently. Returned arrays are independent Java-owned copies.
 */
public final class CLAP implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();
    private final State state;
    private final Cleaner.Cleanable cleanable;

    public enum ActivationPrecision {
        MODEL_DEFAULT(0), F32(1);
        private final int nativeValue;
        ActivationPrecision(int nativeValue) { this.nativeValue = nativeValue; }
    }

    /** Immutable runtime options; zero threads and null/empty device select native defaults. */
    public static final class Options {
        private final int threads;
        private final String device;
        private final ActivationPrecision activationPrecision;

        public Options() { this(0, null, ActivationPrecision.MODEL_DEFAULT); }
        public Options(int threads, String device, ActivationPrecision activationPrecision) {
            if (threads < 0) throw new IllegalArgumentException("threads must be nonnegative");
            if (device != null) utf8(device, "device", true);
            this.threads = threads;
            this.device = device;
            this.activationPrecision = Objects.requireNonNull(activationPrecision, "activationPrecision");
        }
        public int threads() { return threads; }
        public String device() { return device; }
        public ActivationPrecision activationPrecision() { return activationPrecision; }
    }

    /** Load a GGUF model with native runtime defaults. */
    public CLAP(Path modelPath) { this(modelPath, new Options()); }

    /** Load a GGUF model; paths and all other strings cross the ABI as UTF-8. */
    public CLAP(Path modelPath, Options options) {
        String path = pathString(modelPath);
        Objects.requireNonNull(options, "options");
        NativeBindings.Api api = NativeBindings.load();
        NativeBindings.RuntimeOptions nativeOptions = new NativeBindings.RuntimeOptions();
        nativeOptions.threads = options.threads;
        nativeOptions.device = options.device;
        nativeOptions.activation_precision = options.activationPrecision.nativeValue;
        PointerByReference out = new PointerByReference();
        boolean transferred = false;
        try {
            check(api, api.clap_create(path, nativeOptions, out), "clap_create");
            if (out.getValue() == null) throw new IllegalStateException("clap_create returned a null handle");
            state = new State(api, out.getValue());
            cleanable = CLEANER.register(this, state);
            transferred = true;
        } finally {
            if (!transferred && out.getValue() != null) api.clap_destroy(out.getValue());
        }
    }

    /** Encode text; unsupported model modalities throw IllegalStateException. */
    public synchronized float[] encodeText(String text) {
        Pointer handle = handle();
        utf8(text, "text", true);
        return embedding((values, capacity) -> state.api.clap_encode_text(handle, text, values, capacity), "clap_encode_text");
    }

    /** Encode mono PCM already sampled at audioSampleRate(). */
    public synchronized float[] encodeAudio(float[] pcm) {
        return encodeAudio(pcm, audioSampleRate(), 0);
    }

    /** Encode mono PCM, resampling from sampleRate if needed, with zero crop offset. */
    public synchronized float[] encodeAudio(float[] pcm, int sampleRate) {
        return encodeAudio(pcm, sampleRate, 0);
    }

    /**
     * Encode nonempty finite mono PCM. cropOffset is measured at the model sample
     * rate, not the input rate; non-CLAP architectures may reject nonzero offsets.
     * Do not mutate pcm concurrently with this call.
     */
    public synchronized float[] encodeAudio(float[] pcm, int sampleRate, long cropOffset) {
        Pointer handle = handle();
        Objects.requireNonNull(pcm, "pcm");
        if (pcm.length == 0) throw new IllegalArgumentException("pcm must not be empty");
        if (sampleRate <= 0) throw new IllegalArgumentException("sampleRate must be positive");
        NativeBindings.SizeT offset = NativeBindings.checkedSize(cropOffset);
        for (float sample : pcm) {
            if (!Float.isFinite(sample)) throw new IllegalArgumentException("pcm must contain only finite samples");
        }
        return embedding((values, capacity) -> state.api.clap_encode_audio(handle, pcm,
                NativeBindings.checkedSize(pcm.length), sampleRate, offset, values, capacity), "clap_encode_audio");
    }

    /** Decode and encode a WAV/MP3/FLAC file with zero crop offset. */
    public synchronized float[] encodeAudio(Path path) { return encodeAudio(path, 0); }

    /** Decode WAV/MP3/FLAC by contents and encode with model-rate crop semantics. */
    public synchronized float[] encodeAudio(Path path, long cropOffset) {
        String filename = pathString(path);
        NativeBindings.SizeT offset = NativeBindings.checkedSize(cropOffset);
        // Path.toString() is user-controlled and can reentrantly close this model.
        Pointer handle = handle();
        return embedding((values, capacity) -> state.api.clap_encode_audio_file(handle, filename, offset, values, capacity),
                "clap_encode_audio_file");
    }

    /** Model audio sample rate, or zero if unavailable. */
    public synchronized int audioSampleRate() {
        try { return state.api.clap_audio_sample_rate(handle()); }
        finally { Reference.reachabilityFence(this); }
    }

    /** Output projection width, or zero if unavailable. */
    public synchronized int projectionDimension() {
        try { return state.api.clap_projection_dimension(handle()); }
        finally { Reference.reachabilityFence(this); }
    }

    /** Release the model. Idempotent; all subsequent inference/property calls fail. */
    @Override public synchronized void close() {
        try { cleanable.clean(); }
        finally { Reference.reachabilityFence(this); }
    }

    private Pointer handle() {
        if (state.handle == null) throw new IllegalStateException("CLAP model is closed");
        return state.handle;
    }

    @FunctionalInterface private interface Encoder {
        int call(float[] values, NativeBindings.SizeT capacity);
    }

    private float[] embedding(Encoder encoder, String operation) {
        try {
            int dimension = projectionDimension();
            if (dimension <= 0) throw new IllegalStateException("model has no projection dimension");
            float[] values = new float[dimension];
            check(state.api, encoder.call(values, NativeBindings.checkedSize(dimension)), operation);
            return values;
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    private static void check(NativeBindings.Api api, int status, String operation) {
        if (status != 0) {
            // Diagnostic is thread-local and expires at the next fallible C call.
            Pointer diagnostic = api.clap_last_error();
            String message = diagnostic == null ? "native error" : diagnostic.getString(0, "UTF-8");
            throw new IllegalStateException(operation + " failed (status " + status + "): " + message);
        }
    }

    private static String pathString(Path path) {
        return utf8(Objects.requireNonNull(path, "path").toString(), "path", false);
    }

    private static String utf8(String value, String name, boolean allowEmpty) {
        Objects.requireNonNull(value, name);
        if (!allowEmpty && value.isEmpty()) throw new IllegalArgumentException(name + " must not be empty");
        if (value.indexOf('\0') >= 0) throw new IllegalArgumentException(name + " must not contain NUL");
        // Reject malformed UTF-16 rather than silently substituting '?' during UTF-8 conversion.
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (Character.isHighSurrogate(ch)) {
                if (++i == value.length() || !Character.isLowSurrogate(value.charAt(i))) {
                    throw new IllegalArgumentException(name + " contains an unpaired surrogate");
                }
            } else if (Character.isLowSurrogate(ch)) {
                throw new IllegalArgumentException(name + " contains an unpaired surrogate");
            }
        }
        return value;
    }

    /** Static cleanup action deliberately contains no reference to its CLAP owner. */
    static final class State implements Runnable {
        private final NativeBindings.Api api;
        private Pointer handle;
        State(NativeBindings.Api api, Pointer handle) { this.api = api; this.handle = handle; }
        @Override public synchronized void run() {
            if (handle != null) {
                Pointer owned = handle;
                handle = null;
                api.clap_destroy(owned);
            }
        }
    }
}

package io.gorse.clap;

import com.sun.jna.IntegerType;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.ByReference;
import com.sun.jna.ptr.PointerByReference;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Only the supported embedding subset of include/clap.h; no JNI shim. */
final class NativeBindings {
    private NativeBindings() {}
    private static Api library;

    static synchronized Api load() {
        if (library == null) {
            String location = System.getProperty("clap.library.path");
            if (location != null) {
                Path path = Path.of(location);
                if (location.isEmpty() || !path.isAbsolute() || !Files.isRegularFile(path)) {
                    throw new IllegalArgumentException("clap.library.path must name an existing absolute shared library file");
                }
            }
            library = Native.load(location == null ? "clap" : location, Api.class,
                    Map.of(Library.OPTION_STRING_ENCODING, "UTF-8"));
        }
        return library;
    }

    public interface Api extends Library {
        int clap_create(String path, RuntimeOptions options, PointerByReference out);
        void clap_destroy(Pointer model);
        Pointer clap_last_error();
        void clap_free(Pointer allocation);
        int clap_encode_text(Pointer model, String text, PointerByReference values, SizeTReference count);
        int clap_encode_audio(Pointer model, float[] pcm, SizeT samples, int sampleRate, SizeT cropOffset,
                              PointerByReference values, SizeTReference count);
        int clap_encode_audio_file(Pointer model, String path, SizeT cropOffset,
                                   PointerByReference values, SizeTReference count);
        int clap_audio_sample_rate(Pointer model);
        int clap_projection_dimension(Pointer model);
    }

    @Structure.FieldOrder({"threads", "device", "activation_precision"})
    public static class RuntimeOptions extends Structure {
        public int threads;
        public String device;
        public int activation_precision;

        public RuntimeOptions() { setStringEncoding("UTF-8"); }
    }

    /** size_t is unsigned and pointer-width, including LLP64 Windows (not C long). */
    public static class SizeT extends IntegerType {
        private static final long serialVersionUID = 1L;
        public SizeT() { this(0); }
        public SizeT(long value) { super(Native.SIZE_T_SIZE, value, true); }
    }

    static SizeT checkedSize(long value) {
        if (value < 0 || (Native.SIZE_T_SIZE == 4 && value > 0xffffffffL)) {
            throw new IllegalArgumentException("value is outside the supported size_t range: " + value);
        }
        return new SizeT(value);
    }

    public static class SizeTReference extends ByReference {
        public SizeTReference() {
            super(Native.SIZE_T_SIZE);
            getPointer().clear(Native.SIZE_T_SIZE);
        }
        public long getValue() {
            return Native.SIZE_T_SIZE == 4 ? Integer.toUnsignedLong(getPointer().getInt(0)) : getPointer().getLong(0);
        }
    }
}

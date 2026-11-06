package io.gorse.clap;

import static org.junit.jupiter.api.Assertions.*;

import com.sun.jna.Native;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class CLAPTest {
    @Test void optionsValidateBeforeLoadingNativeCode() {
        assertThrows(IllegalArgumentException.class, () -> new CLAP.Options(-1, "cpu", CLAP.ActivationPrecision.F32));
        assertThrows(IllegalArgumentException.class, () -> new CLAP.Options(0, "cp\0u", CLAP.ActivationPrecision.MODEL_DEFAULT));
        assertThrows(NullPointerException.class, () -> new CLAP.Options(0, null, null));
        CLAP.Options options = new CLAP.Options();
        assertEquals(0, options.threads());
        assertNull(options.device());
        assertEquals(CLAP.ActivationPrecision.MODEL_DEFAULT, options.activationPrecision());
    }

    @Test void invalidModelPathDoesNotLoadNativeCode() {
        assertThrows(NullPointerException.class, () -> new CLAP((Path) null));
        assertThrows(IllegalArgumentException.class, () -> new CLAP(Path.of("")));
        assertThrows(NullPointerException.class, () -> new CLAP(Path.of("model.gguf"), null));
    }

    @Test void sizeTIsPlatformWidthAndUnsigned() {
        assertEquals(Native.SIZE_T_SIZE, Native.getNativeSize(NativeBindings.SizeT.class));
        if (Native.SIZE_T_SIZE == 4) {
            assertEquals(0xffffffffL, new NativeBindings.SizeT(0xffffffffL).longValue());
            assertThrows(IllegalArgumentException.class, () -> NativeBindings.checkedSize(0x100000000L));
        }
        assertThrows(IllegalArgumentException.class, () -> NativeBindings.checkedSize(-1));
        assertEquals(7, NativeBindings.checkedSize(7).longValue());
        NativeBindings.SizeTReference reference = new NativeBindings.SizeTReference();
        assertEquals(0, reference.getValue());
        if (Native.SIZE_T_SIZE == 4) reference.getPointer().setInt(0, -1);
        else reference.getPointer().setLong(0, 0xffffffffL);
        assertEquals(0xffffffffL, reference.getValue());
    }

    @Test void optionsStructureUsesCLayoutAndUtf8() {
        NativeBindings.RuntimeOptions options = new NativeBindings.RuntimeOptions();
        options.threads = 7;
        options.device = "设备";
        options.activation_precision = 1;
        options.write();
        int pointerOffset = Native.POINTER_SIZE == 8 ? 8 : 4;
        assertEquals(Native.POINTER_SIZE == 8 ? 24 : 12, options.size());
        assertEquals(7, options.getPointer().getInt(0));
        assertEquals("设备", options.getPointer().getPointer(pointerOffset).getString(0, "UTF-8"));
        assertEquals(1, options.getPointer().getInt(pointerOffset + Native.POINTER_SIZE));
    }

    @Test void publicCallsSerializeWithClose() throws Exception {
        for (java.lang.reflect.Method method : CLAP.class.getDeclaredMethods()) {
            if (Modifier.isPublic(method.getModifiers()) && !Modifier.isStatic(method.getModifiers())) {
                assertTrue(Modifier.isSynchronized(method.getModifiers()), method.toString());
            }
        }
        assertTrue(AutoCloseable.class.isAssignableFrom(CLAP.class));
    }

    @Test void cleanerStateCannotCaptureOwner() {
        assertTrue(Modifier.isStatic(CLAP.State.class.getModifiers()));
        for (java.lang.reflect.Field field : CLAP.State.class.getDeclaredFields()) {
            assertNotEquals(CLAP.class, field.getType());
        }
    }
}

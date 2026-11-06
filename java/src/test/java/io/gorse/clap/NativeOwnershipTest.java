package io.gorse.clap;

import static org.junit.jupiter.api.Assertions.*;

import com.sun.jna.Native;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "clap.library.path", matches = ".*")
class NativeOwnershipTest {
    @Test void pathConversionCanCloseModelWithoutCallingNativeEncoder() throws Exception {
        NativeIntegrationTest.generate();
        NativeBindings.Api real = NativeBindings.load();
        AtomicInteger encodes = new AtomicInteger();
        NativeBindings.Api observed = (NativeBindings.Api) Proxy.newProxyInstance(
                NativeBindings.Api.class.getClassLoader(), new Class<?>[] {NativeBindings.Api.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("clap_encode_audio_file")) {
                        encodes.incrementAndGet();
                        // Fail safely rather than pass a freed handle to the real runtime.
                        throw new AssertionError("native encoder called after reentrant close");
                    }
                    try { return method.invoke(real, args); }
                    catch (InvocationTargetException error) { throw error.getCause(); }
                });
        Field cached = NativeBindings.class.getDeclaredField("library");
        cached.setAccessible(true);
        cached.set(null, observed);
        try (CLAP model = NativeIntegrationTest.clap()) {
            Path path = (Path) Proxy.newProxyInstance(Path.class.getClassLoader(), new Class<?>[] {Path.class},
                    (proxy, method, args) -> {
                        if (method.getName().equals("toString")) {
                            model.close();
                            return NativeIntegrationTest.fixtures.resolve("audio 世界.wav").toString();
                        }
                        throw new AssertionError("unexpected Path method: " + method.getName());
                    });
            IllegalStateException closed = assertThrows(IllegalStateException.class, () -> model.encodeAudio(path, 0));
            assertEquals("CLAP model is closed", closed.getMessage());
            assertEquals(0, encodes.get());
        } finally { cached.set(null, real); }
    }

    @Test void actualAllocatedOutputIsFreedEvenWhenJavaRejectsItsLength() throws Exception {
        NativeIntegrationTest.generate();
        NativeBindings.Api real = NativeBindings.load();
        AtomicInteger frees = new AtomicInteger();
        AtomicInteger destroys = new AtomicInteger();
        AtomicBoolean corruptLength = new AtomicBoolean(true);
        // Delegate every call to the actual library. Only inject an impossible
        // count after successful real inference to exercise the finally path.
        NativeBindings.Api observed = (NativeBindings.Api) Proxy.newProxyInstance(
                NativeBindings.Api.class.getClassLoader(), new Class<?>[] {NativeBindings.Api.class},
                (proxy, method, args) -> {
                    Object result;
                    try { result = method.invoke(real, args); }
                    catch (InvocationTargetException error) { throw error.getCause(); }
                    if (method.getName().equals("clap_free") && args[0] != null) frees.incrementAndGet();
                    if (method.getName().equals("clap_destroy")) destroys.incrementAndGet();
                    if (method.getName().equals("clap_encode_text") && result.equals(0) && corruptLength.get()) {
                        NativeBindings.SizeTReference count = (NativeBindings.SizeTReference) args[3];
                        if (Native.SIZE_T_SIZE == 8) count.getPointer().setLong(0, -1);
                        else count.getPointer().setInt(0, -1);
                    }
                    return result;
                });
        Field cached = NativeBindings.class.getDeclaredField("library");
        cached.setAccessible(true);
        cached.set(null, observed);
        try {
            CLAP model = NativeIntegrationTest.clap();
            try {
                assertThrows(IllegalStateException.class, () -> model.encodeText("hello"));
                assertEquals(1, frees.get());
                corruptLength.set(false);
                float[] embedding = model.encodeText("hello");
                NativeIntegrationTest.finite(embedding, 3);
                assertEquals(2, frees.get());
                model.close();
                model.close();
                assertEquals(1, destroys.get());
                NativeIntegrationTest.finite(embedding, 3);
            } finally { model.close(); }
        } finally { cached.set(null, real); }
    }
}

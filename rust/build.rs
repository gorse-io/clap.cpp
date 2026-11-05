use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    path::PathBuf,
    process::Command,
};

fn run(command: &mut Command) {
    let status = command
        .status()
        .expect("cannot run CMake; install CMake >=3.20 and a C++20 compiler");
    assert!(status.success(), "native CMake build failed: {command:?}");
}

#[derive(Default)]
struct Backends {
    cuda: bool,
    vulkan: bool,
    rocm: bool,
    metal: bool,
}

impl Backends {
    fn from_features(target_os: &str, enabled: impl Fn(&str) -> bool) -> Self {
        Self {
            cuda: enabled("CARGO_FEATURE_CUDA"),
            vulkan: enabled("CARGO_FEATURE_VULKAN"),
            rocm: enabled("CARGO_FEATURE_ROCM"),
            metal: target_os == "macos" || enabled("CARGO_FEATURE_METAL"),
        }
    }

    fn validate(&self, os: &str, target_env: &str) -> Result<(), &'static str> {
        if os == "windows" && target_env != "msvc" {
            return Err("Windows requires the MSVC Rust toolchain");
        }
        if self.cuda && self.rocm {
            return Err("cuda and rocm cannot be enabled together (shared ggml CUDA symbols)");
        }
        if self.metal && os != "macos" {
            return Err("the metal Cargo feature requires macOS");
        }
        if self.rocm && os != "linux" {
            return Err("the rocm Cargo feature currently requires Linux");
        }
        if self.cuda && os != "linux" && os != "windows" {
            return Err("the cuda Cargo feature requires Linux or Windows (MSVC)");
        }
        if self.vulkan && !["linux", "windows", "macos"].contains(&os) {
            return Err("the vulkan Cargo feature requires Linux, Windows (MSVC), or macOS");
        }
        Ok(())
    }

    fn cmake_definitions(&self) -> [(&'static str, bool); 4] {
        [
            ("CUDA", self.cuda),
            ("VULKAN", self.vulkan),
            ("HIP", self.rocm),
            ("METAL", self.metal),
        ]
    }
}

fn validate_link_manifest(directives: &str) {
    assert!(
        directives
            .lines()
            .any(|line| line == "cargo:rustc-link-lib=static=clap"),
        "CMake link manifest is missing the static C API core: {directives:?}"
    );
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let backends = Backends::from_features(&target_os, |name| env::var_os(name).is_some());
    backends
        .validate(&target_os, &env::var("CARGO_CFG_TARGET_ENV").unwrap())
        .unwrap_or_else(|error| panic!("{error}"));
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.clone();
    for required in [
        "CMakeLists.txt",
        "cmake/static-development.cmake",
        "cmake/cargo-link.cmake",
        "include/clap.h",
        "ggml/CMakeLists.txt",
        "thirdparty/json/CMakeLists.txt",
        "thirdparty/icu/icu4c/source/common/unicode/uvernum.h",
        "thirdparty/miniaudio/miniaudio.h",
    ] {
        assert!(root.join(required).is_file(),
            "clap-cpp native source missing: {required}. In a Git checkout run git submodule update --init --recursive. Registry archives must contain the original native source trees (see rust/README.md).");
    }
    for path in [
        "CMakeLists.txt",
        "src",
        "include",
        "cmake",
        "ggml",
        "thirdparty/json",
        "thirdparty/icu",
        "thirdparty/miniaudio",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    for var in [
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CMAKE_PREFIX_PATH",
        "CUDAToolkit_ROOT",
        "CUDA_PATH",
        "CUDACXX",
        "CUDAHOSTCXX",
        "CMAKE_CUDA_ARCHITECTURES",
        "ROCM_PATH",
        "HIP_PATH",
        "HIPCXX",
        "CMAKE_HIP_ARCHITECTURES",
        "AMDGPU_TARGETS",
        "VULKAN_SDK",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let target = env::var("TARGET").unwrap();
    assert_eq!(
        env::var("HOST").unwrap(),
        target,
        "cross-compilation is not supported yet"
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    // Cargo package verification can reuse this OUT_DIR for the extracted crate.
    // CMake caches absolute source paths, so isolate each native source layout.
    let mut source_id = DefaultHasher::new();
    manifest.hash(&mut source_id);
    root.hash(&mut source_id);
    let out = out.join(format!("source-{:x}", source_id.finish()));
    let build = out.join("build");
    let install = out.join("native");
    let query = build.join(".cmake/api/v1/query");
    fs::create_dir_all(&query).expect("cannot create CMake File API query directory");
    fs::write(query.join("codemodel-v2"), "").expect("cannot request CMake codemodel");
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&root)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .args([
            "-DCLAP_BUILD_CLI=OFF",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCLAP_STATIC_DEPENDENCIES=ON",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DCLAP_USE_SYSTEM_ICU=OFF",
            "-DCLAP_GENERATE_LINK_METADATA=ON",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DGGML_STATIC=OFF",
            "-DGGML_NATIVE=OFF",
            "-DGGML_CPU=ON",
        ])
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install.display()));
    // Package policy belongs here, not in a parallel native CMake project.
    // GGML_STATIC also controls SDK runtime linkage and is unsupported by HIP;
    // BUILD_SHARED_LIBS=OFF supplies static core/backend archives independently.
    for option in [
        "METAL",
        "CUDA",
        "HIP",
        "VULKAN",
        "SYCL",
        "OPENCL",
        "OPENVINO",
        "RPC",
        "MUSA",
        "WEBGPU",
        "ZDNN",
        "CANN",
        "VIRTGPU",
        "VIRTGPU_BACKEND",
        "ET",
        "HEXAGON",
        "ZENDNN",
        "BLAS",
        "ACCELERATE",
        "BACKEND_DL",
        "OPENMP",
        "BUILD_TESTS",
        "BUILD_EXAMPLES",
        "CPU_ALL_VARIANTS",
        "SSE42",
        "AVX",
        "AVX2",
        "AVX_VNNI",
        "BMI2",
        "FMA",
        "F16C",
        "AVX512",
        "AVX512_VBMI",
        "AVX512_VNNI",
        "AVX512_BF16",
        "AMX_TILE",
        "AMX_INT8",
        "AMX_BF16",
    ] {
        configure.arg(format!("-DGGML_{option}=OFF"));
    }
    configure.arg(format!(
        "-DGGML_METAL_EMBED_LIBRARY={}",
        if backends.metal { "ON" } else { "OFF" }
    ));
    if let Some(toolchain) = env::var_os("CMAKE_TOOLCHAIN_FILE") {
        configure.arg(format!(
            "-DCMAKE_TOOLCHAIN_FILE={}",
            toolchain.to_string_lossy()
        ));
    }
    for (backend, enabled) in backends.cmake_definitions() {
        configure.arg(format!(
            "-DGGML_{backend}={}",
            if enabled { "ON" } else { "OFF" }
        ));
    }
    // CMake consumes compiler/SDK environment variables itself. Architecture
    // lists and package roots are cache variables, so pass them explicitly.
    for var in [
        "CUDAToolkit_ROOT",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_HIP_ARCHITECTURES",
        "AMDGPU_TARGETS",
    ] {
        if let Ok(value) = env::var(var) {
            configure.arg(format!("-D{var}={value}"));
        }
    }
    run(&mut configure);
    let link_manifest = out.join("cargo-link.txt");
    run(Command::new("cmake")
        .arg(format!("-DCLAP_LINK_BUILD={}", build.display()))
        .arg(format!("-DCLAP_LINK_MANIFEST={}", link_manifest.display()))
        .arg("-P")
        .arg(root.join("cmake/cargo-link.cmake")));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .args(["--config", "Release", "--target", "clap", "--parallel"])
        .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "2".into())));
    run(Command::new("cmake").arg("--install").arg(&build).args([
        "--config",
        "Release",
        "--component",
        "development",
    ]));
    println!(
        "cargo:rustc-link-search=native={}",
        install.join("lib").display()
    );
    let directives = fs::read_to_string(link_manifest).expect("cannot read CMake link manifest");
    validate_link_manifest(&directives);
    print!("{directives}");
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") {
        assert!(
            target.contains("msvc"),
            "Windows requires the MSVC Rust toolchain"
        );
        // Static ggml CPU detection and ICU timezone support use registry APIs.
        println!("cargo:rustc-link-lib=advapi32");
    } else {
        for lib in ["stdc++", "pthread", "m", "dl"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn link_manifest_accepts_windows_cmake_crlf() {
        // CMake file(WRITE) uses a text-mode stream, producing CRLF on Windows.
        validate_link_manifest(
            "cargo:rustc-link-lib=static=clap\r\ncargo:rustc-link-lib=static=ggml\r\n",
        );
    }

    #[test]
    fn link_manifest_requires_the_exact_static_core_directive() {
        validate_link_manifest("cargo:rustc-link-lib=static=clap\n");
        for directives in [
            "",
            "cargo:rustc-link-lib=dylib=clap\n",
            "cargo:rustc-link-lib=static=clap-icu-uc\n",
            "not-cargo:rustc-link-lib=static=clap\n",
            "cargo:rustc-link-lib=static=clap extra\r\n",
        ] {
            assert!(
                std::panic::catch_unwind(|| validate_link_manifest(directives)).is_err(),
                "accepted invalid manifest: {directives:?}"
            );
        }
    }

    #[test]
    fn macos_target_enables_metal_without_a_feature() {
        let backends = Backends::from_features("macos", |_| false);
        assert_eq!(backends.cmake_definitions()[3], ("METAL", true));
        assert!(backends.validate("macos", "").is_ok());
    }

    #[test]
    fn non_macos_targets_remain_cpu_only_without_features() {
        for os in ["linux", "windows"] {
            assert!(Backends::from_features(os, |_| false)
                .cmake_definitions()
                .iter()
                .all(|(_, on)| !on));
        }
    }

    #[test]
    fn explicit_metal_feature_remains_supported_and_platform_checked() {
        for os in ["macos", "linux", "windows"] {
            let backends = Backends::from_features(os, |name| name == "CARGO_FEATURE_METAL");
            assert_eq!(backends.cmake_definitions()[3], ("METAL", true));
            assert_eq!(backends.validate(os, "msvc").is_ok(), os == "macos");
        }
    }

    #[test]
    fn default_is_cpu_only_and_rocm_maps_to_hip() {
        assert!(Backends::default()
            .cmake_definitions()
            .iter()
            .all(|(_, on)| !on));
        let backends = Backends {
            rocm: true,
            ..Default::default()
        };
        assert_eq!(backends.cmake_definitions()[2], ("HIP", true));
    }

    #[test]
    fn validates_platforms_and_cuda_rocm_symbol_conflict() {
        for (os, abi) in [("linux", "gnu"), ("macos", ""), ("windows", "msvc")] {
            assert!(Backends::default().validate(os, abi).is_ok());
        }
        assert!(Backends::default().validate("windows", "gnu").is_err());
        let metal = Backends {
            metal: true,
            ..Default::default()
        };
        assert!(metal.validate("macos", "").is_ok());
        assert!(metal.validate("linux", "gnu").is_err());
        let cuda = Backends {
            cuda: true,
            ..Default::default()
        };
        assert!(cuda.validate("linux", "gnu").is_ok());
        assert!(cuda.validate("windows", "msvc").is_ok());
        assert!(cuda.validate("macos", "").is_err());
        let rocm = Backends {
            rocm: true,
            ..Default::default()
        };
        assert!(rocm.validate("linux", "gnu").is_ok());
        assert!(rocm.validate("windows", "msvc").is_err());
        let conflict = Backends {
            cuda: true,
            rocm: true,
            ..Default::default()
        };
        assert!(conflict
            .validate("linux", "gnu")
            .unwrap_err()
            .contains("cannot be enabled together"));
        let combined = Backends {
            cuda: true,
            vulkan: true,
            ..Default::default()
        };
        assert!(combined.validate("linux", "gnu").is_ok());
    }

    #[cfg(target_os = "linux")]
    struct Fixture(PathBuf);

    #[cfg(target_os = "linux")]
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    // Resolve a real CMake target graph without compiling or installing an SDK.
    #[cfg(target_os = "linux")]
    fn link_fixture(apple: bool, windows: bool) -> String {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let fixture = Fixture(env::temp_dir().join(format!(
            "clap cargo link {} {}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let source = &fixture.0;
        let build = source.join("build");
        fs::create_dir_all(build.join(".cmake/api/v1/query")).unwrap();
        fs::write(build.join(".cmake/api/v1/query/codemodel-v2"), "").unwrap();
        fs::write(source.join("empty.cpp"), "int main() { return 0; }\n").unwrap();
        let sdk = source
            .join("SDK with spaces")
            .to_string_lossy()
            .replace('\\', "/");
        let sdk_file = if windows {
            "libgpu.lib"
        } else if apple {
            "libgpu.dylib"
        } else {
            "libgpu.so.12"
        };
        let framework = if apple { "-framework Foundation" } else { "dl" };
        fs::write(
            source.join("CMakeLists.txt"),
            format!(
                r#"
cmake_minimum_required(VERSION 3.20)
project(link_fixture LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)
add_library(SDK::runtime SHARED IMPORTED)
set_target_properties(SDK::runtime PROPERTIES IMPORTED_LOCATION "{sdk}/{sdk_file}"
    INTERFACE_LINK_LIBRARIES "{framework}")
add_library(ggml-cuda STATIC empty.cpp)
target_link_libraries(ggml-cuda PRIVATE SDK::runtime)
add_library(clap STATIC empty.cpp)
target_link_libraries(clap PRIVATE ggml-cuda)
add_executable(clap-link-probe EXCLUDE_FROM_ALL empty.cpp)
target_link_libraries(clap-link-probe PRIVATE clap)
"#
            ),
        )
        .unwrap();
        run(Command::new("cmake")
            .arg("-S")
            .arg(source)
            .arg("-B")
            .arg(&build));
        fs::write(
            build.join("clap-link-config.cmake"),
            format!(
                "set(CLAP_LINK_ARCHIVES clap ggml-cuda)\nset(CLAP_LINK_APPLE {})\nset(CLAP_LINK_WINDOWS {})\n",
                if apple { "ON" } else { "OFF" },
                if windows { "ON" } else { "OFF" }
            ),
        )
        .unwrap();
        let manifest = source.join("directives.txt");
        let script = PathBuf::from(file!())
            .canonicalize()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("cmake/cargo-link.cmake");
        run(Command::new("cmake")
            .arg(format!("-DCLAP_LINK_BUILD={}", build.display()))
            .arg(format!("-DCLAP_LINK_MANIFEST={}", manifest.display()))
            .arg("-P")
            .arg(script));
        let directives = fs::read_to_string(manifest).unwrap();
        validate_link_manifest(&directives);
        assert!(directives.contains("cargo:rustc-link-lib=static=ggml-cuda\n"));
        assert!(directives.contains(&format!("cargo:rustc-link-search=native={sdk}\n")));
        directives
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn cmake_exports_transitive_sdk_paths_and_versioned_shared_libraries() {
        let directives = link_fixture(false, false);
        assert!(directives.contains("cargo:rustc-link-lib=dylib:+verbatim=libgpu.so.12\n"));
        assert!(directives.contains("cargo:rustc-link-lib=dl\n"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn cmake_preserves_windows_import_library_prefix() {
        let directives = link_fixture(false, true);
        assert!(directives.contains("cargo:rustc-link-lib=dylib=libgpu\n"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn cmake_exports_apple_frameworks_without_linker_args() {
        let directives = link_fixture(true, false);
        assert!(directives.contains("cargo:rustc-link-lib=dylib=gpu\n"));
        assert!(directives.contains("cargo:rustc-link-lib=framework=Foundation\n"));
        assert!(!directives.contains("rustc-link-arg"));
    }
}

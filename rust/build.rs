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

fn main() {
    let enabled = |name| env::var_os(name).is_some();
    let backends = Backends {
        cuda: enabled("CARGO_FEATURE_CUDA"),
        vulkan: enabled("CARGO_FEATURE_VULKAN"),
        rocm: enabled("CARGO_FEATURE_ROCM"),
        metal: enabled("CARGO_FEATURE_METAL"),
    };
    backends
        .validate(
            &env::var("CARGO_CFG_TARGET_OS").unwrap(),
            &env::var("CARGO_CFG_TARGET_ENV").unwrap(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let vendored = manifest.join("vendor");
    let root = if vendored.join("include/clap.h").is_file() {
        vendored
    } else {
        manifest.parent().unwrap().to_path_buf()
    };
    for required in [
        "include/clap.h",
        "ggml/CMakeLists.txt",
        "thirdparty/json/CMakeLists.txt",
        "thirdparty/icu/icu4c/source/common/unicode/uvernum.h",
        "thirdparty/miniaudio/miniaudio.h",
    ] {
        assert!(root.join(required).is_file(),
            "clap-cpp native source missing: {required}. In a Git checkout run git submodule update --init --recursive, then python3 rust/scripts/vendor.py before cargo package/publish. Registry archives must contain vendor/ (see README.md).");
    }
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("native").display()
    );
    for path in [
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
        .arg(manifest.join("native"))
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCLAP_ROOT={}", root.display()))
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install.display()));
    if let Some(toolchain) = env::var_os("CMAKE_TOOLCHAIN_FILE") {
        configure.arg(format!(
            "-DCMAKE_TOOLCHAIN_FILE={}",
            toolchain.to_string_lossy()
        ));
    }
    for (backend, enabled) in backends.cmake_definitions() {
        configure.arg(format!(
            "-DCLAP_RUST_{backend}={}",
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
        .arg(format!("-DCLAP_RUST_BUILD={}", build.display()))
        .arg(format!(
            "-DCLAP_RUST_LINK_MANIFEST={}",
            link_manifest.display()
        ))
        .arg("-P")
        .arg(manifest.join("native/CMakeLists.txt")));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .args([
            "--config",
            "Release",
            "--target",
            "clap-rust-native",
            "--parallel",
        ])
        .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "2".into())));
    run(Command::new("cmake").arg("--install").arg(&build).args([
        "--config",
        "Release",
        "--component",
        "rust",
    ]));
    println!(
        "cargo:rustc-link-search=native={}",
        install.join("lib").display()
    );
    let directives = fs::read_to_string(link_manifest).expect("cannot read CMake link manifest");
    assert!(
        directives.contains("cargo:rustc-link-lib=static=clap-rust-native"),
        "CMake link manifest is missing the Rust bridge"
    );
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
add_library(clap-rust-native STATIC empty.cpp)
target_link_libraries(clap-rust-native PRIVATE ggml-cuda)
add_executable(clap-rust-link-probe EXCLUDE_FROM_ALL empty.cpp)
target_link_libraries(clap-rust-link-probe PRIVATE clap-rust-native)
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
            build.join("cargo-link-config.cmake"),
            format!(
                "set(CLAP_RUST_ARCHIVES clap-rust-native ggml-cuda)\nset(CLAP_RUST_APPLE {})\nset(CLAP_RUST_WINDOWS {})\n",
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
            .join("native/CMakeLists.txt");
        run(Command::new("cmake")
            .arg(format!("-DCLAP_RUST_BUILD={}", build.display()))
            .arg(format!("-DCLAP_RUST_LINK_MANIFEST={}", manifest.display()))
            .arg("-P")
            .arg(script));
        let directives = fs::read_to_string(manifest).unwrap();
        assert!(directives.contains("cargo:rustc-link-lib=static=clap-rust-native\n"));
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

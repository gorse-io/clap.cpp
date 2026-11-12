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
            metal: target_os == "macos",
        }
    }

    fn validate(&self, os: &str, target_env: &str) -> Result<(), &'static str> {
        if os == "windows" && target_env != "msvc" {
            return Err("Windows requires the MSVC Rust toolchain");
        }
        if self.cuda && self.rocm {
            return Err("cuda and rocm cannot be enabled together (shared ggml CUDA symbols)");
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
        "CMAKE_C_COMPILER_LAUNCHER",
        "CMAKE_CXX_COMPILER_LAUNCHER",
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
        "CMAKE_C_COMPILER_LAUNCHER",
        "CMAKE_CXX_COMPILER_LAUNCHER",
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

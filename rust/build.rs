use std::{
    collections::hash_map::DefaultHasher,
    env,
    hash::{Hash, Hasher},
    path::PathBuf,
    process::Command,
};

fn run(command: &mut Command) {
    let status = command
        .status()
        .expect("cannot run CMake; install CMake >=3.20 and a C++17 compiler");
    assert!(status.success(), "native CMake build failed: {command:?}");
}

fn main() {
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
    run(&mut configure);
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
    for lib in [
        "clap-rust-native",
        "clap",
        "ggml",
        "ggml-cpu",
        "ggml-base",
        "clap-icu-i18n",
        "clap-icu-uc",
        "clap-icu-data",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
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

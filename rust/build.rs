use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command
        .status()
        .expect("cannot run CMake; install CMake >=3.20 and a C++17 compiler");
    assert!(status.success(), "native CMake build failed: {command:?}");
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap();
    assert!(root.join("include/clap.h").is_file(),
        "clap-cpp requires the parent repository native sources; use a path dependency from a recursive Git checkout, not a cargo package archive (see rust/README.md)");
    for path in [
        "rust/native",
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
    let build = out.join("build");
    let install = out.join("native");
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(manifest.join("native"))
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
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
    } else {
        for lib in ["stdc++", "pthread", "m", "dl"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

extern crate bindgen;

use std::path::{Path, PathBuf};
use std::{env, io};

use flate2::read::GzDecoder;
use tar::Archive;

#[derive(Clone, Copy, Eq, PartialEq)]
enum OS {
    Linux,
    MacOS,
    Windows,
}

impl OS {
    fn get() -> Self {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            os => panic!("Unsupported system {os}"),
        }
    }
}

fn make_shared_lib<P: AsRef<Path>>(os: OS, xla_dir: P) {
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    match os {
        OS::Linux | OS::MacOS => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .flag(&format!("-isystem{}", xla_dir.as_ref().join("include").display()))
                .flag("-std=c++17")
                .flag("-Wno-deprecated-declarations")
                .flag("-DLLVM_ON_UNIX=1")
                .flag("-DLLVM_VERSION_STRING=")
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
        OS::Windows => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
    };
}

fn env_var_rerun(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing out dir"));
    let os = OS::get();
    let xla_dir = env_var_rerun("XLA_EXTENSION_DIR")
        .map_or_else(|| out_dir.join("xla_extension"), PathBuf::from);
    if !xla_dir.exists() {
        download_xla(&xla_dir).await?;
    }
    let xla_dir = xla_dir.join("xla_extension");

    let jax_metal_dir =
        env_var_rerun("JAX_METAL_DIR").map_or_else(|| out_dir.join("jax_metal"), PathBuf::from);
    if !jax_metal_dir.exists() {
        download_jax_metal(&jax_metal_dir).await?;
    }

    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");

    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("c_xla.rs")).expect("Couldn't write bindings!");

    // Exit early on docs.rs as the C++ library would not be available.
    if std::env::var("DOCS_RS").is_ok() {
        return Ok(());
    }

    make_shared_lib(os, &xla_dir);

    // The --copy-dt-needed-entries -lstdc++ are helpful to get around some
    // "DSO missing from command line" error
    // undefined reference to symbol '_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@@GLIBCXX_3.4.21'
    if os == OS::Linux {
        println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
        println!("cargo:rustc-link-arg=-Wl,-lstdc++");
    }

    println!("cargo:rustc-link-search=native={}", xla_dir.join("lib").display());
    println!("cargo:rustc-link-lib=static=xla_rs");
    println!("cargo:rustc-link-lib=static=xla_extension");
    if os == OS::MacOS {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=SystemConfiguration");
        println!("cargo:rustc-link-lib=framework=Security");
    }
    Ok(())
}

async fn download_jax_metal(jax_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://files.pythonhosted.org/packages/7e/59/ff91dc65e7f945479b08509185d07de0c947e81c07705367b018cb072ee9/jax_metal-0.0.4-py3-none-macosx_11_0_arm64.whl";

    let res = reqwest::get(url).await?;
    let bytes = io::Cursor::new(res.bytes().await?);
    zip_extract::extract(bytes, jax_dir, true)?;
    Ok(())
}

async fn download_xla(xla_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    let url = match (os.as_str(), arch.as_str()) {
        ("macos", arch) => format!("https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-{}-darwin-cpu.tar.gz", arch),
        ("linux", arch) => format!("https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-{}-linux-gnu-cpu.tar.gz", arch),
        (os, arch) => panic!("{}-{} is an unsupported platform", os, arch)
    };
    let res = reqwest::get(url).await?;
    let mut bytes = io::Cursor::new(res.bytes().await?);

    let tar = GzDecoder::new(&mut bytes);
    let mut archive = Archive::new(tar);
    archive.unpack(xla_dir)?;

    Ok(())
}

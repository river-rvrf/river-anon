//! Records the compiler version so the benchmark can print it.
//!
//! A measurement without the toolchain that produced it beside it is not
//! reproducible, and the version is not otherwise visible at run time.
//! No dependency: this shells out to the compiler already building it.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let version = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=RIVER_RUSTC_VERSION={version}");

    // Cargo's own `TARGET`, which is the real triple
    // (`x86_64-unknown-linux-gnu`), not something reconstructed from
    // `std::env::consts::{ARCH, OS}` — those omit the vendor and ABI and
    // so name a different thing.
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=RIVER_TARGET={target}");
}

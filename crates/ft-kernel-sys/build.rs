fn main() {
    // cuda feature 下期望 kernels/ 已构建出 libftkernels.so 并通过
    // FT_KERNELS_DIR 指定位置；未启用时不做任何链接，纯声明编译。
    println!("cargo:rerun-if-env-changed=FT_KERNELS_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        // 注意：rustc/linker 的 cwd 是 workspace 根，而非本 crate；
        // 因此必须用绝对路径（由 CARGO_MANIFEST_DIR 推导）。
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR");
        let default_kernels = std::path::Path::new(&manifest_dir)
            .join("../../kernels/build");
        let dir = std::env::var("FT_KERNELS_DIR").unwrap_or_else(|_| {
            default_kernels.display().to_string()
        });
        let cuda_home =
            std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-search=native={cuda_home}/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=ftkernels");
    }
    if std::env::var("CARGO_FEATURE_CPU_SIMD").is_ok() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR");
        let default_kernels = std::path::Path::new(&manifest_dir)
            .join("../../kernels/build");
        let dir = std::env::var("FT_KERNELS_DIR").unwrap_or_else(|_| {
            default_kernels.display().to_string()
        });
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=ftcpu");
    }
}

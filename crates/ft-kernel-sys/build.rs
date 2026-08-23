fn main() {
    // cuda feature 下期望 kernels/ 已构建出 libftkernels.so 并通过
    // FT_KERNELS_DIR 指定位置；未启用时不做任何链接，纯声明编译。
    println!("cargo:rerun-if-env-changed=FT_KERNELS_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        let dir = std::env::var("FT_KERNELS_DIR")
            .unwrap_or_else(|_| "../../kernels/build".into());
        let cuda_home =
            std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-search=native={cuda_home}/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=ftkernels");
    }
}

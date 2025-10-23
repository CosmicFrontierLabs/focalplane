fn main() {
    // Link apriltag-utils library which contains image_u8_* functions
    // needed by the apriltag crate
    //
    // We use link-arg instead of rustc-link-lib to ensure it comes
    // at the end of the link command after -lapriltag
    println!("cargo:rustc-link-arg=-lapriltag-utils");
}

fn main() {
    // Compile C reference implementation for test comparisons
    // TODO: Bifurcate this to only compile when running tests
    // (CARGO_CFG_TEST env var approach didn't work as expected)
    cc::Build::new()
        .file("src/controllers/reference/CF_LOS_FB_40Hz.c")
        .include("src/controllers/reference")
        .compile("los_fb_reference");
}

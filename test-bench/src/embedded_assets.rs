//! Embedded frontend assets for self-contained server binaries.
//!
//! This module embeds the compiled WASM/JS/CSS frontend files directly into
//! the server binaries at compile time using `rust-embed`. This eliminates
//! runtime dependencies on external frontend files and simplifies deployment.

use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

/// Embedded assets for the calibrate frontend.
#[derive(RustEmbed)]
#[folder = "../test-bench-frontend/dist/calibrate"]
pub struct CalibrateAssets;

/// Embedded assets for the FGS frontend.
#[derive(RustEmbed)]
#[folder = "../test-bench-frontend/dist/fgs"]
pub struct FgsAssets;

/// Serve an embedded asset with SPA fallback to index.html for routes.
///
/// Files with extensions (e.g., `.js`, `.wasm`, `.css`) return 404 if not found.
/// Paths without extensions are treated as SPA routes and serve index.html.
fn serve_embedded<T: RustEmbed>(uri: &Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    // Try to serve the requested file
    if let Some(content) = T::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, mime.as_ref())],
            Body::from(content.data.to_vec()),
        )
            .into_response();
    }

    // Only fallback to index.html for SPA routes (paths without file extensions)
    // Requests for actual files (with extensions) should 404
    let has_extension = path.contains('.') && !path.ends_with('/');
    if has_extension {
        return (StatusCode::NOT_FOUND, "Asset not found").into_response();
    }

    // SPA fallback: serve index.html for client-side routing
    match T::get("index.html") {
        Some(content) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/html")],
            Body::from(content.data.to_vec()),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "index.html not found").into_response(),
    }
}

/// Serve embedded calibrate frontend assets.
pub async fn serve_calibrate_frontend(uri: Uri) -> Response {
    serve_embedded::<CalibrateAssets>(&uri)
}

/// Serve embedded FGS frontend assets.
pub async fn serve_fgs_frontend(uri: Uri) -> Response {
    serve_embedded::<FgsAssets>(&uri)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_file_matching<T: RustEmbed>(prefix: &str, suffix: &str) -> bool {
        T::iter().any(|f| f.starts_with(prefix) && f.ends_with(suffix))
    }

    fn verify_frontend_assets<T: RustEmbed>(name: &str, js_prefix: &str, wasm_prefix: &str) {
        println!("\n{name} frontend assets:");
        for file in T::iter() {
            println!("  - {file}");
        }

        let count = T::iter().count();
        println!("\nTotal {name} assets: {count}");
        assert!(count > 0, "No {name} assets were embedded!");

        assert!(T::get("index.html").is_some(), "index.html missing");
        // Trunk adds cache-busting hashes like: calibrate_wasm-691685085aec2853.js
        assert!(
            has_file_matching::<T>(js_prefix, ".js"),
            "{js_prefix}*.js missing"
        );
        assert!(
            has_file_matching::<T>(wasm_prefix, ".wasm"),
            "{wasm_prefix}*.wasm missing"
        );
    }

    #[test]
    fn test_calibrate_assets_embedded() {
        verify_frontend_assets::<CalibrateAssets>("calibrate", "calibrate_wasm", "calibrate_wasm");
    }

    #[test]
    fn test_fgs_assets_embedded() {
        verify_frontend_assets::<FgsAssets>("fgs", "fgs_wasm", "fgs_wasm");
    }

    #[test]
    fn test_mime_types() {
        let cases = [
            ("index.html", "text/html"),
            ("app.js", "text/javascript"),
            ("style.css", "text/css"),
            ("app.wasm", "application/wasm"),
        ];
        for (path, expected) in cases {
            assert_eq!(
                mime_guess::from_path(path).first_or_octet_stream().as_ref(),
                expected,
                "Wrong MIME type for {path}"
            );
        }
    }
}

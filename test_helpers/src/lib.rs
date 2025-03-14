//! Test helpers for meter-sim
//!
//! This crate provides common utilities for tests in the meter-sim project.

use once_cell::sync::Lazy;
use std::env;
use std::path::{Path, PathBuf};

/// Error type for test helper operations
#[derive(thiserror::Error, Debug)]
pub enum TestHelperError {
    #[error("Failed to find project root: {0}")]
    ProjectRootNotFound(String),
}

/// Returns the path to the project root directory.
///
/// This function searches for the project root by looking for the Cargo.toml file
/// that defines the workspace. It starts from the current directory and moves up
/// until it finds the workspace root.
///
/// # Returns
/// * Ok(PathBuf) - The path to the project root
/// * Err(TestHelperError) - If the project root could not be found
pub fn find_project_root() -> Result<PathBuf, TestHelperError> {
    let mut current_dir = env::current_dir().map_err(|e| {
        TestHelperError::ProjectRootNotFound(format!("Failed to get current directory: {}", e))
    })?;

    // Search for workspace Cargo.toml
    loop {
        let cargo_toml = current_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            // Check if this is the workspace root
            let content = std::fs::read_to_string(&cargo_toml).map_err(|e| {
                TestHelperError::ProjectRootNotFound(format!("Failed to read Cargo.toml: {}", e))
            })?;

            if content.contains("[workspace]") {
                return Ok(current_dir);
            }
        }

        // Go up one directory
        if !current_dir.pop() {
            break;
        }
    }

    Err(TestHelperError::ProjectRootNotFound(
        "Workspace root not found".to_string(),
    ))
}

/// Lazily initialized project root path
static PROJECT_ROOT: Lazy<PathBuf> =
    Lazy::new(|| find_project_root().expect("Failed to find project root directory"));

/// Returns the path to the output directory for test artifacts.
///
/// This function returns a path to a directory where test outputs like
/// images, plots, and other artifacts can be saved. It creates the directory
/// if it doesn't exist.
///
/// # Returns
/// * PathBuf - The path to the output directory
pub fn get_output_dir() -> PathBuf {
    let output_dir = PROJECT_ROOT.join("test_output");

    // Create the directory if it doesn't exist
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    }

    output_dir
}

/// Returns a path within the output directory.
///
/// This is a convenience function for building paths relative to the output directory.
///
/// # Arguments
/// * `path` - The relative path within the output directory
///
/// # Returns
/// * PathBuf - The full path to the file in the output directory
pub fn output_path<P: AsRef<Path>>(path: P) -> PathBuf {
    get_output_dir().join(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_root_exists() {
        let root = find_project_root().expect("Failed to find project root");
        assert!(root.exists());
        assert!(root.join("Cargo.toml").exists());
    }

    #[test]
    fn test_output_dir_created() {
        let output = get_output_dir();
        assert!(output.exists());
        assert!(output.is_dir());
    }

    #[test]
    fn test_output_path() {
        let path = output_path("test.png");
        assert_eq!(path, get_output_dir().join("test.png"));
    }
}

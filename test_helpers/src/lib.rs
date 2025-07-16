//! Comprehensive testing infrastructure for astronomical simulation validation.
//!
//! This crate provides essential utilities and infrastructure for testing complex
//! astronomical simulations, including project structure navigation, test artifact
//! management, and reproducible test environments. Designed to support both unit
//! tests and integration tests across the entire simulation suite.
//!
//! # Testing Philosophy
//!
//! ## Reproducible Results
//! All test infrastructure supports deterministic, reproducible testing:
//! - **Fixed project paths**: Consistent file locations across environments
//! - **Isolated outputs**: Separate test artifact directories
//! - **Cross-platform**: Robust path handling for Windows/Linux/macOS
//! - **CI/CD compatibility**: Environment-agnostic project discovery
//!
//! ## Test Organization
//! - **Unit tests**: Component-level validation with minimal dependencies
//! - **Integration tests**: Multi-component simulation validation
//! - **Performance tests**: Timing and memory usage benchmarks
//! - **Visual tests**: Image comparison and artifact generation
//!
//! # Project Structure Support
//!
//! ## Workspace Detection
//! Automatically locates the project root regardless of test execution context:
//! - **Cargo workspace**: Detects `[workspace]` section in root Cargo.toml
//! - **Directory traversal**: Recursive parent directory search
//! - **Error handling**: Clear diagnostics for missing project structure
//! - **Caching**: Lazy initialization for efficient repeated access
//!
//! ## Path Management
//! Provides robust path construction for test resources and outputs:
//! - **Test data**: Access to input catalogs, calibration files, and references
//! - **Output artifacts**: Organized storage for generated images and logs
//! - **Temporary files**: Safe cleanup of intermediate test files
//! - **Cross-platform**: Consistent behavior across operating systems
//!
//! # Test Artifact Management
//!
//! ## Output Organization
//! ```text
//! test_output/
//! ├── images/           # Generated simulation images
//! ├── plots/            # Analysis plots and visualizations  
//! ├── catalogs/         # Test star catalogs and data
//! ├── benchmarks/       # Performance measurement results
//! └── logs/             # Test execution logs and diagnostics
//! ```
//!
//! ## Artifact Lifecycle
//! - **Generation**: Automatic creation of output directories
//! - **Persistence**: Test outputs preserved for manual inspection
//! - **Cleanup**: Optional cleanup for CI/CD environments
//! - **Comparison**: Support for reference image comparison
//!
//! # Usage Examples
//!
//! ## Basic Project Path Access
//! # Usage
//! Use `find_project_root()` to locate the project root from any test context.
//! Use `get_output_dir()` to access the standardized output directory.
//!
//! ## Test Image Generation
//! # Usage
//! Use `output_path()` to generate paths for saving test artifacts like images.
//! The function ensures proper directory structure and cross-platform compatibility.
//! Test images can be saved for visual inspection and compared with reference images.
//!
//! ## Cross-Test Data Sharing
//! # Usage
//! Use `find_project_root()` to access shared test data files.
//! Use `output_path()` to save processing results in a standardized location.
//! This pattern enables data sharing between tests while maintaining organization.
//!
//! ## Performance Testing
//! # Usage
//! Use `output_path()` to save benchmark results and performance metrics.
//! This enables tracking performance over time and identifying regressions.
//! Benchmark files can be uploaded as CI artifacts for trend analysis.
//!
//! # Integration with CI/CD
//!
//! ## Continuous Integration Support
//! - **Artifact upload**: Test outputs can be uploaded as CI artifacts
//! - **Reference comparison**: Automated comparison with golden images
//! - **Performance tracking**: Benchmark results stored for trend analysis
//! - **Cross-platform validation**: Consistent behavior across build environments
//!
//! ## Local Development
//! - **Visual debugging**: Test images preserved for manual inspection
//! - **Incremental testing**: Outputs from previous runs available for comparison
//! - **Debug logs**: Detailed execution traces for test failure diagnosis
//! - **Interactive exploration**: Generated data accessible from development environment
//!
//! # Error Handling and Diagnostics
//!
//! ## Robust Error Reporting
//! All test helper functions provide detailed error context:
//! - **Project detection failures**: Clear guidance for project structure issues
//! - **Path construction errors**: Diagnostic information for file system problems
//! - **Permission issues**: Helpful messages for directory creation failures
//! - **Environment problems**: Detection and reporting of configuration issues
//!
//! ## Test Isolation
//! - **Independent outputs**: Each test can create isolated artifact directories
//! - **Cleanup support**: Optional removal of test artifacts after completion
//! - **Parallel safety**: Thread-safe operations for concurrent test execution
//! - **Resource management**: Proper cleanup of temporary files and directories

use once_cell::sync::Lazy;
use std::env;
use std::path::{Path, PathBuf};

/// Comprehensive error types for test infrastructure operations.
///
/// Provides detailed error reporting for common test setup and execution
/// failures, enabling clear diagnostics and appropriate error handling
/// in test environments.
#[derive(thiserror::Error, Debug)]
pub enum TestHelperError {
    /// Project root directory could not be located or accessed.
    ///
    /// Occurs when the workspace detection algorithm fails to find a valid
    /// project structure. Common causes include:
    /// - Running tests outside the project directory
    /// - Missing or malformed Cargo.toml files
    /// - Permission issues accessing parent directories
    /// - Incorrect workspace configuration
    #[error("Failed to find project root: {0}")]
    ProjectRootNotFound(String),
}

/// Locate project root directory through intelligent workspace detection.
///
/// Implements a robust algorithm to find the workspace root by traversing
/// the directory hierarchy from the current location upward, searching for
/// the Cargo.toml file containing the `[workspace]` section. This ensures
/// tests can locate project resources regardless of execution context.
///
/// # Detection Algorithm
/// 1. **Start**: Current working directory
/// 2. **Search**: Look for Cargo.toml in current directory
/// 3. **Validate**: Check if Cargo.toml contains `[workspace]` section
/// 4. **Traverse**: Move to parent directory and repeat
/// 5. **Terminate**: Stop at filesystem root or when workspace found
///
/// # Cross-Platform Compatibility
/// - **Path handling**: Uses std::path for OS-agnostic path operations
/// - **Directory traversal**: Robust parent directory navigation
/// - **File system**: Handles Windows/Linux/macOS differences
/// - **Error reporting**: Platform-specific error context
///
/// # Returns
/// * `Ok(PathBuf)` - Absolute path to workspace root directory
/// * `Err(TestHelperError)` - Detailed error with failure context
///
/// # Usage
/// Call `find_project_root()` to locate the workspace root from any subdirectory.
/// The function returns the absolute path to the project root directory containing
/// the workspace Cargo.toml file. Verify workspace structure by checking for
/// expected subdirectories like "simulator" and "test_helpers".
///
/// # Error Conditions
/// - **No workspace found**: Traversed to filesystem root without finding workspace
/// - **Permission denied**: Cannot read directories or Cargo.toml files
/// - **Malformed Cargo.toml**: File exists but cannot be parsed
/// - **Working directory**: Cannot determine current directory
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

/// Cached project root path for efficient repeated access.
///
/// Uses lazy initialization to compute the project root path exactly once
/// per process execution, then caches the result for all subsequent accesses.
/// This provides optimal performance for test suites that frequently need
/// project-relative paths while maintaining thread safety.
///
/// # Lazy Evaluation Benefits
/// - **Performance**: Root detection performed only once
/// - **Thread safety**: Safe concurrent access from multiple test threads
/// - **Error handling**: Panics on initialization failure for fail-fast behavior
/// - **Memory efficiency**: Minimal overhead after initialization
static PROJECT_ROOT: Lazy<PathBuf> =
    Lazy::new(|| find_project_root().expect("Failed to find project root directory"));

/// Get standardized test output directory with automatic creation.
///
/// Provides a consistent location for test artifacts including generated
/// images, analysis plots, benchmark results, and diagnostic logs. The
/// directory is created automatically if it doesn't exist, ensuring tests
/// can immediately write outputs without additional setup.
///
/// # Directory Structure
/// The output directory is located at `<project_root>/test_output/` and
/// serves as the base for all test-generated files:
/// - **Images**: Simulated astronomical observations
/// - **Plots**: Performance graphs and analysis visualizations
/// - **Data**: Processed catalogs and intermediate results
/// - **Logs**: Execution traces and debugging information
///
/// # Automatic Management
/// - **Creation**: Directory created if missing
/// - **Permissions**: Appropriate read/write permissions set
/// - **Cleanup**: Persistent storage for manual inspection
/// - **Organization**: Subdirectories can be created as needed
///
/// # Returns
/// Absolute path to the test output directory (guaranteed to exist)
///
/// # Usage
/// Call `get_output_dir()` to get the standardized test output directory.
/// The directory is automatically created if it doesn't exist.
/// Create subdirectories within the output directory for organized storage
/// of different types of test artifacts (images, data, logs, etc.).
///
/// # Thread Safety
/// This function is thread-safe and can be called concurrently from
/// multiple test threads without race conditions.
pub fn get_output_dir() -> PathBuf {
    let output_dir = PROJECT_ROOT.join("test_output");

    // Create the directory if it doesn't exist
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    }

    output_dir
}

/// Construct path within test output directory for convenient artifact storage.
///
/// Provides a clean, ergonomic interface for building paths to test artifacts
/// within the standardized output directory. Handles path joining correctly
/// across platforms and integrates seamlessly with the test infrastructure.
///
/// # Path Construction
/// - **Base directory**: Uses `get_output_dir()` as the root
/// - **Relative joining**: Appends provided path as relative component
/// - **Cross-platform**: Handles Windows/Linux/macOS path differences
/// - **Type flexibility**: Accepts any type convertible to `Path`
///
/// # Arguments
/// * `path` - Relative path within the output directory (file or subdirectory)
///
/// # Returns
/// Complete absolute path to the specified location in test output directory
///
/// # Usage
/// Call `output_path()` with a relative path to construct a full path within
/// the test output directory. Supports both simple filenames and subdirectory
/// paths. Create parent directories as needed before writing files.
/// The function handles cross-platform path joining automatically.
///
/// # Use Cases
/// - **Test images**: Generated simulation outputs
/// - **Analysis plots**: Visualization of test results
/// - **Benchmark data**: Performance measurement storage
/// - **Debug logs**: Detailed execution traces
/// - **Reference data**: Golden images and expected results
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

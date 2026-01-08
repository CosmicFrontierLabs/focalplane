# TODO

## Future Improvements

### Test Bench
- [ ] Add NSV455 camera support to camera_server
  - Create NSV455 CameraInterface implementation
  - Update camera_server to support NSV455 alongside PlayerOne cameras
- [ ] Unify gain settings across NSV455 and POA cameras
  - Current: Different gain units/ranges between camera types (NSV455 uses V4L2 controls, POA uses SDK-specific values)
  - Needed: Define common gain abstraction with consistent units (e.g., linear multiplier or dB)
  - Location: shared/src/camera_interface/mod.rs (CameraInterface trait)
  - Why: Allows test code and binaries to use same gain values regardless of camera type
  - Benefit: Simpler command-line interfaces and consistent behavior across hardware
- [ ] Refactor Array2<u16> ↔ image conversions into shared module
  - Duplicate conversion code exists in multiple locations (calibration_analysis.rs, display_patterns/apriltag.rs, etc.)
  - Create shared utility functions for bidirectional conversions with proper grayscale handling
  - Update all call sites to use shared functions
- [ ] Add uncompressed frame endpoint for speed testing
  - Create endpoint that returns raw frame data without JPEG compression
  - Useful for benchmarking network throughput vs compression overhead
- [ ] Accept SpotParams directly in render_gaussian_spot
  - Location: test-bench/src/display_patterns/shared.rs (render_gaussian_spot function)
  - Current: Takes x, y, fwhm_pixels, normalization_factor as separate arguments
  - Needed: Accept SpotParams struct to reduce argument count and improve type safety
  - Why: SpotParams already bundles these values, passing struct is cleaner
- [ ] Create shared clap args for exposure time that return Duration
  - Current: Multiple binaries have their own exposure_ms: f64 args with manual Duration conversion
  - Needed: Create a reusable clap arg struct (like CameraArgs) for exposure settings
  - Location: Could go in test-bench/src/camera_init.rs or shared/src/cli_args.rs
  - Why: DRY - reduce duplicate arg definitions and Duration conversion code
  - Benefit: Consistent CLI interface for exposure across all camera binaries
- [ ] Get proper NSV455 camera serial number from Neutralino
  - Location: `hardware/src/nsv455/camera/nsv455_camera.rs` (get_serial method)
  - Current: Hardcoded to "NSV455_UNKNOWN" with warning
  - Problem: V4L2 doesn't provide standard serial, device path not camera-specific
  - V4L2 card name shows: `vi-output, imx455 1-0030` (I2C bus/address)
  - Needed: Ask Neutralino how to query unique camera identifier
  - Why: Bad pixel maps need unique identifier per physical camera

### CI/CD
- [ ] Add ARM64 build verification to CI pipeline
  - Current: ARM64 cross-compile job removed due to OpenSSL dependency issues
  - Options:
    1. Set up dedicated ARM64 self-hosted runner (preferred - native builds, no cross-compile complexity)
    2. Fix cross-compilation with proper multi-arch OpenSSL setup
    3. Use Docker-based cross-compilation with QEMU
  - Why: Need to verify ARM64 builds (orin_monitor, cam_serve) work on Jetson Orin hardware
  - Location: .github/workflows/ci.yml

### Monocle (FGS/Tracking)
- [ ] Refactor star selection into filter bank architecture with debug mode
  - Location: `monocle/src/selection.rs:57` (detect_and_select_guides function)
  - Current: Star filtering is done sequentially with separate filter stages
  - Needed: Decouple into composable filter modules with debug mode that generates pass/fail matrix
  - Why: Would enable visualization of which stars pass/fail each filter stage (heatmap/table format)
  - Benefit: Makes debugging guide star selection much easier and more transparent
  - Consider: Output annotated FITS files or debug PNGs showing filter results

- [ ] Implement reacquisition logic
  - Location: `monocle/src/lib.rs` (attempt_reacquisition function)
  - Current: Stub function with placeholder comments
  - Needed: Search in expanded ROIs, match with known guide stars
  - Why: Critical for robust star tracking when guide stars are temporarily lost

- [ ] Fix star detection in calibration
  - Location: Multiple test files failing to transition to Tracking state
  - Files affected: `monocle/tests/simple.rs`, `monocle/tests/synthetic.rs`, `monocle/tests/end_to_end.rs`
  - Current: Tests have warnings when not tracking after calibration
  - Why: Calibration should reliably detect stars and transition to tracking

- [ ] Fix tracking loss detection
  - Location: `monocle/tests/synthetic.rs` (large motion test)
  - Current: Still tracks even with 50px motion when it should lose lock
  - Needed: Improve detection of when stars have moved too far to track
  - Why: System should recognize when tracking is unreliable and trigger reacquisition

### Simulator - Hardware
- [ ] Homogenize Zeus 455M Pro gain settings with NSV455 camera configuration
  - Location: `simulator/src/hardware/sensor.rs` (IMX455 sensor config)
  - Current: Using gain setting of "60" in "High gain mode" based on QHY spreadsheet
  - Needed: Verify gain parameters match Player One Zeus 455M Pro manual specifications
  - Reference: <https://player-one-astronomy.com/download/Product%20specification%20manual/Zeus%20455M%20PRO%20camera%20Manual.pdf>
  - Why: Both cameras use IMX455 sensor but may have different gain mappings/settings
  - Benefit: Ensures consistent behavior between simulated and real hardware configurations

- [ ] Confirm telescope light efficiency value
  - Location: `simulator/src/hardware/telescope.rs`
  - Current: Using 0.70 as placeholder
  - Needed: Verify actual efficiency value for 470mm aperture, 3450mm focal length telescope
  - Why: Accurate light transmission affects photometric calculations

### Simulator - Noise Modeling
- [ ] Add zodiacal background contribution to noise calculation
  - Location: `simulator/src/bin/sensor_noise_renderer.rs`
  - Current: Only includes read noise and dark current
  - Needed: Add sky background (zodiacal light) to total expected noise
  - Why: Realistic noise floor includes sky background contribution

### Simulator - Motion Simulation
- [ ] Implement motion simulator
  - Location: `simulator/src/bin/motion_simulator.rs`
  - Current: Stub with placeholder message
  - Needed: Full implementation of spacecraft motion simulation
  - Why: Required for testing tracking under realistic motion profiles

### Simulator - Scene Processing
- [ ] Make StellarSource implement Locatable2d trait
  - Location: `simulator/src/sims/scene_runner.rs`
  - Current: Requires unnecessary transform in star detection
  - Needed: Add Locatable2d trait implementation to avoid conversion
  - Why: Cleaner API and removes needless transformations

- [ ] Fix catalog mock for testing
  - Location: `simulator/src/sims/scene_runner.rs`
  - Current: Test ignored due to unclear StarData structure
  - Needed: Create proper mock catalog after clarifying StarData structure
  - Why: Enables testing of experiment runner with known star catalogs

### Shared - Camera Interface
- [ ] Use camera constants to compute saturation values more accurately
  - Location: `shared/src/camera_interface/mod.rs` (CameraConfig::get_saturation)
  - Current: Simple placeholder using only bit depth (2^bit_depth - 1)
  - Needed: Account for max well depth and conversion gain (DN per electron) like SensorConfig::saturating_reading()
  - Why: Real sensors may saturate below ADC maximum due to well capacity limits
  - Note: Requires storing additional sensor parameters in CameraInterface implementations

### Shared - Image Processing
- [ ] Consolidate spot shape types into SpotShape
  - Location: `shared/src/image_proc/centroid.rs` (SpotShape struct)
  - Current: Multiple structs with duplicate flux/diameter/aspect_ratio/moments fields:
    - `test-bench/src/calibration/psf_pixels.rs:10` - PSFPixel
    - `shared/src/image_proc/detection/naive.rs:64` - StarDetection
    - `monocle/src/lib.rs:84` - GuideStar
  - Needed: Refactor these to embed or derive from SpotShape
  - Why: Reduces duplication and ensures consistent spot characterization across codebase
  - Benefit: Single source of truth for spot shape data

## Code Quality & Clarity Improvements

### CLI Documentation
- [ ] **Add comprehensive documentation to all clap CLI arguments**
  - Pattern: Search for clap args missing `long_help` attribute
  - Search command: `grep -r "#\[arg(" --include="*.rs" | grep -v "long_help"`
  - Example of well-documented arg: `test-bench/src/bin/dark_frame_analysis.rs` (all args have `help` + `long_help`)
  - Action: For each CLI binary, add `long_help` to all args explaining:
    - What the option does in detail
    - Valid ranges or constraints
    - Interaction with other options
    - Example usage if non-obvious
  - Why: `--help` is the primary interface for users discovering CLI tools
  - Binaries to check:
    - `test-bench/src/bin/cam_track.rs`
    - `test-bench/src/bin/cam_serve.rs`
    - `test-bench/src/bin/calibration_analysis.rs`
    - `simulator/src/bin/sensor_shootout.rs`
    - `simulator/src/bin/sensor_floor_est.rs`

### Documentation Patterns
- [ ] **Ensure all public functions have documentation**
  - Pattern: Search for `pub fn` without preceding `///` doc comments
  - Example: `shared/src/camera_interface/mod.rs` (Timestamp::to_duration)
  - Action: Add doc comments explaining purpose, parameters, returns, and examples
  - Why: All public API functions need documentation for clarity and discoverability

- [ ] **Add documentation to public struct and enum fields**
  - Pattern: Search for `pub struct` and `pub enum` with undocumented fields
  - Example: `monocle/src/selection.rs` (StarDetectionStats fields)
  - Action: Add `///` comments explaining each field's purpose
  - Why: Field documentation improves IDE assistance and API clarity

- [ ] **Clarify ambiguous parameter documentation**
  - Pattern: Look for tuple/array parameters with unclear dimension ordering
  - Example: `shared/src/image_proc/noise/generate.rs` ("(height, width)" vs "(rows, columns)")
  - Action: Use explicit terminology like "(rows, columns)" consistently
  - Why: Prevents bugs from reversed dimension parameters

- [ ] **Document configuration parameter ranges and defaults**
  - Pattern: Configuration structs with numeric fields lacking range guidance
  - Example: `monocle/src/config.rs` (detection_threshold_sigma)
  - Action: Add comments like `/// Detection threshold (typically 3.0-10.0 sigma)`
  - Why: Helps users understand reasonable parameter ranges

- [ ] **Expand type alias documentation with usage guidance**
  - Pattern: Type aliases with minimal or unclear documentation
  - Example: `shared/src/units.rs` (Wavelength type alias)
  - Action: Explain semantic distinctions and provide usage examples
  - Why: Clarifies when to use specialized type vs underlying type

- [ ] **Add examples to trait implementations**
  - Pattern: Trait implementations lacking usage examples
  - Example: `shared/src/units.rs` (from_celsius missing example)
  - Action: Add consistent `# Examples` sections to all trait methods
  - Why: Extension traits need clear usage examples

- [ ] **Document algorithm trade-offs in multi-algorithm modules**
  - Pattern: Modules offering multiple algorithm choices without guidance
  - Example: `shared/src/image_proc/detection/mod.rs`
  - Action: Add module-level docs comparing algorithms and selection criteria
  - Why: Guides users toward appropriate algorithm choice

- [ ] **Use consistent documentation formatting**
  - Pattern: Mixed documentation styles (inline comments vs `# Arguments` sections)
  - Example: `monocle/src/filters.rs`
  - Action: Standardize on proper `# Arguments`, `# Returns`, `# Examples` sections
  - Why: Consistency aids documentation generation and readability

- [ ] **Add return value documentation to helper functions**
  - Pattern: Helper functions with parameter docs but missing `# Returns`
  - Example: `simulator/src/hardware/dark_current.rs` (generate_temperature_points)
  - Action: Add `# Returns` sections even for private helpers
  - Why: Complete documentation aids maintenance

### Magic Number Extraction
- [ ] **Replace magic numbers with named constants**
  - Pattern: Hardcoded numeric literals in formulas (excluding 0, 1, 2)
  - Examples:
    - `shared/src/image_proc/centroid.rs` - hardcoded `65535.0` (16-bit saturation)
    - `shared/src/image_proc/detection/config.rs` - factors (1.2, 0.5, 0.8, 2.0)
    - `monocle/src/filters.rs` - sigmoid parameters (0.1, 20.0)
  - Action: Extract to named constants at module level (e.g., `const SATURATION_16BIT: f64 = 65535.0;`)
  - Why: Makes tuning parameters explicit, discoverable, and easier to adjust

### Naming Consistency
- [ ] **Use type-specific names for similar functions operating on different types**
  - Pattern: Similar functions with same name but different parameter types
  - Example: `monocle/src/filters.rs` (filter_by_saturation for u16 vs f64)
  - Action: Add type suffix like `filter_by_saturation_u16` vs `filter_by_saturation_f64`
  - Why: Prevents calling wrong function for wrong data type

- [ ] **Standardize abbreviations vs full words**
  - Pattern: Mixed use of "temp" vs "temperature", "calc" vs "calculate", etc.
  - Action: Choose one convention and apply consistently
  - Why: Improves code searchability and readability

### Error Handling Improvements
- [ ] **Replace .unwrap() with .expect() containing context**
  - Pattern: Search for `.unwrap()` calls outside test code
  - Example: `shared/src/image_proc/noise/generate.rs`
  - Action: Replace with `.expect("Descriptive message explaining what can't fail")`
  - Why: Better error messages aid debugging when invariants are violated

- [ ] **Standardize error message formatting**
  - Pattern: Inconsistent capitalization, punctuation, and phrasing in error messages
  - Example: `simulator/src/shared_args.rs` (mixed "Invalid" vs "must be" phrasing)
  - Action: Use consistent format like "must be in range [a, b]" or "Invalid {field}: {reason}"
  - Why: Professional, consistent error messages improve user experience

- [ ] **Implement error source chaining for complex errors**
  - Pattern: Error types implementing Error trait without source() method
  - Example: `shared/src/camera_interface/mod.rs` (CameraError)
  - Action: Add source() method to preserve error chains
  - Why: Improves error diagnostics and debugging of cascading failures

### Code Comments & Clarity
- [ ] **Document non-obvious safety techniques**
  - Pattern: Use of saturating arithmetic, unchecked operations, or bounds assumptions
  - Example: `simulator/src/shared_args.rs` (saturating_sub without explanation)
  - Action: Add comments explaining why technique is safe/necessary
  - Why: Explains non-obvious safety techniques for future maintainers

### Testing
- [ ] **Ensure all public functions have tests or are made private**
  - Pattern: Search for `pub fn` in library code without corresponding `#[test]`
  - Example: `shared/src/algo/icp.rs` (squared_distance, find_closest_points)
  - Action: Either add unit tests or change visibility to `pub(crate)` or private
  - Why: All public API functions should have test coverage

# TODO

## Future Improvements

### Test Bench
- [ ] Add NSV455 camera support to camera_server
  - Create NSV455 CameraInterface implementation
  - Update camera_server to support NSV455 alongside PlayerOne cameras
- [ ] Implement gain control for NSV455 camera
  - Add get_gain() and set_gain() methods to NSV455 CameraInterface implementation
  - Location: nsv455/src/camera/neutralino_imx455.rs (if CameraInterface impl exists)
- [ ] Refactor Array2<u16> ↔ image conversions into shared module
  - Duplicate conversion code exists in multiple locations (calibration_analysis.rs, display_patterns/apriltag.rs, etc.)
  - Create shared utility functions for bidirectional conversions with proper grayscale handling
  - Update all call sites to use shared functions
- [ ] Add uncompressed frame endpoint for speed testing
  - Create endpoint that returns raw frame data without JPEG compression
  - Useful for benchmarking network throughput vs compression overhead
- [ ] Reduce web viewer fetch timeout when Neutralino gets proper ethernet
  - Location: test-bench/templates/live_view.html:222
  - Current: 120 second timeout for 10 Mbps adapter (NSV455 full-res frames are ~102MB)
  - Needed: Reduce to ~30 seconds when proper ethernet is installed
  - Why: Current timeout is unnecessarily long for normal network conditions

### Dependencies
- [x] Update ndarray from 0.15 to 0.16 across all crates
  - Updated all crates to use ndarray 0.16
  - Updated starfield to 0.2.3 for ndarray 0.16 compatibility

### CI/CD
- [ ] Add ARM64 build verification to CI pipeline
  - Current: ARM64 cross-compile job removed due to OpenSSL dependency issues
  - Options:
    1. Set up dedicated ARM64 self-hosted runner (preferred - native builds, no cross-compile complexity)
    2. Fix cross-compilation with proper multi-arch OpenSSL setup
    3. Use Docker-based cross-compilation with QEMU
  - Why: Need to verify ARM64 builds (flight_monitor, cam_serve_poa) work on Jetson Orin hardware
  - Location: .github/workflows/ci.yml

### Monocle (FGS/Tracking)
- [ ] Refactor star selection into filter bank architecture with debug mode
  - Location: `monocle/src/selection.rs:61-67` (detect_and_select_guides function)
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
  - Location: `simulator/src/hardware/sensor.rs:667-677` (IMX455 sensor config)
  - Current: Using gain setting of "60" in "High gain mode" based on QHY spreadsheet
  - Needed: Verify gain parameters match Player One Zeus 455M Pro manual specifications
  - Reference: <https://player-one-astronomy.com/download/Product%20specification%20manual/Zeus%20455M%20PRO%20camera%20Manual.pdf>
  - Why: Both cameras use IMX455 sensor but may have different gain mappings/settings
  - Benefit: Ensures consistent behavior between simulated and real hardware configurations

- [ ] Confirm telescope light efficiency value
  - Location: `simulator/src/hardware/telescope.rs:518`
  - Current: Using 0.70 as placeholder
  - Needed: Verify actual efficiency value for 470mm aperture, 3450mm focal length telescope
  - Why: Accurate light transmission affects photometric calculations

### Simulator - Photometry
- [ ] Convert quantum efficiency internal storage to f64
  - Location: `simulator/src/photometry/quantum_efficiency.rs`
  - Current: Using lower precision storage
  - Needed: Change internal representation to f64 for better precision
  - Why: Photometric accuracy requires high precision in QE curves

### Simulator - Noise Modeling
- [ ] Add zodiacal background contribution to noise calculation
  - Location: `simulator/src/bin/sensor_noise_renderer.rs:234`
  - Current: Only includes read noise and dark current
  - Needed: Add sky background (zodiacal light) to total expected noise
  - Why: Realistic noise floor includes sky background contribution

### Simulator - Motion Simulation
- [ ] Implement motion simulator
  - Location: `simulator/src/bin/motion_simulator.rs:176`
  - Current: Stub with placeholder message
  - Needed: Full implementation of spacecraft motion simulation
  - Why: Required for testing tracking under realistic motion profiles

### Simulator - Scene Processing
- [ ] Make StellarSource implement Locatable2d trait
  - Location: `simulator/src/sims/scene_runner.rs:336`
  - Current: Requires unnecessary transform in star detection
  - Needed: Add Locatable2d trait implementation to avoid conversion
  - Why: Cleaner API and removes needless transformations

- [ ] Fix catalog mock for testing
  - Location: `simulator/src/sims/scene_runner.rs:746-787`
  - Current: Test ignored due to unclear StarData structure
  - Needed: Create proper mock catalog after clarifying StarData structure
  - Why: Enables testing of experiment runner with known star catalogs

### Shared - Camera Interface
- [ ] Use camera constants to compute saturation values more accurately
  - Location: `shared/src/camera_interface/mod.rs:131` (CameraConfig::get_saturation)
  - Current: Simple placeholder using only bit depth (2^bit_depth - 1)
  - Needed: Account for max well depth and conversion gain (DN per electron) like SensorConfig::saturating_reading()
  - Why: Real sensors may saturate below ADC maximum due to well capacity limits
  - Note: Requires storing additional sensor parameters in CameraInterface implementations

### Shared - Image Processing
- [ ] Cleanup constants in Airy disk gaussian approximation
  - Location: `shared/src/image_proc/airy.rs:469`
  - Current: Magic constants in gaussian normalization
  - Needed: Extract and document normalization constants
  - Why: Makes 2D integration and radius scaling explicit and maintainable

## Code Quality & Clarity Improvements

### Documentation Patterns
- [ ] **Ensure all public functions have documentation**
  - Pattern: Search for `pub fn` without preceding `///` doc comments
  - Example: `shared/src/camera_interface/mod.rs:40-43` (Timestamp::to_duration)
  - Action: Add doc comments explaining purpose, parameters, returns, and examples
  - Why: All public API functions need documentation for clarity and discoverability

- [ ] **Add documentation to public struct and enum fields**
  - Pattern: Search for `pub struct` and `pub enum` with undocumented fields
  - Example: `monocle/src/selection.rs:11-16` (StarDetectionStats fields)
  - Action: Add `///` comments explaining each field's purpose
  - Why: Field documentation improves IDE assistance and API clarity

- [ ] **Document error enum variants**
  - Pattern: Find error enums with undocumented variants
  - Example: `simulator/src/hardware/read_noise.rs:14-20` (ReadNoiseError)
  - Action: Add `///` comments explaining when each error occurs
  - Why: Improves error handling and debugging documentation

- [ ] **Clarify ambiguous parameter documentation**
  - Pattern: Look for tuple/array parameters with unclear dimension ordering
  - Example: `shared/src/image_proc/noise/generate.rs:65-76` ("(height, width)" vs "(rows, columns)")
  - Action: Use explicit terminology like "(rows, columns)" consistently
  - Why: Prevents bugs from reversed dimension parameters

- [ ] **Document configuration parameter ranges and defaults**
  - Pattern: Configuration structs with numeric fields lacking range guidance
  - Example: `monocle/src/config.rs:9-20` (detection_threshold_sigma)
  - Action: Add comments like `/// Detection threshold (typically 3.0-10.0 sigma)`
  - Why: Helps users understand reasonable parameter ranges

- [ ] **Expand type alias documentation with usage guidance**
  - Pattern: Type aliases with minimal or unclear documentation
  - Example: `shared/src/units.rs:19-23` (Wavelength type alias)
  - Action: Explain semantic distinctions and provide usage examples
  - Why: Clarifies when to use specialized type vs underlying type

- [ ] **Add examples to trait implementations**
  - Pattern: Trait implementations lacking usage examples
  - Example: `shared/src/units.rs:32-44` (from_celsius missing example)
  - Action: Add consistent `# Examples` sections to all trait methods
  - Why: Extension traits need clear usage examples

- [ ] **Document algorithm trade-offs in multi-algorithm modules**
  - Pattern: Modules offering multiple algorithm choices without guidance
  - Example: `shared/src/image_proc/detection/mod.rs`
  - Action: Add module-level docs comparing algorithms and selection criteria
  - Why: Guides users toward appropriate algorithm choice

- [ ] **Use consistent documentation formatting**
  - Pattern: Mixed documentation styles (inline comments vs `# Arguments` sections)
  - Example: `monocle/src/filters.rs:47-49`
  - Action: Standardize on proper `# Arguments`, `# Returns`, `# Examples` sections
  - Why: Consistency aids documentation generation and readability

- [ ] **Add return value documentation to helper functions**
  - Pattern: Helper functions with parameter docs but missing `# Returns`
  - Example: `simulator/src/hardware/dark_current.rs:32-40` (generate_temperature_points)
  - Action: Add `# Returns` sections even for private helpers
  - Why: Complete documentation aids maintenance

### TODO Comment Cleanup
- [x] **Consolidate scattered TODO comments into tracked issues**
  - Pattern: Search for `TODO`, `FIXME`, `XXX`, `HACK` comments in code
  - Action: Moved all inline TODOs to appropriate sections below
  - Why: TODOs should be tracked centrally for visibility and prioritization

### Magic Number Extraction
- [ ] **Replace magic numbers with named constants**
  - Pattern: Hardcoded numeric literals in formulas (excluding 0, 1, 2)
  - Examples:
    - `shared/src/image_proc/centroid.rs:52` - hardcoded `65535.0` (16-bit saturation)
    - `shared/src/image_proc/detection/config.rs:66-77` - factors (1.2, 0.5, 0.8, 2.0)
    - `monocle/src/filters.rs:147-148` - sigmoid parameters (0.1, 20.0)
  - Action: Extract to named constants at module level (e.g., `const SATURATION_16BIT: f64 = 65535.0;`)
  - Why: Makes tuning parameters explicit, discoverable, and easier to adjust

### Naming Consistency
- [ ] **Use type-specific names for similar functions operating on different types**
  - Pattern: Similar functions with same name but different parameter types
  - Example: `monocle/src/filters.rs:14-45, 103-133` (filter_by_saturation for u16 vs f64)
  - Action: Add type suffix like `filter_by_saturation_u16` vs `filter_by_saturation_f64`
  - Why: Prevents calling wrong function for wrong data type

- [ ] **Standardize abbreviations vs full words**
  - Pattern: Mixed use of "temp" vs "temperature", "calc" vs "calculate", etc.
  - Action: Choose one convention and apply consistently
  - Why: Improves code searchability and readability

### Error Handling Improvements
- [ ] **Replace .unwrap() with .expect() containing context**
  - Pattern: Search for `.unwrap()` calls outside test code
  - Example: `shared/src/image_proc/noise/generate.rs:74, 96, 128`
  - Action: Replace with `.expect("Descriptive message explaining what can't fail")`
  - Why: Better error messages aid debugging when invariants are violated

- [ ] **Standardize error message formatting**
  - Pattern: Inconsistent capitalization, punctuation, and phrasing in error messages
  - Example: `simulator/src/shared_args.rs:192-208` (mixed "Invalid" vs "must be" phrasing)
  - Action: Use consistent format like "must be in range [a, b]" or "Invalid {field}: {reason}"
  - Why: Professional, consistent error messages improve user experience

- [ ] **Implement error source chaining for complex errors**
  - Pattern: Error types implementing Error trait without source() method
  - Example: `shared/src/camera_interface/mod.rs:79` (CameraError)
  - Action: Add source() method to preserve error chains
  - Why: Improves error diagnostics and debugging of cascading failures

### Code Comments & Clarity
- [ ] **Document non-obvious safety techniques**
  - Pattern: Use of saturating arithmetic, unchecked operations, or bounds assumptions
  - Example: `simulator/src/shared_args.rs:353` (saturating_sub without explanation)
  - Action: Add comments explaining why technique is safe/necessary
  - Why: Explains non-obvious safety techniques for future maintainers

### Testing
- [ ] **Ensure all public functions have tests or are made private**
  - Pattern: Search for `pub fn` in library code without corresponding `#[test]`
  - Example: `shared/src/algo/icp.rs` (squared_distance, find_closest_points)
  - Action: Either add unit tests or change visibility to `pub(crate)` or private
  - Why: All public API functions should have test coverage

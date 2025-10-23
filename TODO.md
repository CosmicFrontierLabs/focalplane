# TODO

## Future Improvements

### Test Bench
- [ ] Add NSV455 camera support to camera_server
  - Create NSV455 CameraInterface implementation
  - Update camera_server to support NSV455 alongside PlayerOne cameras
- [ ] Refactor Array2<u16> ↔ image conversions into shared module
  - Duplicate conversion code exists in multiple locations (calibration_analysis.rs, display_patterns/apriltag.rs, etc.)
  - Create shared utility functions for bidirectional conversions with proper grayscale handling
  - Update all call sites to use shared functions
- [ ] Optimize Orin build script apt checks
  - Add flag file to skip repeated apt package checks on subsequent builds
  - Only recheck if flag file missing or after timeout period
- [ ] Add uncompressed frame endpoint for speed testing
  - Create endpoint that returns raw frame data without JPEG compression
  - Useful for benchmarking network throughput vs compression overhead

### Dependencies
- [ ] Update ndarray from 0.15 to 0.16 across all crates
  - Currently using 0.15 in shared/camera_interface
  - Need to update camera-server and other dependent crates


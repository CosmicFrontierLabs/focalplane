# Monocle Test Suite

## Purpose

These tests are designed to ensure the **basic behaviors** of the Fine Guidance System (FGS) state machine loop. They verify that:

- State transitions occur correctly (Idle → Acquiring → Calibrating → Tracking)
- Basic star detection works with simple synthetic images
- The FGS can maintain tracking with small perturbations
- Callback events fire at appropriate times
- The system can recover from tracking loss

## What These Tests Are NOT

These tests are **not** intended to:

- Validate convergence properties of the tracking algorithms
- Measure tracking accuracy or precision
- Test performance under realistic star field conditions
- Verify behavior with complex motion patterns
- Benchmark computational performance

## For Accuracy and Performance Testing

For comprehensive testing of:
- Tracking accuracy and precision
- Algorithm convergence properties
- Performance with realistic star fields
- Complex motion scenarios
- Visual debugging and plotting

Please refer to the **`monocle_harness`** crate, which provides:
- Integration with the full simulator for realistic star fields
- Plotting and visualization tools
- Accuracy metrics and benchmarking
- Extensive parameterized testing across multiple scenarios

## Test Organization

- `common.rs` - Shared utilities for creating simple synthetic star images
- `simple.rs` - Basic single-star tracking tests
- `synthetic.rs` - Multi-star tracking with synthetic frames
- `position_tracking.rs` - Basic position reporting and motion handling
- `end_to_end.rs` - Full lifecycle tests of the FGS state machine

## Note on Synthetic Images

The synthetic star images created in these tests are deliberately simple and NOT realistic representations of actual star fields. They exist solely to verify that the FGS algorithms can:
1. Detect bright point sources
2. Compute centroids
3. Track motion

For realistic star field testing, use `monocle_harness`.
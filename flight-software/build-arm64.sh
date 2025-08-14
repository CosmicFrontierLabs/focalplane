#!/bin/bash
set -euo pipefail

# Quick build script for ARM64 after setup is complete

echo "Building flight-monitor for ARM64..."

# Build for ARM64 with proper linker
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
cargo build --target aarch64-unknown-linux-gnu --release --bin flight_monitor

echo "✓ ARM64 build complete!"
echo "Binary location: target/aarch64-unknown-linux-gnu/release/flight_monitor"

# Show binary info
echo ""
echo "Binary details:"
file target/aarch64-unknown-linux-gnu/release/flight_monitor
ls -lh target/aarch64-unknown-linux-gnu/release/flight_monitor
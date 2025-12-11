#!/bin/bash
set -euo pipefail

# Generic ARM64 cross-compilation build script for Jetson Orin
# Usage: ./build-arm64.sh <package-name> [binary-name]
# Example: ./build-arm64.sh test-bench orin_monitor
# Example: ./build-arm64.sh hardware

PACKAGE_NAME=${1:-}
BINARY_NAME=${2:-}

if [ -z "$PACKAGE_NAME" ]; then
    echo "Error: Package name required"
    echo "Usage: $0 <package-name> [binary-name]"
    echo ""
    echo "Available packages:"
    echo "  test-bench"
    echo "  hardware"
    exit 1
fi

echo "Building $PACKAGE_NAME for ARM64 (aarch64-unknown-linux-gnu)..."

# Ensure target is installed
rustup target add aarch64-unknown-linux-gnu 2>/dev/null || true

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Build command
BUILD_CMD="cargo build --target aarch64-unknown-linux-gnu --release --package $PACKAGE_NAME"

if [ -n "$BINARY_NAME" ]; then
    BUILD_CMD="$BUILD_CMD --bin $BINARY_NAME"
fi

# Execute build with ARM64 linker
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    $BUILD_CMD

echo "✓ ARM64 build complete!"
echo ""
echo "Binary location: target/aarch64-unknown-linux-gnu/release/"

# Show built binaries
echo ""
echo "Built binaries:"
if [ -n "$BINARY_NAME" ]; then
    BINARY_PATH="target/aarch64-unknown-linux-gnu/release/$BINARY_NAME"
    if [ -f "$BINARY_PATH" ]; then
        file "$BINARY_PATH"
        ls -lh "$BINARY_PATH"
    fi
else
    find target/aarch64-unknown-linux-gnu/release/ -maxdepth 1 -type f -executable | while read -r binary; do
        file "$binary"
        ls -lh "$binary"
    done
fi

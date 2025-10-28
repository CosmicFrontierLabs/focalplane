#!/bin/bash
set -euo pipefail

# Generic deployment script for Jetson Orin
# Builds locally, transfers to remote Orin, and optionally runs binaries

# Configuration
REMOTE_HOST="${ORIN_HOST:-cosmicfrontiers@orin-nano.tail944341.ts.net}"
TAILSCALE_DEVICE_NAME="orin-nano"
TARGET="aarch64-unknown-linux-gnu"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
PACKAGE_NAME=""
BINARY_NAME=""
SKIP_BUILD=false
KEEP_REMOTE=false
RUN_COMMAND=""

usage() {
    echo "Usage: $0 --package PACKAGE [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --package PKG        Package to deploy (flight-software, orin-dev)"
    echo ""
    echo "Options:"
    echo "  --binary BIN         Specific binary to deploy (deploys all if not specified)"
    echo "  --skip-build         Skip the build step"
    echo "  --keep-remote        Keep remote directory after deployment"
    echo "  --run CMD            Command to run remotely (e.g., './playerone_info --detailed')"
    echo "  -h, --help           Show this help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --package)
            PACKAGE_NAME="$2"
            shift 2
            ;;
        --binary)
            BINARY_NAME="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --keep-remote)
            KEEP_REMOTE=true
            shift
            ;;
        --run)
            RUN_COMMAND="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$PACKAGE_NAME" ]; then
    print_error "Package name is required"
    usage
fi

REMOTE_TEMP_DIR="/home/cosmicfrontiers/${PACKAGE_NAME}-deploy-$(date +%Y%m%d_%H%M%S)"

print_info "Deploying $PACKAGE_NAME to Jetson Orin..."
print_info "Remote host: $REMOTE_HOST"

# Step 0: Check Tailscale connectivity
print_info "Checking Tailscale connectivity..."
if ! command -v tailscale &> /dev/null; then
    print_error "Tailscale not installed. Install from: https://tailscale.com/download"
    exit 1
fi

TAILSCALE_STATUS=$(tailscale status 2>&1)
if echo "$TAILSCALE_STATUS" | grep -q "Logged out"; then
    print_error "Tailscale not logged in. Run: tailscale login"
    exit 1
fi

if ! echo "$TAILSCALE_STATUS" | grep -q "$TAILSCALE_DEVICE_NAME"; then
    print_error "Device '$TAILSCALE_DEVICE_NAME' not found on Tailscale network"
    print_error "Ensure Orin is powered on and running Tailscale"
    exit 1
fi
print_success "Tailscale verified - device '$TAILSCALE_DEVICE_NAME' is online"

# Step 1: Build
if [ "$SKIP_BUILD" = false ]; then
    print_info "Building $PACKAGE_NAME for $TARGET..."

    cd "$PROJECT_ROOT"

    rustup target add $TARGET 2>/dev/null || true

    BUILD_CMD="cargo build --target $TARGET --release --package $PACKAGE_NAME"
    if [ -n "$BINARY_NAME" ]; then
        BUILD_CMD="$BUILD_CMD --bin $BINARY_NAME"
    fi

    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
        $BUILD_CMD

    print_success "Build completed"
else
    print_warning "Skipping build (using existing binaries)"
fi

# Step 2: Create remote directory
print_info "Creating remote directory: $REMOTE_TEMP_DIR"
ssh $REMOTE_HOST "mkdir -p $REMOTE_TEMP_DIR"

# Step 3: Copy binaries
print_info "Copying binaries to remote host..."

if [ -n "$BINARY_NAME" ]; then
    # Copy specific binary
    BINARY_PATH="$PROJECT_ROOT/target/$TARGET/release/$BINARY_NAME"
    if [ -f "$BINARY_PATH" ]; then
        print_info "Copying $BINARY_NAME..."
        scp "$BINARY_PATH" "$REMOTE_HOST:$REMOTE_TEMP_DIR/"
    else
        print_error "Binary not found: $BINARY_PATH"
        exit 1
    fi
else
    # Copy all binaries from the package
    print_info "Finding binaries for $PACKAGE_NAME..."

    # Find binaries in the release directory
    find "$PROJECT_ROOT/target/$TARGET/release" -maxdepth 1 -type f -executable ! -name "*.so" ! -name "*.d" | while read -r binary; do
        binary_name=$(basename "$binary")
        # Skip deps directory artifacts
        if [[ ! "$binary_name" =~ ^(build|deps|examples|incremental)$ ]]; then
            print_info "Copying $binary_name..."
            scp "$binary" "$REMOTE_HOST:$REMOTE_TEMP_DIR/" || print_warning "Failed to copy $binary_name"
        fi
    done
fi

print_success "File transfer completed"

# Step 4: Run command if specified
if [ -n "$RUN_COMMAND" ]; then
    print_info "Running command on remote host: $RUN_COMMAND"
    echo ""
    ssh $REMOTE_HOST "cd $REMOTE_TEMP_DIR && $RUN_COMMAND" || print_error "Command failed"
    echo ""
fi

# Step 5: Cleanup
if [ "$KEEP_REMOTE" = false ]; then
    print_info "Cleaning up remote directory..."
    ssh $REMOTE_HOST "rm -rf $REMOTE_TEMP_DIR"
    print_success "Cleanup complete"
else
    print_warning "Keeping remote directory: $REMOTE_TEMP_DIR"
    print_info "To access: ssh $REMOTE_HOST 'cd $REMOTE_TEMP_DIR && bash'"
fi

print_success "Deployment complete!"

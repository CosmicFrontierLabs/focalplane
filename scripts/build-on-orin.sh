#!/bin/bash
set -euo pipefail

# Build Rust projects directly on remote ARM devices
# This script syncs source code to remote host and builds natively, avoiding cross-compilation issues
#
# Camera Hardware Configuration:
# - Neutralino (--neut): NSV455 camera (V4L2-based)
#   * Binaries like cam_serve_nsv do NOT require the "playerone" feature flag
#   * V4L2 libraries are incompatible with PlayerOne SDK in the same binary
# - Orin Nano (--orin): PlayerOne astronomy cameras
#   * Binaries like cam_serve_poa REQUIRE the "playerone" feature flag
#   * Build with: cargo build --release --features playerone --bin cam_serve_poa

# Configuration
REMOTE_HOST=""
DEVICE_TYPE=""
TAILSCALE_DEVICE_NAME=""
REMOTE_BUILD_DIR="rust-builds"
PACKAGE_NAME=""
BINARY_NAME=""
FEATURES=""
RUN_AFTER_BUILD=false
RUN_COMMAND=""

# Host presets
ORIN_HOST="${ORIN_HOST:-meawoppl@orin-nano.tail944341.ts.net}"
ORIN_TAILSCALE_NAME="orin-nano"
NEUT_HOST="cosmicfrontiers@orin-416.tail944341.ts.net"
NEUT_DEVICE_NAME="neutralino"
TEST_BENCH_HOST="meawoppl@cfl-test-bench.tail944341.ts.net"
TEST_BENCH_DEVICE_NAME="cfl-test-bench"

# Apt dependencies management
# The semaphore file caches apt package verification to speed up repeated builds.
# When you add a new apt package to the check_package list below:
#   1. Increment APT_DEPS_VERSION
#   2. Add the check_package call in the apt checks section
# The script will detect the version change and recheck all packages on the next run.
APT_DEPS_VERSION=2
APT_SEMAPHORE_FILE=".meter-sim-apt-deps-installed"

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

usage() {
    echo "Usage: $0 --package PACKAGE [--orin|--neut|--test-bench] [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --package PKG        Package to build (e.g., test-bench)"
    echo ""
    echo "Device Selection (one required):"
    echo "  --orin               Build on Jetson Orin Nano (${ORIN_HOST})"
    echo "  --neut               Build on Neutralino computer (${NEUT_HOST})"
    echo "  --test-bench         Build on CFL test bench (${TEST_BENCH_HOST})"
    echo ""
    echo "Options:"
    echo "  --binary BIN         Specific binary to build (e.g., camera_server)"
    echo "  --features FEAT      Cargo features to enable (auto-detected: orin=playerone)"
    echo "  --run CMD            Command to run after successful build"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  ORIN_HOST            Override Orin host (default: meawoppl@orin-nano.tail944341.ts.net)"
    echo ""
    echo "Examples:"
    echo "  $0 --package test-bench --orin --binary cam_track"
    echo "  $0 --package test-bench --neut --binary cam_serve_nsv --run './target/release/cam_serve_nsv'"
    echo "  $0 --package test-bench --test-bench --binary cam_track"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --orin)
            DEVICE_TYPE="orin"
            REMOTE_HOST="$ORIN_HOST"
            TAILSCALE_DEVICE_NAME="$ORIN_TAILSCALE_NAME"
            shift
            ;;
        --neut)
            DEVICE_TYPE="neut"
            REMOTE_HOST="$NEUT_HOST"
            TAILSCALE_DEVICE_NAME="$NEUT_DEVICE_NAME"
            shift
            ;;
        --test-bench)
            DEVICE_TYPE="test-bench"
            REMOTE_HOST="$TEST_BENCH_HOST"
            TAILSCALE_DEVICE_NAME="$TEST_BENCH_DEVICE_NAME"
            shift
            ;;
        --package)
            PACKAGE_NAME="$2"
            shift 2
            ;;
        --binary)
            BINARY_NAME="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --run)
            RUN_AFTER_BUILD=true
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

if [ -z "$DEVICE_TYPE" ]; then
    print_error "Device type is required. Use --orin, --neut, or --test-bench"
    usage
fi

# Auto-enable features based on device type if not explicitly set
if [ -z "$FEATURES" ]; then
    if [ "$DEVICE_TYPE" = "orin" ]; then
        FEATURES="playerone"
        print_info "Auto-enabling 'playerone' feature for Orin device"
    elif [ "$DEVICE_TYPE" = "neut" ]; then
        FEATURES="nsv455"
        print_info "Auto-enabling 'nsv455' feature for Neut device"
    fi
fi

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

print_info "Building $PACKAGE_NAME on ${DEVICE_TYPE} at $REMOTE_HOST"
print_info "Remote build directory: ~/$REMOTE_BUILD_DIR/$PROJECT_NAME"

# Step 0: Check Tailscale connectivity (for Tailscale-connected devices)
if [ "$DEVICE_TYPE" = "orin" ] || [ "$DEVICE_TYPE" = "test-bench" ]; then
    print_info "Checking Tailscale connectivity..."
    if ! command -v tailscale &> /dev/null; then
        print_error "Tailscale is not installed on this machine"
        print_error "Please install Tailscale from: https://tailscale.com/download"
        exit 1
    fi

    TAILSCALE_STATUS=$(tailscale status 2>&1)
    if echo "$TAILSCALE_STATUS" | grep -q "Logged out"; then
        print_error "Tailscale is not logged in"
        print_error "Please run: tailscale login"
        exit 1
    fi

    if ! echo "$TAILSCALE_STATUS" | grep -q "$TAILSCALE_DEVICE_NAME"; then
        print_error "Device '$TAILSCALE_DEVICE_NAME' not found on Tailscale network"
        print_error ""
        print_error "Please ensure:"
        print_error "  1. The Orin Nano is powered on"
        print_error "  2. Tailscale is running on the Orin Nano"
        print_error "  3. The device is logged into your Tailscale network"
        print_error ""
        print_error "Available devices:"
        echo "$TAILSCALE_STATUS" | head -10
        exit 1
    fi

    print_success "Tailscale connectivity verified - device '$TAILSCALE_DEVICE_NAME' is online"
fi

# Step 1: Check SSH connection
print_info "Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connection OK'" > /dev/null 2>&1; then
    print_error "Cannot connect to $REMOTE_HOST"
    print_error "Check SSH keys and network connectivity"
    exit 1
fi
print_success "SSH connection verified"

# Step 2: Ensure Rust is installed on Orin
print_info "Checking Rust installation on Orin..."
RUST_CHECK=$(ssh "$REMOTE_HOST" 'bash -l -c "command -v cargo"' || echo "not_found")
if [ "$RUST_CHECK" = "not_found" ]; then
    print_warning "Rust not found on Orin. Installing..."
    ssh "$REMOTE_HOST" 'bash -c "curl --proto \"=https\" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"'
    print_success "Rust installed on Orin"
else
    print_success "Rust already installed: $RUST_CHECK"
fi

# Step 3: Check and install required system libraries
print_info "Checking required system libraries on Orin..."

# Check semaphore file to skip repeated apt checks
SEMAPHORE_VERSION=$(ssh "$REMOTE_HOST" "cat ~/$APT_SEMAPHORE_FILE 2>/dev/null || echo '0'")
if [ "$SEMAPHORE_VERSION" = "$APT_DEPS_VERSION" ]; then
    print_success "Apt dependencies already verified (version $APT_DEPS_VERSION)"
else
    if [ "$SEMAPHORE_VERSION" != "0" ]; then
        print_info "Apt dependencies version mismatch (found $SEMAPHORE_VERSION, expected $APT_DEPS_VERSION)"
    fi

    MISSING_PACKAGES=()

    check_package() {
        local package_name=$1
        local check_result=$(ssh "$REMOTE_HOST" "dpkg -l | grep \"^ii  $package_name\" || echo 'not_found'")
        if [ "$check_result" = "not_found" ]; then
            MISSING_PACKAGES+=("$package_name")
            return 1
        else
            print_success "$package_name already installed"
            return 0
        fi
    }

    check_package "libusb-1.0-0-dev" || print_warning "libusb-1.0-0-dev not found"
    check_package "libssl-dev" || print_warning "libssl-dev not found"
    check_package "pkg-config" || print_warning "pkg-config not found"
    check_package "libcfitsio-dev" || print_warning "libcfitsio-dev not found"
    check_package "libclang-dev" || print_warning "libclang-dev not found"
    check_package "clang" || print_warning "clang not found"
    check_package "libapriltag-dev" || print_warning "libapriltag-dev not found"

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "Missing packages: ${MISSING_PACKAGES[*]}"
        print_info "Installing missing packages..."
        ssh "$REMOTE_HOST" "sudo apt-get update && sudo apt-get install -y ${MISSING_PACKAGES[*]}"
        print_success "All required packages installed"
    else
        print_success "All required system libraries already installed"
    fi

    # Write semaphore file with current version
    ssh "$REMOTE_HOST" "echo '$APT_DEPS_VERSION' > ~/$APT_SEMAPHORE_FILE"
    print_success "Apt dependencies verified, semaphore updated to version $APT_DEPS_VERSION"
fi

# Step 4: Create build directory structure
print_info "Creating build directory on Orin..."
ssh "$REMOTE_HOST" "mkdir -p ~/$REMOTE_BUILD_DIR"
print_success "Build directory ready"

# Step 5: Sync source code to Orin
print_info "Syncing source code to Orin..."
rsync -avz --delete \
    --include 'test-bench/display_assets/***' \
    --exclude 'target/' \
    --exclude 'debug/' \
    --exclude '.git/' \
    --exclude 'cats/' \
    --exclude 'analysis/' \
    --exclude 'plots/' \
    --exclude 'fgs_output/' \
    --exclude 'test_output/' \
    --exclude 'exp_logs/' \
    --exclude 'experiment_output*/' \
    --exclude 'docs/presentation_scripts/media/' \
    --exclude 'docs/presentation_scripts/.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.png' \
    --exclude '*.jpg' \
    --exclude '*.avi' \
    --exclude '*.mp4' \
    --exclude '*.bin' \
    --exclude '*.fits' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude 'add_custom_mode*.sh' \
    --exclude 'make_checkerboard.py' \
    "$PROJECT_ROOT/" \
    "$REMOTE_HOST:~/$REMOTE_BUILD_DIR/$PROJECT_NAME/"
print_success "Source code synced"

# Step 6: Build on Orin
print_info "Building $PACKAGE_NAME on Orin..."
echo ""

BUILD_CMD="cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && source ~/.cargo/env && cargo build --release --package $PACKAGE_NAME"

if [ -n "$BINARY_NAME" ]; then
    BUILD_CMD="$BUILD_CMD --bin $BINARY_NAME"
fi
if [ -n "$FEATURES" ]; then
    BUILD_CMD="$BUILD_CMD --features $FEATURES"
fi

ssh "$REMOTE_HOST" "bash -l -c '$BUILD_CMD'"

print_success "Build completed successfully!"

# Step 7: Show binary info
print_info "Binary information:"
if [ -n "$BINARY_NAME" ]; then
    ssh "$REMOTE_HOST" "bash -l -c 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && ls -lh target/release/$BINARY_NAME && file target/release/$BINARY_NAME'"
else
    ssh "$REMOTE_HOST" "bash -l -c 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && find target/release/ -maxdepth 1 -type f -executable -not -name \"*.so\" | head -5 | xargs ls -lh'"
fi

# Step 8: Run command if specified
if [ "$RUN_AFTER_BUILD" = true ]; then
    print_info "Running command on Orin: $RUN_COMMAND"
    echo ""
    ssh "$REMOTE_HOST" "bash -l -c 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && $RUN_COMMAND'"
    echo ""
fi

print_success "All operations completed!"
print_info "Build location on Orin: ~/$REMOTE_BUILD_DIR/$PROJECT_NAME/target/release/"
print_info "To access: ssh $REMOTE_HOST 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && bash'"

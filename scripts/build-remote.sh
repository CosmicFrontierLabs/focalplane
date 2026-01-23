#!/bin/bash
set -euo pipefail

# Build Rust projects on cfl-test-bench and deploy binaries to target ARM64 devices
# This script:
#   1. Syncs source code to cfl-test-bench (fast ARM64 build server)
#   2. Builds natively on cfl-test-bench (or in Docker for Jetson targets)
#   3. Copies only the binary to the target device
#   4. Optionally runs the binary on the target
#
# Build Modes:
# - Native build: Used for --test-bench (Ubuntu 24.04 glibc 2.39)
# - Docker build: Used for --orin/--neut/--nsv (Ubuntu 22.04 glibc 2.35)
#   * Uses jammy-builder Docker image to match Jetson's glibc version
#
# Supported Target Devices:
# - Orin Nano (--orin): PlayerOne astronomy cameras
#   * Auto-enables 'playerone' feature flag
# - Neutralino (--neut): NSV455 camera (V4L2-based) on orin-005
#   * Auto-enables 'nsv455' feature flag
# - NSV (--nsv): NSV455 camera (V4L2-based) on orin-416
#   * Auto-enables 'nsv455' feature flag
# - Test Bench (--test-bench): Build and run on cfl-test-bench itself

# Configuration
TARGET_HOST=""
DEVICE_TYPE=""
TAILSCALE_DEVICE_NAME=""
REMOTE_BUILD_DIR="rust-builds"
PACKAGE_NAME=""
BINARY_NAME=""
FEATURES=""
RUN_AFTER_BUILD=false
RUN_COMMAND=""

# Build server (always cfl-test-bench)
BUILD_HOST="meawoppl@cfl-test-bench.tail944341.ts.net"
BUILD_DEVICE_NAME="cfl-test-bench"

# Target host presets
ORIN_HOST="${ORIN_HOST:-meawoppl@orin-nano.tail944341.ts.net}"
ORIN_TAILSCALE_NAME="orin-nano"
NEUT_HOST="cosmicfrontier@orin-005.tail944341.ts.net"
NEUT_DEVICE_NAME="orin-005"
NSV_HOST="cosmicfrontier@orin-416.tail944341.ts.net"
NSV_DEVICE_NAME="orin-416"

# Apt dependencies management (for build server only)
APT_DEPS_VERSION=5
APT_SEMAPHORE_FILE=".meter-sim-apt-deps-installed"

# Docker image for Jetson-compatible builds (Ubuntu 22.04 / glibc 2.35)
DOCKER_IMAGE_NAME="meter-sim-jammy-builder"
DOCKER_IMAGE_VERSION="2"

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
    echo "Usage: $0 --package PACKAGE --binary BINARY [--orin|--neut|--nsv|--test-bench] [OPTIONS]"
    echo ""
    echo "Build on cfl-test-bench and deploy binary to target device."
    echo ""
    echo "Required:"
    echo "  --package PKG        Package to build (e.g., test-bench)"
    echo "  --binary BIN         Binary to build (required for deployment)"
    echo ""
    echo "Target Device (one required):"
    echo "  --orin               Deploy to Jetson Orin Nano (${ORIN_HOST})"
    echo "  --neut               Deploy to Neutralino/orin-005 (${NEUT_HOST})"
    echo "  --nsv                Deploy to NSV/orin-416 (${NSV_HOST})"
    echo "  --test-bench         Build and run on cfl-test-bench (no deployment)"
    echo ""
    echo "Options:"
    echo "  --features FEAT      Cargo features to enable (auto-detected for some devices)"
    echo "  --run CMD            Command to run after deployment (relative to binary location)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --package test-bench --binary fgs_server --neut"
    echo "  $0 --package test-bench --binary fgs_server --nsv --run './fgs_server'"
    echo "  $0 --package test-bench --binary calibrate_serve --test-bench --features sdl2"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --orin)
            DEVICE_TYPE="orin"
            TARGET_HOST="$ORIN_HOST"
            TAILSCALE_DEVICE_NAME="$ORIN_TAILSCALE_NAME"
            shift
            ;;
        --neut)
            DEVICE_TYPE="neut"
            TARGET_HOST="$NEUT_HOST"
            TAILSCALE_DEVICE_NAME="$NEUT_DEVICE_NAME"
            shift
            ;;
        --nsv)
            DEVICE_TYPE="nsv"
            TARGET_HOST="$NSV_HOST"
            TAILSCALE_DEVICE_NAME="$NSV_DEVICE_NAME"
            shift
            ;;
        --test-bench)
            DEVICE_TYPE="test-bench"
            TARGET_HOST="$BUILD_HOST"
            TAILSCALE_DEVICE_NAME="$BUILD_DEVICE_NAME"
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

if [ -z "$BINARY_NAME" ]; then
    print_error "Binary name is required for deployment"
    usage
fi

if [ -z "$DEVICE_TYPE" ]; then
    print_error "Device type is required. Use --orin, --neut, --nsv, or --test-bench"
    usage
fi

# Auto-enable features based on device type if not explicitly set
if [ -z "$FEATURES" ]; then
    if [ "$DEVICE_TYPE" = "orin" ]; then
        FEATURES="playerone"
        print_info "Auto-enabling 'playerone' feature for Orin device"
    elif [ "$DEVICE_TYPE" = "neut" ]; then
        FEATURES="nsv455"
        print_info "Auto-enabling 'nsv455' feature for Neutralino device"
    elif [ "$DEVICE_TYPE" = "nsv" ]; then
        FEATURES="nsv455"
        print_info "Auto-enabling 'nsv455' feature for NSV device"
    fi
fi

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

# Determine if we need to deploy to a separate target
DEPLOY_TO_TARGET=true
USE_DOCKER_BUILD=false
if [ "$DEVICE_TYPE" = "test-bench" ]; then
    DEPLOY_TO_TARGET=false
    USE_DOCKER_BUILD=false
    print_info "Building natively on cfl-test-bench (no deployment needed)"
else
    DEPLOY_TO_TARGET=true
    USE_DOCKER_BUILD=true
    print_info "Building in Docker (Ubuntu 22.04) on cfl-test-bench, deploying to ${DEVICE_TYPE} (${TARGET_HOST})"
fi

# Step 0: Check Tailscale connectivity
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

# Check build server is online
if ! echo "$TAILSCALE_STATUS" | grep -q "$BUILD_DEVICE_NAME"; then
    print_error "Build server '$BUILD_DEVICE_NAME' not found on Tailscale network"
    exit 1
fi
print_success "Build server '$BUILD_DEVICE_NAME' is online"

# Check target device is online (if deploying)
if [ "$DEPLOY_TO_TARGET" = true ]; then
    if ! echo "$TAILSCALE_STATUS" | grep -q "$TAILSCALE_DEVICE_NAME"; then
        print_error "Target device '$TAILSCALE_DEVICE_NAME' not found on Tailscale network"
        print_error "Available devices:"
        echo "$TAILSCALE_STATUS" | head -10
        exit 1
    fi
    print_success "Target device '$TAILSCALE_DEVICE_NAME' is online"
fi

# Step 1: Check SSH connection to build server
print_info "Testing SSH connection to build server..."
if ! ssh -o ConnectTimeout=5 "$BUILD_HOST" "echo 'Connection OK'" > /dev/null 2>&1; then
    print_error "Cannot connect to build server $BUILD_HOST"
    print_error "Check SSH keys and network connectivity"
    exit 1
fi
print_success "SSH connection to build server verified"

# Step 2: Check SSH connection to target (if deploying)
if [ "$DEPLOY_TO_TARGET" = true ]; then
    print_info "Testing SSH connection to target device..."
    if ! ssh -o ConnectTimeout=5 "$TARGET_HOST" "echo 'Connection OK'" > /dev/null 2>&1; then
        print_error "Cannot connect to target device $TARGET_HOST"
        print_error "Check SSH keys and network connectivity"
        exit 1
    fi
    print_success "SSH connection to target device verified"
fi

# Step 3: Ensure Rust is installed on build server
print_info "Checking Rust installation on build server..."
RUST_CHECK=$(ssh "$BUILD_HOST" 'bash -l -c "command -v cargo"' || echo "not_found")
if [ "$RUST_CHECK" = "not_found" ]; then
    print_warning "Rust not found on build server. Installing..."
    ssh "$BUILD_HOST" 'bash -c "curl --proto \"=https\" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"'
    print_success "Rust installed on build server"
else
    print_success "Rust already installed: $RUST_CHECK"
fi

# Step 4: Check and install required system libraries on build server
print_info "Checking required system libraries on build server..."

SEMAPHORE_VERSION=$(ssh "$BUILD_HOST" "cat ~/$APT_SEMAPHORE_FILE 2>/dev/null || echo '0'")
if [ "$SEMAPHORE_VERSION" = "$APT_DEPS_VERSION" ]; then
    print_success "Apt dependencies already verified (version $APT_DEPS_VERSION)"
else
    if [ "$SEMAPHORE_VERSION" != "0" ]; then
        print_info "Apt dependencies version mismatch (found $SEMAPHORE_VERSION, expected $APT_DEPS_VERSION)"
    fi

    MISSING_PACKAGES=()

    check_package() {
        local package_name=$1
        local check_result=$(ssh "$BUILD_HOST" "dpkg -l | grep \"^ii  $package_name\" || echo 'not_found'")
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
    check_package "libsdl2-dev" || print_warning "libsdl2-dev not found"
    check_package "libsdl2-image-dev" || print_warning "libsdl2-image-dev not found"
    check_package "libzmq3-dev" || print_warning "libzmq3-dev not found"
    check_package "libfontconfig1-dev" || print_warning "libfontconfig1-dev not found"

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "Missing packages: ${MISSING_PACKAGES[*]}"
        print_info "Installing missing packages..."
        ssh "$BUILD_HOST" "sudo apt-get update && sudo apt-get install -y ${MISSING_PACKAGES[*]}"
        print_success "All required packages installed"
    else
        print_success "All required system libraries already installed"
    fi

    ssh "$BUILD_HOST" "echo '$APT_DEPS_VERSION' > ~/$APT_SEMAPHORE_FILE"
    print_success "Apt dependencies verified, semaphore updated to version $APT_DEPS_VERSION"
fi

# Step 5: Create build directory structure on build server
print_info "Creating build directory on build server..."
ssh "$BUILD_HOST" "mkdir -p ~/$REMOTE_BUILD_DIR"
print_success "Build directory ready"

# Step 6: Sync source code to build server
print_info "Syncing source code to build server..."
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
    "$BUILD_HOST:~/$REMOTE_BUILD_DIR/$PROJECT_NAME/"
print_success "Source code synced"

# Step 7: Build on build server
if [ "$USE_DOCKER_BUILD" = true ]; then
    # Docker build for Jetson targets (Ubuntu 22.04 / glibc 2.35)
    print_info "Building $PACKAGE_NAME in Docker (Ubuntu 22.04)..."
    echo ""

    # Check if Docker image exists, build if not
    IMAGE_EXISTS=$(ssh "$BUILD_HOST" "docker images -q ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} 2>/dev/null || echo ''")
    if [ -z "$IMAGE_EXISTS" ]; then
        print_info "Building Docker image ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION}..."
        # Use --network=host during build to work around Docker bridge networking issues
        ssh "$BUILD_HOST" "cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && docker build --network=host -t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} -f docker/jammy-builder.Dockerfile ."
        print_success "Docker image built"
    else
        print_success "Docker image ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} already exists"
    fi

    # Build cargo command
    CARGO_CMD="cargo build --release --package $PACKAGE_NAME --bin $BINARY_NAME"
    if [ -n "$FEATURES" ]; then
        CARGO_CMD="$CARGO_CMD --features $FEATURES"
    fi

    # Run build in Docker container
    # Mount source and cargo cache for faster rebuilds
    ssh "$BUILD_HOST" "docker run --rm \
        -v ~/$REMOTE_BUILD_DIR/$PROJECT_NAME:/build \
        -v ~/.cargo/registry:/root/.cargo/registry \
        -v ~/.cargo/git:/root/.cargo/git \
        ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} \
        $CARGO_CMD"
else
    # Native build for cfl-test-bench itself
    print_info "Building $PACKAGE_NAME natively on build server..."
    echo ""

    BUILD_CMD="cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && source ~/.cargo/env && cargo build --release --package $PACKAGE_NAME --bin $BINARY_NAME"

    if [ -n "$FEATURES" ]; then
        BUILD_CMD="$BUILD_CMD --features $FEATURES"
    fi

    ssh "$BUILD_HOST" "bash -l -c '$BUILD_CMD'"
fi

print_success "Build completed successfully!"

# Step 8: Show binary info
print_info "Binary information:"
ssh "$BUILD_HOST" "bash -l -c 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && ls -lh target/release/$BINARY_NAME && file target/release/$BINARY_NAME'"

# Step 9: Deploy binary to target device (if not test-bench)
if [ "$DEPLOY_TO_TARGET" = true ]; then
    print_info "Deploying binary to target device..."

    # Create bin directory on target
    TARGET_USER=$(echo "$TARGET_HOST" | cut -d@ -f1)
    ssh "$TARGET_HOST" "mkdir -p ~/bin"

    # Copy binary from build server to target via local machine
    # (build server -> local -> target, since they may not have direct SSH access)
    TEMP_BINARY=$(mktemp)
    scp "$BUILD_HOST:~/$REMOTE_BUILD_DIR/$PROJECT_NAME/target/release/$BINARY_NAME" "$TEMP_BINARY"
    scp "$TEMP_BINARY" "$TARGET_HOST:~/bin/$BINARY_NAME"
    rm "$TEMP_BINARY"

    # Make executable
    ssh "$TARGET_HOST" "chmod +x ~/bin/$BINARY_NAME"

    print_success "Binary deployed to $TARGET_HOST:~/bin/$BINARY_NAME"

    # Deploy frontend files if they exist (required for fgs_server, calibrate_serve)
    if [ -d "$PROJECT_ROOT/test-bench-frontend/dist" ]; then
        print_info "Deploying frontend files to target device..."
        ssh "$TARGET_HOST" "mkdir -p ~/bin/test-bench-frontend"
        rsync -avz "$PROJECT_ROOT/test-bench-frontend/dist/" "$TARGET_HOST:~/bin/test-bench-frontend/dist/"
        print_success "Frontend files deployed"
    fi
fi

# Step 10: Run command if specified
if [ "$RUN_AFTER_BUILD" = true ]; then
    if [ "$DEPLOY_TO_TARGET" = true ]; then
        print_info "Running on target device: $RUN_COMMAND"
        echo ""
        # Run from ~/bin which contains both binary and test-bench-frontend/dist/
        ssh "$TARGET_HOST" "bash -l -c 'cd ~/bin && $RUN_COMMAND'"
    else
        print_info "Running on build server: $RUN_COMMAND"
        echo ""
        # Run from project root so frontend paths work
        ssh "$BUILD_HOST" "bash -l -c 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && ./target/release/$RUN_COMMAND'"
    fi
    echo ""
fi

print_success "All operations completed!"
if [ "$DEPLOY_TO_TARGET" = true ]; then
    print_info "Binary location on target: ~/bin/$BINARY_NAME"
    print_info "To run: ssh $TARGET_HOST '~/bin/$BINARY_NAME'"
else
    print_info "Binary location: ~/$REMOTE_BUILD_DIR/$PROJECT_NAME/target/release/$BINARY_NAME"
fi

#!/bin/bash
set -euo pipefail

# Build Rust projects directly on Jetson Orin
# This script syncs source code to Orin and builds natively, avoiding cross-compilation issues

# Configuration
REMOTE_HOST="${ORIN_HOST:-meawoppl@192.168.15.246}"
REMOTE_BUILD_DIR="rust-builds"
PACKAGE_NAME=""
BINARY_NAME=""
RUN_AFTER_BUILD=false
RUN_COMMAND=""

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
    echo "Usage: $0 --package PACKAGE [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --package PKG        Package to build (e.g., orin-dev)"
    echo ""
    echo "Options:"
    echo "  --binary BIN         Specific binary to build"
    echo "  --run CMD            Command to run after successful build"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  ORIN_HOST            Remote host (default: meawoppl@192.168.15.246)"
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

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

print_info "Building $PACKAGE_NAME on Jetson Orin at $REMOTE_HOST"
print_info "Remote build directory: ~/$REMOTE_BUILD_DIR/$PROJECT_NAME"

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

# Step 3: Ensure libusb is installed
print_info "Checking libusb-1.0-dev on Orin..."
LIBUSB_CHECK=$(ssh "$REMOTE_HOST" "dpkg -l | grep libusb-1.0-0-dev || echo 'not_found'")
if [ "$LIBUSB_CHECK" = "not_found" ]; then
    print_warning "libusb-1.0-dev not found. Installing..."
    ssh "$REMOTE_HOST" "sudo apt-get update && sudo apt-get install -y libusb-1.0-0-dev"
    print_success "libusb-1.0-dev installed"
else
    print_success "libusb-1.0-dev already installed"
fi

# Step 4: Create build directory structure
print_info "Creating build directory on Orin..."
ssh "$REMOTE_HOST" "mkdir -p ~/$REMOTE_BUILD_DIR"
print_success "Build directory ready"

# Step 5: Sync source code to Orin
print_info "Syncing source code to Orin..."
rsync -avz --delete \
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
    --exclude 'display_assets/' \
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
    ssh "$REMOTE_HOST" "bash -l -c 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/third_party/playerone-sdk/lib/arm64 $RUN_COMMAND'"
    echo ""
fi

print_success "All operations completed!"
print_info "Build location on Orin: ~/$REMOTE_BUILD_DIR/$PROJECT_NAME/target/release/"
print_info "To access: ssh $REMOTE_HOST 'cd ~/$REMOTE_BUILD_DIR/$PROJECT_NAME && bash'"

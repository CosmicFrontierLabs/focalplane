#!/bin/bash
set -euo pipefail

# Remote testing script for Nvidia Orin device
# Compiles locally, transfers to remote, and executes tests

# Configuration
REMOTE_HOST="cosmicfrontiers@192.168.15.229"
REMOTE_TEMP_DIR="/home/cosmicfrontiers/flight-software-test-$(date +%Y%m%d_%H%M%S)"
TARGET="aarch64-unknown-linux-gnu"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
TEST_MODE="all"
SKIP_BUILD=false
KEEP_REMOTE=false
VIDEO_DEVICE="/dev/video0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            TEST_MODE="$2"
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
        --video-device)
            VIDEO_DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE          Test mode: all, unit, v4l2, or bin (default: all)"
            echo "  --skip-build         Skip the build step (use existing binaries)"
            echo "  --keep-remote        Keep remote test directory after completion"
            echo "  --video-device DEV   Video device to use (default: /dev/video0)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_info "Starting remote testing on Nvidia Orin..."
print_info "Remote host: $REMOTE_HOST"
print_info "Test mode: $TEST_MODE"

# Step 1: Build for aarch64 target
if [ "$SKIP_BUILD" = false ]; then
    print_info "Building flight-software for $TARGET..."
    
    cd "$SCRIPT_DIR"
    
    # Ensure the target is available
    rustup target add $TARGET 2>/dev/null || true
    
    # Build all binaries and tests
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    cargo build --target $TARGET --release --all-targets
    
    # Build tests separately to ensure they're built
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    cargo test --target $TARGET --no-run --release
    
    print_success "Build completed successfully"
else
    print_warning "Skipping build step (using existing binaries)"
fi

# Step 2: Create remote directory
print_info "Creating remote test directory..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_TEMP_DIR"

# Step 3: Copy binaries to remote
print_info "Copying binaries to remote host..."

# Find and copy all test binaries
print_info "Finding test binaries..."
TEST_BINARIES=$(find "$PROJECT_ROOT/target/$TARGET/release/deps" -maxdepth 1 -type f -executable -name "flight_software-*" 2>/dev/null || true)

if [ -n "$TEST_BINARIES" ]; then
    for binary in $TEST_BINARIES; do
        print_info "Copying test binary: $(basename $binary)"
        scp "$binary" "$REMOTE_HOST:$REMOTE_TEMP_DIR/" || print_warning "Failed to copy $(basename $binary)"
    done
fi

# Copy the main binaries
if [ -f "$PROJECT_ROOT/target/$TARGET/release/flight_monitor" ]; then
    print_info "Copying flight_monitor binary..."
    scp "$PROJECT_ROOT/target/$TARGET/release/flight_monitor" "$REMOTE_HOST:$REMOTE_TEMP_DIR/"
fi

if [ -f "$PROJECT_ROOT/target/$TARGET/release/v4l2_test" ]; then
    print_info "Copying v4l2_test binary..."
    scp "$PROJECT_ROOT/target/$TARGET/release/v4l2_test" "$REMOTE_HOST:$REMOTE_TEMP_DIR/"
fi

print_success "File transfer completed"

# Step 4: Run tests remotely
print_info "Running tests on remote host..."

run_remote_tests() {
    local test_name=$1
    local command=$2
    
    print_info "Running $test_name..."
    if ssh $REMOTE_HOST "cd $REMOTE_TEMP_DIR && $command"; then
        print_success "$test_name passed"
    else
        print_error "$test_name failed"
        return 1
    fi
}

# Track test results
TEST_RESULTS=0

case $TEST_MODE in
    all|unit)
        # Run unit tests
        print_info "Running unit tests..."
        for test_binary in $TEST_BINARIES; do
            test_name=$(basename $test_binary)
            if [[ $test_name == flight_software-* ]]; then
                run_remote_tests "Unit test: $test_name" "./$test_name --test-threads=1" || TEST_RESULTS=$?
            fi
        done
        ;;&
        
    all|v4l2)
        # Run V4L2 tests
        if [ -f "$PROJECT_ROOT/target/$TARGET/release/v4l2_test" ]; then
            print_info "Running V4L2 camera tests..."
            
            # Single frame capture test
            run_remote_tests "V4L2 single capture" "VIDEO_DEVICE=$VIDEO_DEVICE ./v4l2_test single" || TEST_RESULTS=$?
            
            # Continuous capture test (10 frames)
            run_remote_tests "V4L2 continuous capture" "VIDEO_DEVICE=$VIDEO_DEVICE ./v4l2_test continuous 10" || TEST_RESULTS=$?
            
            # Test standard resolution profiles
            run_remote_tests "V4L2 resolution profiles" "VIDEO_DEVICE=$VIDEO_DEVICE ./v4l2_test profiles" || TEST_RESULTS=$?
        else
            print_warning "v4l2_test binary not found, skipping V4L2 tests"
        fi
        ;;&
        
    all|bin)
        # Test the main flight monitor binary
        if [ -f "$PROJECT_ROOT/target/$TARGET/release/flight_monitor" ]; then
            print_info "Testing flight_monitor binary startup..."
            # Start it and kill it after 5 seconds to ensure it starts correctly
            if ssh $REMOTE_HOST "cd $REMOTE_TEMP_DIR && timeout 5 ./flight_monitor || [ \$? -eq 124 ]"; then
                print_success "flight_monitor starts successfully"
            else
                print_error "flight_monitor failed to start"
                TEST_RESULTS=1
            fi
        else
            print_warning "flight_monitor binary not found"
        fi
        ;;
        
    *)
        print_error "Unknown test mode: $TEST_MODE"
        exit 1
        ;;
esac

# Step 5: Retrieve test results and artifacts
print_info "Retrieving test artifacts..."
mkdir -p "$SCRIPT_DIR/remote-test-results"

# Copy any generated files back (like captured images)
ssh $REMOTE_HOST "cd $REMOTE_TEMP_DIR && ls *.raw *.png *.jpg 2>/dev/null" | while read -r file; do
    if [ -n "$file" ]; then
        print_info "Retrieving $file..."
        scp "$REMOTE_HOST:$REMOTE_TEMP_DIR/$file" "$SCRIPT_DIR/remote-test-results/" || true
    fi
done

# Step 6: Cleanup
if [ "$KEEP_REMOTE" = false ]; then
    print_info "Cleaning up remote test directory..."
    ssh $REMOTE_HOST "rm -rf $REMOTE_TEMP_DIR"
else
    print_warning "Keeping remote test directory: $REMOTE_TEMP_DIR"
fi

# Final report
echo ""
echo "====================================="
if [ $TEST_RESULTS -eq 0 ]; then
    print_success "All tests completed successfully!"
else
    print_error "Some tests failed. Check the output above."
fi
echo "====================================="

if [ -d "$SCRIPT_DIR/remote-test-results" ] && [ "$(ls -A $SCRIPT_DIR/remote-test-results)" ]; then
    print_info "Test artifacts saved to: $SCRIPT_DIR/remote-test-results/"
fi

exit $TEST_RESULTS
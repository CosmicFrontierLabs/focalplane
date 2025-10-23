#!/bin/bash
set -euo pipefail

# Install PlayerOne Camera SDK on Jetson Orin
# This script downloads and installs the PlayerOne SDK system-wide

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

# Configuration
SDK_VERSION="3.9.0"
SDK_URL="https://player-one-astronomy.com/download/softwares/PlayerOne_Camera_SDK_Linux_V${SDK_VERSION}.tar.gz"
TEMP_DIR="/tmp/playerone-sdk-install"
INSTALL_LIB_DIR="/usr/local/lib"
INSTALL_INCLUDE_DIR="/usr/local/include"

print_info "Installing PlayerOne Camera SDK v${SDK_VERSION} on Jetson Orin"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    print_error "This script is for ARM64/aarch64 architecture (Jetson Orin)"
    print_error "Detected architecture: $ARCH"
    exit 1
fi
print_success "Architecture check passed: $ARCH"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    print_error "This script must be run with sudo"
    exit 1
fi

# Create temporary directory
print_info "Creating temporary directory..."
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Download SDK
print_info "Downloading PlayerOne SDK v${SDK_VERSION}..."
if [ -f "PlayerOne_Camera_SDK_Linux_V${SDK_VERSION}.tar.gz" ]; then
    print_warning "SDK archive already exists, using cached version"
else
    wget -O "PlayerOne_Camera_SDK_Linux_V${SDK_VERSION}.tar.gz" "$SDK_URL"
    print_success "SDK downloaded"
fi

# Extract SDK
print_info "Extracting SDK..."
tar -xzf "PlayerOne_Camera_SDK_Linux_V${SDK_VERSION}.tar.gz"
print_success "SDK extracted"

# Find the extracted directory
SDK_DIR=$(find . -maxdepth 1 -type d -name "PlayerOne*" | head -1)
if [ -z "$SDK_DIR" ]; then
    print_error "Could not find extracted SDK directory"
    exit 1
fi
print_info "Found SDK directory: $SDK_DIR"

# Install ARM64 libraries
print_info "Installing ARM64 libraries to $INSTALL_LIB_DIR..."
if [ -d "$SDK_DIR/lib/arm64" ] || [ -d "$SDK_DIR/lib/aarch64" ] || [ -d "$SDK_DIR/lib/armv8" ]; then
    # Try different possible directory names
    for lib_dir in "$SDK_DIR/lib/arm64" "$SDK_DIR/lib/aarch64" "$SDK_DIR/lib/armv8"; do
        if [ -d "$lib_dir" ]; then
            cp -v "$lib_dir"/*.so* "$INSTALL_LIB_DIR/" 2>/dev/null || true
            print_success "Libraries copied from $lib_dir"
            break
        fi
    done
else
    print_error "Could not find ARM64 library directory in SDK"
    print_info "Available directories:"
    ls -la "$SDK_DIR/lib/" || true
    exit 1
fi

# Install header files
print_info "Installing header files to $INSTALL_INCLUDE_DIR..."
if [ -d "$SDK_DIR/include" ]; then
    cp -v "$SDK_DIR/include"/*.h "$INSTALL_INCLUDE_DIR/" 2>/dev/null || true
    print_success "Header files copied"
else
    print_warning "No include directory found, skipping headers"
fi

# Install udev rules if present
if [ -d "$SDK_DIR/udev" ]; then
    print_info "Installing udev rules..."
    cp -v "$SDK_DIR/udev"/*.rules /etc/udev/rules.d/ 2>/dev/null || true
    udevadm control --reload-rules
    udevadm trigger
    print_success "udev rules installed"
fi

# Update library cache
print_info "Updating library cache..."
ldconfig
print_success "Library cache updated"

# Verify installation
print_info "Verifying installation..."
if ldconfig -p | grep -q libPlayerOneCamera; then
    print_success "PlayerOne Camera SDK successfully installed!"
    ldconfig -p | grep PlayerOneCamera
else
    print_error "Installation verification failed - library not found in ldconfig"
    exit 1
fi

# Cleanup
print_info "Cleaning up temporary files..."
cd /
rm -rf "$TEMP_DIR"
print_success "Cleanup complete"

print_success "Installation complete!"
print_info "You can now run camera_server without setting LD_LIBRARY_PATH"

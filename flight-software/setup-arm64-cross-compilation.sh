#!/bin/bash
set -euo pipefail

echo "Setting up ARM64 cross-compilation for Rust..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if rustup is installed
if ! command -v rustup &> /dev/null; then
    print_error "rustup is not installed. Please install Rust first:"
    echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

print_status "Adding ARM64 target to Rust toolchain..."
rustup target add aarch64-unknown-linux-gnu

# Detect OS and install cross-compilation toolchain
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Detected Linux - installing ARM64 cross-compilation tools..."
    
    # Check package manager and install accordingly
    if command -v apt-get &> /dev/null; then
        print_status "Using apt package manager..."
        sudo apt-get update
        sudo apt-get install -y gcc-aarch64-linux-gnu libc6-dev-arm64-cross
        
        # Set up pkg-config for cross-compilation
        export PKG_CONFIG_ALLOW_CROSS=1
        export PKG_CONFIG_PATH_aarch64_unknown_linux_gnu=/usr/lib/aarch64-linux-gnu/pkgconfig
        
    elif command -v yum &> /dev/null; then
        print_status "Using yum package manager..."
        sudo yum install -y gcc-aarch64-linux-gnu glibc-devel.aarch64
        
    elif command -v dnf &> /dev/null; then
        print_status "Using dnf package manager..."
        sudo dnf install -y gcc-aarch64-linux-gnu glibc-devel.aarch64
        
    elif command -v pacman &> /dev/null; then
        print_status "Using pacman package manager..."
        sudo pacman -S --needed aarch64-linux-gnu-gcc
        
    else
        print_warning "Unknown package manager. You may need to install ARM64 cross-compilation tools manually."
        print_warning "Required: gcc-aarch64-linux-gnu and ARM64 libc development files"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Detected macOS - ARM64 cross-compilation supported natively"
    print_warning "If you encounter linking issues, you may need to install additional tools via Homebrew"
    
else
    print_warning "Unsupported OS: $OSTYPE"
    print_warning "You may need to set up cross-compilation manually"
fi

# Create cargo config for cross-compilation
print_status "Creating cargo configuration for ARM64 cross-compilation..."
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[env]
PKG_CONFIG_ALLOW_CROSS = "1"
PKG_CONFIG_PATH_aarch64_unknown_linux_gnu = "/usr/lib/aarch64-linux-gnu/pkgconfig"

# For OpenSSL cross-compilation
OPENSSL_DIR_aarch64_unknown_linux_gnu = "/usr/lib/aarch64-linux-gnu"
OPENSSL_LIB_DIR_aarch64_unknown_linux_gnu = "/usr/lib/aarch64-linux-gnu"
OPENSSL_INCLUDE_DIR_aarch64_unknown_linux_gnu = "/usr/include"
EOF

print_status "Cargo configuration created at .cargo/config.toml"

# Test the cross-compilation setup
print_status "Testing ARM64 cross-compilation..."
if cargo build --target aarch64-unknown-linux-gnu --bin flight_monitor --release; then
    print_status "✓ ARM64 cross-compilation successful!"
    print_status "Binary location: target/aarch64-unknown-linux-gnu/release/flight_monitor"
    
    # Verify the binary
    print_status "Verifying ARM64 binary..."
    file target/aarch64-unknown-linux-gnu/release/flight_monitor
else
    print_error "✗ ARM64 cross-compilation failed"
    print_warning "This usually means missing ARM64 cross-compilation tools."
    print_warning ""
    print_warning "Alternative approaches:"
    print_warning "1. Use Docker multi-arch build (see Dockerfile.flight-monitor-arm64)"
    print_warning "2. Build natively on Jetson Orin device"
    print_warning "3. Install cross-compilation tools with: sudo apt install gcc-aarch64-linux-gnu"
    print_warning ""
    print_warning "For Docker approach, run: docker buildx build --platform linux/arm64 ."
fi

print_status "Cross-compilation setup complete!"
echo ""
echo "To build for ARM64 in the future, use:"
echo "  cargo build --target aarch64-unknown-linux-gnu --release"
echo ""
echo "To build for ARM64 with specific features:"
echo "  cargo build --target aarch64-unknown-linux-gnu --release --features your-feature"
echo ""
echo "Binary will be located at:"
echo "  target/aarch64-unknown-linux-gnu/release/flight_monitor"
#!/bin/bash
set -euo pipefail

# Setup self-hosted GitHub Actions runner on ARM64 build server
#
# This script installs and configures a GitHub Actions runner as a systemd user service.
# The runner will automatically start on boot and restart on failure.
#
# Prerequisites:
#   - Run this on cfl-test-bench (ARM64 build server)
#   - Have a GitHub Personal Access Token or use the web UI to get a runner token
#
# Usage:
#   ./setup-github-runner.sh
#
# The script will prompt for the runner registration token.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Configuration
RUNNER_DIR="$HOME/actions-runner"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)}"
RUNNER_LABELS="arm64-builder,cfl-test-bench"
REPO_URL="https://github.com/CosmicFrontierLabs/meter-sim"
RUNNER_VERSION="2.331.0"  # Check https://github.com/actions/runner/releases for latest

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

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    print_error "This script is for ARM64 (aarch64) systems only"
    print_error "Detected architecture: $ARCH"
    exit 1
fi

print_info "Setting up GitHub Actions runner on $(hostname)"
print_info "Architecture: $ARCH"
print_info "Runner directory: $RUNNER_DIR"
print_info "Runner name: $RUNNER_NAME"
print_info "Runner labels: $RUNNER_LABELS"
echo ""

# Step 1: Check/install system dependencies
print_info "Checking system dependencies..."
MISSING_PACKAGES=()

check_package() {
    if ! dpkg -l "$1" 2>/dev/null | grep -q "^ii"; then
        MISSING_PACKAGES+=("$1")
    fi
}

# Build dependencies (same as build-remote.sh)
check_package "libusb-1.0-0-dev"
check_package "libssl-dev"
check_package "pkg-config"
check_package "libclang-dev"
check_package "clang"
check_package "libsdl2-dev"
check_package "libsdl2-image-dev"
check_package "libzmq3-dev"
check_package "libfontconfig1-dev"

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_warning "Missing packages: ${MISSING_PACKAGES[*]}"
    print_info "Installing missing packages (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
    print_success "Dependencies installed"
else
    print_success "All system dependencies already installed"
fi

# Step 2: Check Rust installation
print_info "Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    print_warning "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    print_success "Rust installed"
else
    print_success "Rust already installed: $(rustc --version)"
fi

# Step 3: Create runner directory
print_info "Creating runner directory..."
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Step 4: Download runner if not present or version mismatch
RUNNER_TAR="actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
if [ ! -f "./config.sh" ]; then
    print_info "Downloading GitHub Actions runner v${RUNNER_VERSION}..."
    curl -o "$RUNNER_TAR" -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TAR}"
    tar xzf "$RUNNER_TAR"
    rm "$RUNNER_TAR"
    print_success "Runner downloaded and extracted"
else
    print_success "Runner already downloaded"
fi

# Step 5: Get registration token
echo ""
print_info "To register the runner, you need a registration token from GitHub."
print_info ""
print_info "Get the token by visiting:"
print_info "  ${REPO_URL}/settings/actions/runners/new"
print_info ""
print_info "Or use the GitHub CLI:"
print_info "  gh api repos/CosmicFrontierLabs/meter-sim/actions/runners/registration-token -X POST --jq '.token'"
print_info ""

# Check if already configured
if [ -f ".runner" ]; then
    print_warning "Runner already configured. Remove .runner to reconfigure."
    read -p "Do you want to reconfigure? (y/N): " RECONFIGURE
    if [ "$RECONFIGURE" != "y" ] && [ "$RECONFIGURE" != "Y" ]; then
        print_info "Skipping configuration..."
    else
        ./config.sh remove
        rm -f .runner .credentials .credentials_rsaparams
    fi
fi

if [ ! -f ".runner" ]; then
    read -p "Enter your runner registration token: " RUNNER_TOKEN
    if [ -z "$RUNNER_TOKEN" ]; then
        print_error "No token provided. Exiting."
        exit 1
    fi

    print_info "Configuring runner..."
    ./config.sh \
        --url "$REPO_URL" \
        --token "$RUNNER_TOKEN" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS" \
        --unattended \
        --replace

    print_success "Runner configured"
fi

# Step 6: Set up systemd user service
print_info "Setting up systemd user service..."

mkdir -p "$HOME/.config/systemd/user"

cat > "$HOME/.config/systemd/user/github-runner.service" << EOF
[Unit]
Description=GitHub Actions Runner
After=network.target

[Service]
Type=simple
WorkingDirectory=$RUNNER_DIR
ExecStart=$RUNNER_DIR/run.sh
Restart=always
RestartSec=10
Environment=RUNNER_ALLOW_RUNASROOT=false
Environment=PATH=/usr/local/bin:/usr/bin:/bin:$HOME/.cargo/bin

[Install]
WantedBy=default.target
EOF

print_success "Systemd service file created"

# Step 7: Enable lingering (allows user services to run without login)
print_info "Enabling lingering for user $(whoami)..."
sudo loginctl enable-linger "$(whoami)"
print_success "Lingering enabled"

# Step 8: Enable and start service
print_info "Enabling and starting runner service..."
systemctl --user daemon-reload
systemctl --user enable github-runner.service
systemctl --user start github-runner.service

# Wait a moment for service to start
sleep 3

# Step 9: Check status
print_info "Checking runner status..."
if systemctl --user is-active --quiet github-runner.service; then
    print_success "Runner is running!"
else
    print_error "Runner failed to start. Check logs with:"
    print_error "  journalctl --user -u github-runner.service -f"
    exit 1
fi

echo ""
print_success "============================================"
print_success "GitHub Actions runner setup complete!"
print_success "============================================"
echo ""
print_info "Runner name: $RUNNER_NAME"
print_info "Runner labels: self-hosted, Linux, ARM64, $RUNNER_LABELS"
print_info "Runner directory: $RUNNER_DIR"
echo ""
print_info "Useful commands:"
print_info "  Check status:    systemctl --user status github-runner.service"
print_info "  View logs:       journalctl --user -u github-runner.service -f"
print_info "  Stop runner:     systemctl --user stop github-runner.service"
print_info "  Start runner:    systemctl --user start github-runner.service"
print_info "  Restart runner:  systemctl --user restart github-runner.service"
echo ""
print_info "The runner will automatically start on boot."
print_info "Check the runner at: ${REPO_URL}/settings/actions/runners"

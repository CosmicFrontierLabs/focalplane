#!/bin/bash
set -euo pipefail

# Deploy calibrate_serve to cfl-test-bench
#
# Usage:
#   ./deploy-calibrate-test-bench.sh           # Update: build, sync, restart service
#   ./deploy-calibrate-test-bench.sh --setup   # Full setup: build, sync, install service, enable on boot

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
REMOTE_HOST="meawoppl@cfl-test-bench.tail944341.ts.net"
PROJECT_NAME="$(basename "$(git -C "$PROJECT_ROOT" remote get-url origin)" .git)"
REMOTE_BUILD_DIR="rust-builds/$PROJECT_NAME"
SERVICE_NAME="calibrate-serve"
SETUP_MODE=false

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
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy calibrate_serve to cfl-test-bench."
    echo ""
    echo "Modes:"
    echo "  (default)     Update: build and restart service"
    echo "  --setup       Full setup: build, install systemd service"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help"
    echo ""
    echo "Note: Frontend assets are now embedded in the binary at compile time."
    echo "      No separate frontend build or sync is needed."
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            SETUP_MODE=true
            shift
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

# Step 1: Build calibrate_serve on cfl-test-bench
# Note: Frontend assets are embedded in the binary at compile time,
# so we need to build the frontend first before compiling the server.
print_info "Building frontend WASM files locally (for embedding)..."
if ! command -v trunk &> /dev/null; then
    print_error "trunk not found. Install with: cargo install --locked trunk"
    exit 1
fi

"$SCRIPT_DIR/build-yew-frontends.sh"

# Step 2: Build calibrate_serve on cfl-test-bench
# The build-remote.sh script syncs the source including frontend dist files,
# which are then embedded into the binary during compilation.
print_info "Building calibrate_serve on cfl-test-bench..."
"$SCRIPT_DIR/build-remote.sh" --package test-bench --binary calibrate_serve --test-bench --features sdl2

if [ "$SETUP_MODE" = true ]; then
    # Full setup: install systemd service
    print_info "Installing systemd service..."

    # cfl-test-bench uses X11 on the RTX 4060 (card1 HDMI-A-1 at 2560x2560)
    SERVICE_FILE=$(cat <<EOF
[Unit]
Description=Calibration Pattern Server
After=network.target graphical.target
Wants=graphical.target

[Service]
Type=simple
User=meawoppl
WorkingDirectory=/home/meawoppl/$REMOTE_BUILD_DIR
Environment=DISPLAY=:0
Environment=SDL_VIDEODRIVER=x11
Environment=RUST_LOG=info
ExecStart=/home/meawoppl/$REMOTE_BUILD_DIR/target/release/calibrate_serve --wait-for-oled --ftdi-device 0 --lock-display
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
EOF
)

    # Write service file to test-bench
    echo "$SERVICE_FILE" | ssh "$REMOTE_HOST" "cat > /tmp/${SERVICE_NAME}.service"

    # Install service (requires sudo - user will be prompted)
    print_info "Installing service file (may require password)..."
    ssh "$REMOTE_HOST" "sudo mv /tmp/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service"
    ssh "$REMOTE_HOST" "sudo systemctl daemon-reload"
    ssh "$REMOTE_HOST" "sudo systemctl enable ${SERVICE_NAME}.service"
    print_success "Service installed and enabled"

    # Set up port 80 redirect (so we can access via http://host/ instead of :3001)
    print_info "Setting up port 80 redirect to 3001..."
    ssh "$REMOTE_HOST" "sudo apt-get install -y iptables-persistent" 2>/dev/null || true

    # Add iptables rules for port 80 -> 3001 redirect
    ssh "$REMOTE_HOST" "sudo iptables -t nat -C PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3001 2>/dev/null || sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3001"
    ssh "$REMOTE_HOST" "sudo iptables -t nat -C OUTPUT -o lo -p tcp --dport 80 -j REDIRECT --to-port 3001 2>/dev/null || sudo iptables -t nat -A OUTPUT -o lo -p tcp --dport 80 -j REDIRECT --to-port 3001"

    # Persist iptables rules
    ssh "$REMOTE_HOST" "sudo netfilter-persistent save" 2>/dev/null || true
    print_success "Port 80 redirect configured (port 80 -> 3001)"
fi

# Step 4: Restart the service
print_info "Restarting ${SERVICE_NAME} service..."
ssh "$REMOTE_HOST" "sudo systemctl restart ${SERVICE_NAME}.service" || {
    print_warning "Service restart failed (may not be installed yet)"
    print_info "Run with --setup to install the service"
}

# Step 5: Check status
sleep 2
print_info "Service status:"
ssh "$REMOTE_HOST" "sudo systemctl status ${SERVICE_NAME}.service --no-pager" || true

print_success "Deployment complete!"
if [ "$SETUP_MODE" = true ]; then
    print_info "The calibrate_serve binary will now start automatically on boot"
    print_info "OLED wait mode enabled: will poll until 2560x2560 display is detected"
    print_info "Port 80 redirect enabled: access via http://cfl-test-bench/"
fi
print_info ""
print_info "Useful commands:"
print_info "  Check status:  ssh $REMOTE_HOST 'sudo systemctl status ${SERVICE_NAME}'"
print_info "  View logs:     ssh $REMOTE_HOST 'sudo journalctl -u ${SERVICE_NAME} -f'"
print_info "  Restart:       ssh $REMOTE_HOST 'sudo systemctl restart ${SERVICE_NAME}'"
print_info "  Stop:          ssh $REMOTE_HOST 'sudo systemctl stop ${SERVICE_NAME}'"

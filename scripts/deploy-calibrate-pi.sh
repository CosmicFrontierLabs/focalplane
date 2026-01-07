#!/bin/bash
set -euo pipefail

# Deploy calibrate_serve to test-bench-pi
#
# Usage:
#   ./deploy-calibrate-pi.sh           # Update: build, sync, restart service
#   ./deploy-calibrate-pi.sh --setup   # Full setup: build, sync, install service, enable on boot

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
REMOTE_HOST="meawoppl@test-bench-pi.tail944341.ts.net"
REMOTE_BUILD_DIR="rust-builds/meter-sim"
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
    echo "Deploy calibrate_serve to test-bench-pi."
    echo ""
    echo "Modes:"
    echo "  (default)     Update: build, sync frontend, restart service"
    echo "  --setup       Full setup: build, sync, install systemd service"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help"
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

# Step 1: Build calibrate_serve on the Pi
print_info "Building calibrate_serve on test-bench-pi..."
"$SCRIPT_DIR/build-remote.sh" --package test-bench --binary calibrate_serve --test-bench-pi --features sdl2

# Step 2: Build frontend locally (requires trunk)
print_info "Building frontend WASM files locally..."
if ! command -v trunk &> /dev/null; then
    print_error "trunk not found. Install with: cargo install --locked trunk"
    exit 1
fi

"$SCRIPT_DIR/build-yew-frontends.sh"

# Step 3: Sync frontend files to Pi
print_info "Syncing frontend files to Pi..."
rsync -avz \
    "$PROJECT_ROOT/test-bench-frontend/dist/calibrate/" \
    "$REMOTE_HOST:~/$REMOTE_BUILD_DIR/test-bench-frontend/dist/calibrate/"
print_success "Frontend files synced"

if [ "$SETUP_MODE" = true ]; then
    # Full setup: install systemd service
    print_info "Installing systemd service..."

    SERVICE_FILE=$(cat <<'EOF'
[Unit]
Description=Calibration Pattern Server
After=network.target graphical.target
Wants=graphical.target

[Service]
Type=simple
User=meawoppl
WorkingDirectory=/home/meawoppl/rust-builds/meter-sim
Environment=DISPLAY=:0
Environment=RUST_LOG=info
ExecStart=/home/meawoppl/rust-builds/meter-sim/target/release/calibrate_serve --wait-for-oled
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
EOF
)

    # Write service file to Pi
    echo "$SERVICE_FILE" | ssh "$REMOTE_HOST" "cat > /tmp/${SERVICE_NAME}.service"

    # Install service (requires sudo - user will be prompted)
    print_info "Installing service file (may require password)..."
    ssh "$REMOTE_HOST" "sudo mv /tmp/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service"
    ssh "$REMOTE_HOST" "sudo systemctl daemon-reload"
    ssh "$REMOTE_HOST" "sudo systemctl enable ${SERVICE_NAME}.service"
    print_success "Service installed and enabled"
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
fi
print_info ""
print_info "Useful commands:"
print_info "  Check status:  ssh $REMOTE_HOST 'sudo systemctl status ${SERVICE_NAME}'"
print_info "  View logs:     ssh $REMOTE_HOST 'sudo journalctl -u ${SERVICE_NAME} -f'"
print_info "  Restart:       ssh $REMOTE_HOST 'sudo systemctl restart ${SERVICE_NAME}'"
print_info "  Stop:          ssh $REMOTE_HOST 'sudo systemctl stop ${SERVICE_NAME}'"

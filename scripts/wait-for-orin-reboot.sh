#!/bin/bash
set -euo pipefail

# Wait for Orin to complete reboot and become accessible via SSH

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_waiting() { echo -e "${YELLOW}[WAITING]${NC} $1"; }

ORIN_HOST="${ORIN_HOST:-meawoppl@orin-nano.tail944341.ts.net}"
MAX_ATTEMPTS=60
SLEEP_INTERVAL=5

print_info "Waiting for Orin to reboot and become accessible..."
print_info "Will check every ${SLEEP_INTERVAL}s for up to $((MAX_ATTEMPTS * SLEEP_INTERVAL))s"

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))

    if ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no "$ORIN_HOST" "echo 'System ready' && uname -a" &>/dev/null; then
        print_success "Orin is back online!"
        ssh "$ORIN_HOST" "echo 'System info:' && uname -a && uptime"
        exit 0
    else
        print_waiting "Attempt $attempt/$MAX_ATTEMPTS - Orin not responding yet..."
        sleep $SLEEP_INTERVAL
    fi
done

print_error "Timeout: Orin did not come back online within $((MAX_ATTEMPTS * SLEEP_INTERVAL))s"
exit 1

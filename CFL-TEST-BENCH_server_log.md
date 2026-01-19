# CFL-TEST-BENCH Server Configuration Log

## Change Log

### 2026-01-19
- Installed NVIDIA GeForce RTX 4060 GPU
- Created deploy script for calibrate_serve (`scripts/deploy-calibrate-test-bench.sh`)
- Removed old Pi deployment script (`scripts/deploy-calibrate-pi.sh`)
- Configured systemd service for calibrate-serve
- Set up port 80 redirect to 3001
- Documented auto-login requirement for graphical display access

## Machine Details
- **Hostname:** cfl-test-bench.tail944341.ts.net
- **SSH Access:** `ssh meawoppl@cfl-test-bench.tail944341.ts.net`
- **Architecture:** ARM64 (Ampere Altra processor)
- **OS:** Ubuntu with GNOME/GDM (Wayland)

## Hardware

### GPUs (3 total)
- NVIDIA GeForce RTX 4060 (AD107) - card1
- NVIDIA device 25b2 - card0
- NVIDIA GeForce GT 710 - card2

### Connected Displays
- **card0 VGA-1:** 1024x768 (ASPEED BMC - server management)
- **card1 HDMI-A-1:** 2560x2560 (RTX 4060 - OLED display for calibration)

### Other Hardware
- Kingston KC3000/FURY Renegade NVMe SSD
- Intel I210 Gigabit Ethernet
- Intel X550 10GbE (dual port)
- Intel Wi-Fi 6E AX210
- ASMedia USB 3.2 Gen 1 controller

## Calibration Service Setup

### Step 1: Create the Service File

Create the file `/etc/systemd/system/calibrate-serve.service` with the following contents:

```ini
[Unit]
Description=Calibration Pattern Server
After=network.target graphical.target
Wants=graphical.target

[Service]
Type=simple
User=meawoppl
WorkingDirectory=/home/meawoppl/rust-builds/meter-sim
Environment=DISPLAY=:0
Environment=SDL_VIDEODRIVER=x11
Environment=RUST_LOG=info
ExecStart=/home/meawoppl/rust-builds/meter-sim/target/release/calibrate_serve --wait-for-oled
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
```

**Service file explanation:**
- `After=network.target graphical.target` - Wait for network and graphical session before starting
- `Wants=graphical.target` - Prefer to have graphical session available
- `User=meawoppl` - Run as meawoppl user (must match the logged-in graphical session user)
- `Environment=DISPLAY=:0` - Connect to X display :0 (may need adjustment based on actual display)
- `Environment=SDL_VIDEODRIVER=x11` - Force SDL2 to use X11 backend
- `--wait-for-oled` - Binary will poll until 2560x2560 display is detected before rendering

### Step 2: Install and Enable the Service

```bash
# Write the service file (if created in /tmp first)
sudo mv /tmp/calibrate-serve.service /etc/systemd/system/calibrate-serve.service

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable calibrate-serve.service

# Start the service now
sudo systemctl start calibrate-serve.service
```

### Step 3: Port 80 Redirect (Optional)

The calibrate_serve binary listens on port 3001 by default. To allow accessing the web UI at `http://cfl-test-bench/` instead of `http://cfl-test-bench:3001/`, set up an iptables redirect:

```bash
# Install iptables-persistent to save rules across reboots
sudo apt-get install -y iptables-persistent

# Redirect external traffic on port 80 to port 3001
# -t nat: Use the NAT table
# -A PREROUTING: Append to PREROUTING chain (for incoming packets)
# -p tcp --dport 80: Match TCP packets destined for port 80
# -j REDIRECT --to-port 3001: Redirect to port 3001
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3001

# Redirect localhost traffic on port 80 to port 3001
# -A OUTPUT: Append to OUTPUT chain (for locally-generated packets)
# -o lo: Match packets going to loopback interface
sudo iptables -t nat -A OUTPUT -o lo -p tcp --dport 80 -j REDIRECT --to-port 3001

# Save the rules so they persist across reboots
sudo netfilter-persistent save
```

**To verify the rules are in place:**
```bash
sudo iptables -t nat -L -n -v
```

**To remove the rules if needed:**
```bash
sudo iptables -t nat -D PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3001
sudo iptables -t nat -D OUTPUT -o lo -p tcp --dport 80 -j REDIRECT --to-port 3001
sudo netfilter-persistent save
```

## Required Configuration: Auto-Login

The calibrate_serve binary requires access to a graphical display. Since the machine runs Wayland/GNOME, a user must be logged into a graphical session for the service to work.

### Enable Auto-Login for meawoppl
Edit `/etc/gdm3/custom.conf`:

```ini
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=meawoppl
```

Then reboot:
```bash
sudo reboot
```

## Service Management Commands
```bash
# Check status
sudo systemctl status calibrate-serve

# View logs
sudo journalctl -u calibrate-serve -f

# Restart service
sudo systemctl restart calibrate-serve

# Stop service
sudo systemctl stop calibrate-serve
```

## Build Locations
- **Binary:** `~/rust-builds/meter-sim/target/release/calibrate_serve`
- **Frontend:** `~/rust-builds/meter-sim/test-bench-frontend/dist/calibrate/`

## Deployment Script
Use `scripts/deploy-calibrate-test-bench.sh` from the meter-sim repo:

```bash
# Update only (rebuild and restart)
./scripts/deploy-calibrate-test-bench.sh

# Full setup (install systemd service)
./scripts/deploy-calibrate-test-bench.sh --setup
```

## Troubleshooting

### "x11 not available" Error
This means no graphical session is available. Ensure:
1. meawoppl is logged into GNOME desktop (auto-login recommended)
2. The DISPLAY environment variable is correct

Check active sessions:
```bash
loginctl list-sessions
```

### Check Display Configuration
```bash
# List connected displays
for card in /sys/class/drm/card*/card*-*; do
  echo "$card: $(cat $card/status) - $(cat $card/modes | head -1)"
done
```

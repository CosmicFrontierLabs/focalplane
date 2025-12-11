# Orin Monitor Hardware Access Setup

This document explains the system permissions and configuration needed to run `orin_monitor` on Jetson Orin hardware.

## Hardware Interfaces Used

The `orin_monitor` binary reads directly from kernel interfaces to collect telemetry:

1. **INA3221 Power Monitors** - I2C sensors for power rail monitoring
   - Path: `/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/`
   - Requires: Read access to sysfs I2C devices

2. **Thermal Zones** - Temperature sensors
   - Path: `/sys/class/thermal/thermal_zone*/temp`
   - Requires: Read access to sysfs thermal devices

3. **GPIO Controller** - For GPIO pin control (via gpiod)
   - Path: `/dev/gpiochip*`
   - Requires: Read/write access to GPIO character devices

4. **System Stats** - CPU, memory, uptime
   - Paths: `/proc/uptime`, `/proc/meminfo`, etc.
   - Requires: Read access to procfs (typically world-readable)

## Permission Requirements

### Option 1: Run as Root (Not Recommended for Production)

```bash
sudo ./orin_monitor
```

**Pros:** Simple, works immediately
**Cons:** Security risk, violates principle of least privilege

### Option 2: Add User to Required Groups (Recommended)

The monitoring user needs access to I2C, GPIO, and hardware monitoring devices.

```bash
# Add user to gpio and i2c groups
sudo usermod -a -G gpio,i2c,dialout $USER

# Log out and back in for group changes to take effect
# Or use: newgrp gpio
```

### Option 3: Custom udev Rules (Most Secure)

Create a dedicated group and udev rules for hardware monitoring:

```bash
# Create a hardware monitoring group
sudo groupadd hwmon
sudo usermod -a -G hwmon $USER
```

Create `/etc/udev/rules.d/99-jetson-hwmon.rules`:

```udev
# INA3221 power monitors - read-only access for hwmon group
SUBSYSTEM=="hwmon", KERNEL=="hwmon*", GROUP="hwmon", MODE="0640"

# I2C devices - read-only access for i2c group
SUBSYSTEM=="i2c-dev", GROUP="i2c", MODE="0660"

# GPIO character devices - read/write for gpio group
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"

# Thermal zones - read-only for hwmon group (usually already world-readable)
SUBSYSTEM=="thermal", GROUP="hwmon", MODE="0644"
```

Reload udev rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Verification

Verify permissions before running:

```bash
# Check INA3221 sensors are accessible
ls -l /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/

# Check thermal zones
ls -l /sys/class/thermal/thermal_zone*/temp

# Check GPIO devices
ls -l /dev/gpiochip*

# Check group memberships
groups $USER
```

Expected output should show your user has read access to all these paths.

## Running the Monitor

### Interactive Mode

```bash
# With environment variables
BIND_ADDRESS=0.0.0.0 PORT=9090 ./orin_monitor

# Or use defaults (0.0.0.0:9090)
./orin_monitor
```

### As a systemd Service

Create `/etc/systemd/system/orin-monitor.service`:

```ini
[Unit]
Description=Jetson Orin Hardware Monitoring Service
After=network.target

[Service]
Type=simple
User=cosmicfrontiers
Group=hwmon
SupplementaryGroups=gpio i2c

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/
ReadWritePaths=/dev/gpiochip0

# Allow access to required devices
DeviceAllow=/dev/gpiochip0 rw
DevicePolicy=closed

# Environment
Environment="BIND_ADDRESS=0.0.0.0"
Environment="PORT=9090"
Environment="RUST_LOG=info"

# Binary and working directory
ExecStart=/usr/local/bin/orin_monitor
WorkingDirectory=/opt/orin-monitor

# Restart policy
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable orin-monitor
sudo systemctl start orin-monitor

# Check status
sudo systemctl status orin-monitor

# View logs
journalctl -u orin-monitor -f
```

## Troubleshooting

### Permission Denied Errors

```
Error: Failed to open GPIO chip 'gpiochip0'
```

**Solution:** Add user to `gpio` group and verify `/dev/gpiochip*` permissions

```
Error: I/O error reading hardware sensors: Permission denied
```

**Solution:** Add user to `i2c` or `hwmon` group and check sysfs permissions

### No INA3221 Sensors Found

```
WARN No INA3221 sensors found - may not be running on Jetson Orin
```

**Possible causes:**
- Not running on Jetson Orin hardware
- I2C driver not loaded: `sudo modprobe ina3221`
- Device tree not configured correctly

Check if sensors are present:

```bash
ls -la /sys/bus/i2c/drivers/ina3221/
```

### GPIO Access Issues

If GPIO operations fail, verify the chip and line numbers:

```bash
# List available GPIO chips
gpiodetect

# List lines on chip 0
gpioinfo gpiochip0
```

The constants in `hardware::orin::gpio` are:
- `ORIN_GPIO_CHIP` = "gpiochip0"
- `ORIN_PX04_LINE` = 127

Verify these match your hardware configuration.

## Security Considerations

1. **Principle of Least Privilege**: Use Option 3 (udev rules) for production
2. **Network Binding**: Bind to `127.0.0.1` if only local access needed
3. **Firewall**: Restrict access to metrics endpoint (default port 9090)
4. **TLS**: Consider adding TLS termination via nginx/reverse proxy
5. **Authentication**: Prometheus scraping should use network-level security

## Hardware Specifications

### Supported Jetson Orin Variants

- **Jetson AGX Orin Series**: Dual INA3221 at I2C addresses 0x40, 0x41
- **Jetson Orin NX Series**: Single INA3221 at I2C address 0x40
- **Jetson Orin Nano Series**: Single INA3221 at I2C address 0x40

Each INA3221 provides 3 channels of voltage/current monitoring for different power rails.

## References

- [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/)
- [INA3221 Datasheet](https://www.ti.com/product/INA3221)
- [Linux GPIO Character Device Interface](https://www.kernel.org/doc/html/latest/driver-api/gpio/consumer.html)

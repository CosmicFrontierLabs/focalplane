# Flight Software - Jetson Orin Monitoring

Comprehensive monitoring system for NVIDIA Jetson Orin Industrial flight computers, providing real-time telemetry through Prometheus metrics and Grafana dashboards.

## Features

- **Power Rail Monitoring**: Real-time voltage, current, and power consumption via INA3221 sensors
- **Thermal Management**: Temperature monitoring across all thermal zones
- **System Resources**: CPU, memory, disk, and GPU utilization tracking
- **Health Monitoring**: Service health checks and collection error tracking
- **Prometheus Integration**: Standard metrics endpoint for time-series data
- **Grafana Dashboards**: Pre-configured visualization for flight computer telemetry

## Hardware Support

Designed specifically for NVIDIA Jetson Orin Industrial boards:
- Jetson AGX Orin Series (dual INA3221 at 0x40, 0x41)
- Jetson Orin NX Series (single INA3221 at 0x40) 
- Jetson Orin Nano Series (single INA3221 at 0x40)

## Quick Start

### Build and Run Locally

```bash
# Build the flight monitor
cargo build --release --bin flight_monitor

# Run with default settings
./target/release/flight_monitor

# Or with custom configuration
BIND_ADDRESS=0.0.0.0 PORT=9090 ./target/release/flight_monitor
```

### Docker Deployment

```bash
# Start the complete monitoring stack
docker-compose up -d

# View logs
docker-compose logs -f flight-monitor

# Stop the stack
docker-compose down
```

## Monitoring Endpoints

- **Metrics**: `http://localhost:9090/metrics` - Prometheus format metrics
- **Health**: `http://localhost:9090/health` - Service health status
- **Prometheus UI**: `http://localhost:9091` - Prometheus query interface
- **Grafana**: `http://localhost:3000` - Dashboard interface (admin/admin)

## Key Metrics

### Power Rails
- `jetson_power_rail_voltage_mv{rail_name}` - Rail voltage in millivolts
- `jetson_power_rail_current_ma{rail_name}` - Rail current in milliamperes  
- `jetson_power_rail_power_mw{rail_name}` - Rail power in milliwatts

### Thermal
- `jetson_thermal_temperature_celsius{zone_name}` - Temperature in Celsius

### System
- `jetson_cpu_usage_percent` - CPU utilization percentage
- `jetson_memory_usage_mb` - Memory usage in megabytes
- `jetson_memory_total_mb` - Total memory in megabytes
- `jetson_disk_usage_percent` - Disk usage percentage
- `jetson_gpu_usage_percent` - GPU utilization percentage
- `jetson_uptime_seconds` - System uptime in seconds

### Health
- `jetson_metrics_collection_total` - Total metrics collections
- `jetson_metrics_collection_errors_total` - Collection error count
- `jetson_last_collection_timestamp` - Last successful collection time

## Configuration

### Environment Variables

- `BIND_ADDRESS` - Server bind address (default: 0.0.0.0)
- `PORT` - Server port (default: 9090)
- `RUST_LOG` - Log level (default: info)

### Collection Intervals

- Metrics collection: 1 second
- Prometheus scraping: 1 second
- Health checks: 30 seconds

## Hardware Interface

The system reads directly from Linux sysfs interfaces:

### INA3221 Power Monitors
```
/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/
/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*/
├── in1_input, in2_input, in3_input     # Voltage readings
├── curr1_input, curr2_input, curr3_input # Current readings
└── in1_label, in2_label, in3_label     # Rail names
```

### Thermal Zones
```
/sys/class/thermal/thermal_zone*/temp
```

### System Statistics
- CPU usage via systemstat crate
- Memory info from /proc/meminfo equivalent
- Disk usage from filesystem stats
- Uptime from /proc/uptime

## Safety Notes

⚠️ **IMPORTANT**: Do not modify INA3221 sysfs values. This monitoring system is read-only to prevent hardware damage.

## Development

### Building
```bash
cargo build --release
```

### Testing
```bash
cargo test
```

### Linting
```bash
cargo clippy -- -W clippy::all
```

### Formatting
```bash
cargo fmt
```

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Jetson Orin   │    │    Flight    │    │ Prometheus  │
│   Hardware      │───▶│   Monitor    │───▶│   Server    │
│   (INA3221,     │    │   Service    │    │             │
│    Thermal)     │    │              │    │             │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │                     │
                              ▼                     ▼
                       ┌──────────────┐    ┌─────────────┐
                       │   Health     │    │   Grafana   │
                       │  Endpoint    │    │ Dashboard   │
                       └──────────────┘    └─────────────┘
```

## License

MIT License - See workspace root for details.
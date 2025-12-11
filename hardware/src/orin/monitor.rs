use super::errors::MonitoringError;
use glob::glob;
use serde::{Deserialize, Serialize};
use std::fs;
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerRailReading {
    pub name: String,
    pub voltage_mv: Option<f64>,
    pub current_ma: Option<f64>,
    pub power_mw: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalReading {
    pub name: String,
    pub temperature_celsius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub power_rails: Vec<PowerRailReading>,
    pub thermal_zones: Vec<ThermalReading>,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub memory_total_mb: f64,
    pub disk_usage_percent: f64,
    pub gpu_usage_percent: Option<f64>,
    pub uptime_seconds: u64,
}

#[derive(Clone)]
pub struct JetsonOrinMonitor {
    ina3221_paths: Vec<String>,
    thermal_paths: Vec<String>,
}

impl JetsonOrinMonitor {
    pub fn new() -> Result<Self, MonitoringError> {
        let ina3221_paths = Self::find_ina3221_sensors()?;
        let thermal_paths = Self::find_thermal_sensors()?;

        Ok(Self {
            ina3221_paths,
            thermal_paths,
        })
    }

    fn find_ina3221_sensors() -> Result<Vec<String>, MonitoringError> {
        let mut paths = Vec::new();

        // Look for INA3221 sensors in hwmon
        for entry in glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*")
            .map_err(|e| MonitoringError::SystemError(format!("Glob pattern error: {e}")))?
        {
            match entry {
                Ok(path) => {
                    if path.is_dir() {
                        paths.push(path.to_string_lossy().to_string());
                    }
                }
                Err(e) => {
                    tracing::warn!("Error accessing INA3221 path: {e}");
                }
            }
        }

        if paths.is_empty() {
            tracing::warn!("No INA3221 sensors found - may not be running on Jetson Orin");
        }

        Ok(paths)
    }

    fn find_thermal_sensors() -> Result<Vec<String>, MonitoringError> {
        let mut paths = Vec::new();

        // Look for thermal zones
        for entry in glob("/sys/class/thermal/thermal_zone*/temp")
            .map_err(|e| MonitoringError::SystemError(format!("Glob pattern error: {e}")))?
        {
            match entry {
                Ok(path) => paths.push(path.to_string_lossy().to_string()),
                Err(e) => {
                    tracing::warn!("Error accessing thermal zone: {e}");
                }
            }
        }

        Ok(paths)
    }

    pub async fn collect_metrics(&self) -> Result<SystemMetrics, MonitoringError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let power_rails = self.read_power_rails().await?;
        let thermal_zones = self.read_thermal_zones().await?;
        let (cpu_usage, memory_usage, memory_total, disk_usage) = self.read_system_stats().await?;
        let gpu_usage = self.read_gpu_usage().await;
        let uptime = self.read_uptime().await?;

        Ok(SystemMetrics {
            timestamp,
            power_rails,
            thermal_zones,
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: memory_usage,
            memory_total_mb: memory_total,
            disk_usage_percent: disk_usage,
            gpu_usage_percent: gpu_usage,
            uptime_seconds: uptime,
        })
    }

    async fn read_power_rails(&self) -> Result<Vec<PowerRailReading>, MonitoringError> {
        let mut rails = Vec::new();

        for hwmon_path in &self.ina3221_paths {
            // Read each channel (typically 3 channels per INA3221)
            for channel in 1..=3 {
                let name = self
                    .read_rail_name(hwmon_path, channel)
                    .unwrap_or_else(|_| {
                        format!(
                            "Rail_{}_Ch{}",
                            hwmon_path.split('/').next_back().unwrap_or("unknown"),
                            channel
                        )
                    });

                let voltage_mv = self
                    .read_sensor_value(hwmon_path, &format!("in{channel}_input"))
                    .ok();
                let current_ma = self
                    .read_sensor_value(hwmon_path, &format!("curr{channel}_input"))
                    .ok();

                let power_mw = match (voltage_mv, current_ma) {
                    (Some(v), Some(i)) => Some(v * i / 1000.0), // Convert from mV*mA to mW
                    _ => None,
                };

                rails.push(PowerRailReading {
                    name,
                    voltage_mv,
                    current_ma,
                    power_mw,
                });
            }
        }

        Ok(rails)
    }

    fn read_rail_name(&self, hwmon_path: &str, channel: usize) -> Result<String, MonitoringError> {
        let name_path = format!("{hwmon_path}/in{channel}_label");
        match fs::read_to_string(&name_path) {
            Ok(content) => Ok(content.trim().to_string()),
            Err(_) => {
                // Fallback to generic naming if label file doesn't exist
                Ok(format!("in{channel}"))
            }
        }
    }

    fn read_sensor_value(
        &self,
        hwmon_path: &str,
        sensor_file: &str,
    ) -> Result<f64, MonitoringError> {
        let sensor_path = format!("{hwmon_path}/{sensor_file}");
        let content = fs::read_to_string(&sensor_path)?;
        let value = content.trim().parse::<f64>().map_err(|e| {
            MonitoringError::ParseError(format!("Failed to parse {sensor_path}: {e}"))
        })?;
        Ok(value)
    }

    async fn read_thermal_zones(&self) -> Result<Vec<ThermalReading>, MonitoringError> {
        let mut thermal_readings = Vec::new();

        for temp_path in &self.thermal_paths {
            if let Ok(content) = fs::read_to_string(temp_path) {
                if let Ok(temp_millicelsius) = content.trim().parse::<f64>() {
                    let temp_celsius = temp_millicelsius / 1000.0;

                    // Extract zone name from path
                    let zone_name = temp_path
                        .split('/')
                        .nth_back(1)
                        .unwrap_or("unknown")
                        .to_string();

                    thermal_readings.push(ThermalReading {
                        name: zone_name,
                        temperature_celsius: temp_celsius,
                    });
                }
            }
        }

        Ok(thermal_readings)
    }

    async fn read_system_stats(&self) -> Result<(f64, f64, f64, f64), MonitoringError> {
        // Use sysinfo for cross-platform system metrics
        let mut sys = System::new_all();
        sys.refresh_all();

        // CPU usage (average across all cores)
        let cpu_usage = sys
            .cpus()
            .iter()
            .map(|cpu| cpu.cpu_usage() as f64)
            .sum::<f64>()
            / sys.cpus().len() as f64;

        // Memory info
        let memory_total_mb = sys.total_memory() as f64 / 1024.0 / 1024.0;
        let memory_usage_mb = sys.used_memory() as f64 / 1024.0 / 1024.0;

        // Simple disk usage fallback - read from /proc/mounts and statvfs if available
        // For now, return 0 for disk usage to avoid complex filesystem parsing
        let disk_usage = 0.0;

        Ok((cpu_usage, memory_usage_mb, memory_total_mb, disk_usage))
    }

    async fn read_gpu_usage(&self) -> Option<f64> {
        // Try to read GPU usage from tegrastats or nvml if available
        // For now, return None as this requires additional dependencies
        None
    }

    async fn read_uptime(&self) -> Result<u64, MonitoringError> {
        let uptime_content = fs::read_to_string("/proc/uptime")?;
        let uptime_seconds = uptime_content
            .split_whitespace()
            .next()
            .ok_or_else(|| MonitoringError::ParseError("Invalid uptime format".to_string()))?
            .parse::<f64>()
            .map_err(|e| MonitoringError::ParseError(format!("Failed to parse uptime: {e}")))?;

        Ok(uptime_seconds as u64)
    }
}

impl Default for JetsonOrinMonitor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            ina3221_paths: Vec::new(),
            thermal_paths: Vec::new(),
        })
    }
}

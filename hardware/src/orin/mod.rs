pub mod errors;
pub mod gpio;
pub mod monitor;

pub use errors::MonitoringError;
pub use gpio::{GpioController, ORIN_GPIO_CHIP, ORIN_PX04_LINE};
pub use monitor::{JetsonOrinMonitor, PowerRailReading, SystemMetrics, ThermalReading};

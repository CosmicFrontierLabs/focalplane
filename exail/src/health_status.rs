//! Health status register for Exail Asterix NS gyro

use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

bitflags! {
    /// Health status register for Exail Asterix NS (32 bits)
    ///
    /// Contains per-channel FOG status flags and communication error indicators.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct HealthStatus: u32 {
        // Channel X flags (bits 0-7)
        /// ADC saturation integration channel X
        const INTADCSAT_X = 1 << 0;
        /// ADC saturation channel X
        const INSADCSAT_X = 1 << 1;
        /// VPI error channel X
        const VPIOVF_X = 1 << 2;
        /// Dynamic overflow status channel X
        const OMEGAPI_X = 1 << 3;
        /// Measured power lower bound saturation channel X
        const POWMINSAT_X = 1 << 4;
        /// Measured power upper bound saturation channel X
        const POWMAXSAT_X = 1 << 5;
        // Bit 6 reserved
        /// FOG Channel X validity (1=valid, 0=error detected)
        const VALIDITY_X = 1 << 7;

        // Channel Y flags (bits 8-15)
        /// ADC saturation integration channel Y
        const INTADCSAT_Y = 1 << 8;
        /// ADC saturation channel Y
        const INSADCSAT_Y = 1 << 9;
        /// VPI error channel Y
        const VPIOVF_Y = 1 << 10;
        /// Dynamic overflow status channel Y
        const OMEGAPI_Y = 1 << 11;
        /// Measured power lower bound saturation channel Y
        const POWMINSAT_Y = 1 << 12;
        /// Measured power upper bound saturation channel Y
        const POWMAXSAT_Y = 1 << 13;
        // Bit 14 reserved
        /// FOG Channel Y validity (1=valid, 0=error detected)
        const VALIDITY_Y = 1 << 15;

        // Channel Z flags (bits 16-23)
        /// ADC saturation integration channel Z
        const INTADCSAT_Z = 1 << 16;
        /// ADC saturation channel Z
        const INSADCSAT_Z = 1 << 17;
        /// VPI error channel Z
        const VPIOVF_Z = 1 << 18;
        /// Dynamic overflow status channel Z
        const OMEGAPI_Z = 1 << 19;
        /// Measured power lower bound saturation channel Z
        const POWMINSAT_Z = 1 << 20;
        /// Measured power upper bound saturation channel Z
        const POWMAXSAT_Z = 1 << 21;
        // Bit 22 reserved
        /// FOG Channel Z validity (1=valid, 0=error detected)
        const VALIDITY_Z = 1 << 23;

        // Communication and global flags (bits 24-31)
        /// Parity error detected on Nominal channel (frame not rejected)
        const PARITY_ERR_NOM = 1 << 24;
        /// CRC error on Nominal channel (frame rejected)
        const CRC_ERR_NOM = 1 << 25;
        /// Parity error detected on Redundant channel (frame not rejected)
        const PARITY_ERR_RED = 1 << 26;
        /// CRC error on Redundant channel (frame rejected)
        const CRC_ERR_RED = 1 << 27;
        /// Parity error detected on Stimuli channel (frame not rejected)
        const PARITY_ERR_STM = 1 << 28;
        /// CRC error on Stimuli channel (frame rejected)
        const CRC_ERR_STM = 1 << 29;
        /// Gyro state: 0=GYRO_MODE, 1=BOOT_MODE
        const GYRO_STATE = 1 << 30;
        /// ASTRIX NS validity (1=all channels valid, 0=at least one channel invalid)
        const FOG_VALIDITY = 1 << 31;

        // Compound flags for convenience
        /// All X channel error flags
        const ERRORS_X = Self::INTADCSAT_X.bits() | Self::INSADCSAT_X.bits()
            | Self::VPIOVF_X.bits() | Self::OMEGAPI_X.bits()
            | Self::POWMINSAT_X.bits() | Self::POWMAXSAT_X.bits();
        /// All Y channel error flags
        const ERRORS_Y = Self::INTADCSAT_Y.bits() | Self::INSADCSAT_Y.bits()
            | Self::VPIOVF_Y.bits() | Self::OMEGAPI_Y.bits()
            | Self::POWMINSAT_Y.bits() | Self::POWMAXSAT_Y.bits();
        /// All Z channel error flags
        const ERRORS_Z = Self::INTADCSAT_Z.bits() | Self::INSADCSAT_Z.bits()
            | Self::VPIOVF_Z.bits() | Self::OMEGAPI_Z.bits()
            | Self::POWMINSAT_Z.bits() | Self::POWMAXSAT_Z.bits();
        /// All communication error flags
        const COMM_ERRORS = Self::PARITY_ERR_NOM.bits() | Self::CRC_ERR_NOM.bits()
            | Self::PARITY_ERR_RED.bits() | Self::CRC_ERR_RED.bits()
            | Self::PARITY_ERR_STM.bits() | Self::CRC_ERR_STM.bits();
    }
}

impl HealthStatus {
    /// Returns true if the gyro is in boot mode (not operational)
    pub fn is_boot_mode(&self) -> bool {
        self.contains(Self::GYRO_STATE)
    }

    /// Returns true if the gyro is in operational mode
    pub fn is_gyro_mode(&self) -> bool {
        !self.contains(Self::GYRO_STATE)
    }

    /// Returns true if the overall FOG validity bit is set (all channels valid)
    pub fn is_fog_valid(&self) -> bool {
        self.contains(Self::FOG_VALIDITY)
    }

    /// Returns true if channel X validity bit is set
    pub fn is_x_valid(&self) -> bool {
        self.contains(Self::VALIDITY_X)
    }

    /// Returns true if channel Y validity bit is set
    pub fn is_y_valid(&self) -> bool {
        self.contains(Self::VALIDITY_Y)
    }

    /// Returns true if channel Z validity bit is set
    pub fn is_z_valid(&self) -> bool {
        self.contains(Self::VALIDITY_Z)
    }

    /// Returns true if any communication errors are present
    pub fn has_comm_errors(&self) -> bool {
        self.intersects(Self::COMM_ERRORS)
    }
}

// SAFETY: HealthStatus is #[repr(transparent)] over u32, which is Pod
unsafe impl Pod for HealthStatus {}
// SAFETY: HealthStatus is #[repr(transparent)] over u32, which is Zeroable
unsafe impl Zeroable for HealthStatus {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_defaults() {
        // Default state: validity bits set, everything else clear
        let default_healthy = HealthStatus::VALIDITY_X
            | HealthStatus::VALIDITY_Y
            | HealthStatus::VALIDITY_Z
            | HealthStatus::FOG_VALIDITY;

        assert!(default_healthy.is_fog_valid());
        assert!(default_healthy.is_x_valid());
        assert!(default_healthy.is_y_valid());
        assert!(default_healthy.is_z_valid());
        assert!(default_healthy.is_gyro_mode());
        assert!(!default_healthy.has_comm_errors());
    }

    #[test]
    fn test_health_status_boot_mode() {
        let boot = HealthStatus::GYRO_STATE;
        assert!(boot.is_boot_mode());
        assert!(!boot.is_gyro_mode());
    }

    #[test]
    fn test_health_status_errors() {
        let with_x_error = HealthStatus::INTADCSAT_X;
        assert!(with_x_error.intersects(HealthStatus::ERRORS_X));
        assert!(!with_x_error.intersects(HealthStatus::ERRORS_Y));

        let with_comm_error = HealthStatus::CRC_ERR_NOM;
        assert!(with_comm_error.has_comm_errors());
    }
}

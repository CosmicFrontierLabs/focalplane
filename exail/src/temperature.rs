//! Temperature decoder for Exail Astrix NS thermistor readings.
//!
//! Implements the Steinhart-Hart-like equations from sections 2.7.7.1 and 2.7.7.2
//! of the Astrix NS User Guide.

const ALPHA: f64 = 32768.0; // 2^15

/// Configuration for a thermistor temperature conversion.
#[derive(Debug, Clone, Copy)]
pub struct TempDecoder {
    /// Prefactor applied to the ln argument (R0/Rt0 for SIA, 1.0 for board temps)
    pub prefactor: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl TempDecoder {
    /// Create a new temperature decoder with the given coefficients.
    pub const fn new(prefactor: f64, a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            prefactor,
            a,
            b,
            c,
            d,
        }
    }

    /// Convert a raw u16 ADC reading to temperature in degrees Celsius.
    ///
    /// Returns `None` if the input would cause a domain error (ln of non-positive).
    pub fn convert(&self, input: u16) -> Option<f64> {
        let t_inc = input as f64;

        // Compute the ln argument: prefactor * T_inc / (α - 1 - T_inc)
        let denom = ALPHA - 1.0 - t_inc;
        if denom <= 0.0 {
            return None;
        }

        let ln_arg = self.prefactor * t_inc / denom;
        if ln_arg <= 0.0 {
            return None;
        }

        let ln_val = ln_arg.ln();
        let ln2 = ln_val * ln_val;
        let ln3 = ln2 * ln_val;

        // Steinhart-Hart: T = [A + B*ln(x) + C*ln(x)^2 + D*ln(x)^3]^(-1)
        let inv_kelvin = self.a + self.b * ln_val + self.c * ln2 + self.d * ln3;

        if inv_kelvin == 0.0 {
            return None;
        }

        let kelvin = 1.0 / inv_kelvin;
        let celsius = kelvin - 273.15;

        Some(celsius)
    }
}

// Pre-defined decoders from the Astrix NS datasheet

/// SIA Filter Temperature (SIAFilTemp) - Section 2.7.7.1
/// Uses R0=10000, Rt0=15000, so prefactor = 0.6666...
pub const SIA_FIL_TEMP: TempDecoder = TempDecoder::new(
    10000.0 / 15000.0, // R0/Rt0
    0.003_354_002_877_286_514,
    0.000_278_828_995_767_310_36,
    0.0000033862644724_025175,
    0.0000001629834253_74661,
);

/// Board/Optics/Interface Temperature (BoardTemp, OrgFilTemp, InterTemp) - Section 2.7.7.2
/// No resistance prefactor (effectively 1.0)
pub const BOARD_TEMP: TempDecoder = TempDecoder::new(
    1.0,
    0.003_354_812_296_554_496_2,
    0.000_307_687_629_805_939_6,
    0.0000071253543002_31235,
    -0.0000000868980552_065042,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sia_fil_temp_midrange() {
        // Test with a mid-range value
        let temp = SIA_FIL_TEMP.convert(16384).unwrap();
        println!("SIA @ 16384: {:.2}°C", temp);
        assert!(temp > -50.0 && temp < 100.0);
    }

    #[test]
    fn test_board_temp_midrange() {
        let temp = BOARD_TEMP.convert(16384).unwrap();
        println!("Board @ 16384: {:.2}°C", temp);
        assert!(temp > -50.0 && temp < 100.0);
    }

    #[test]
    fn test_edge_cases() {
        // Zero input - ln(0) is undefined
        assert!(SIA_FIL_TEMP.convert(0).is_none());

        // Max value that causes division issues
        assert!(SIA_FIL_TEMP.convert(32767).is_none());
    }
}

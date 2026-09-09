//! The Sun as a simulator [`Spectrum`], from the TSIS-1 Hybrid Solar
//! Reference Spectrum.
//!
//! The table comes from `starfield-solar-spectrum`: 1 nm box means of
//! TSIS-1 HSRS v2 (Coddington et al. 2023) over 202–2730 nm at 1 AU,
//! built from the native-resolution archive product. Integrated over the
//! archive range it gives 1325.8 W m⁻², 97.4 % of the solar constant.
//!
//! Two disciplines carried over from the datasource:
//!
//! - Values are the containing bin's box mean. Nothing here interpolates,
//!   because interpolating between box means of a line-blanketed spectrum
//!   implies a resolution the data does not have.
//! - A band that is not wholly inside the archive coverage is an error,
//!   never a silently short number. The [`Spectrum`] trait cannot return
//!   an error, so callers must run [`TsisSolarSpectrum::check_covers`]
//!   (or use [`TsisSolarSpectrum::photo_electron_rate`], which does) before
//!   integrating; outside coverage the trait methods return zero.

use std::fmt;
use std::time::Duration;

use shared::units::{Area, LengthExt, Wavelength};
use starfield_solar_spectrum::SolarSpectrum;
use thiserror::Error;

use crate::photometry::quantum_efficiency::QuantumEfficiency;
use crate::photometry::spectrum::{Band, Spectrum};

/// Provenance string for assertion messages and metadata.
pub const HSRS_PROVENANCE: &str =
    "TSIS-1 HSRS v2 (Coddington et al. 2023) via starfield-solar-spectrum 0.1.0, 1 nm box means";

/// 1 W m⁻² in erg s⁻¹ cm⁻².
const W_M2_TO_ERG_S_CM2: f64 = 1.0e3;

/// Errors from the solar spectrum adapter.
#[derive(Debug, Error)]
pub enum SolarSpectrumError {
    /// The embedded table failed to parse.
    #[error("loading the embedded TSIS-1 table: {0}")]
    Load(String),
    /// A requested band leaves the archive coverage.
    #[error(
        "band {lo_nm:.1}–{hi_nm:.1} nm is not wholly inside the TSIS-1 coverage \
         {cover_lo_nm:.0}–{cover_hi_nm:.0} nm"
    )]
    OutsideCoverage {
        lo_nm: f64,
        hi_nm: f64,
        cover_lo_nm: f64,
        cover_hi_nm: f64,
    },
}

/// TSIS-1 solar irradiance at 1 AU as a [`Spectrum`].
pub struct TsisSolarSpectrum {
    table: SolarSpectrum,
}

impl fmt::Debug for TsisSolarSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.table.range_nm();
        write!(
            f,
            "TsisSolarSpectrum({lo:.0}–{hi:.0} nm, {} bins)",
            self.table.len()
        )
    }
}

impl TsisSolarSpectrum {
    /// Load the embedded table (no network).
    pub fn load() -> Result<Self, SolarSpectrumError> {
        SolarSpectrum::load_embedded()
            .map(|table| Self { table })
            .map_err(|e| SolarSpectrumError::Load(e.to_string()))
    }

    /// The underlying table.
    pub fn table(&self) -> &SolarSpectrum {
        &self.table
    }

    /// Error unless `band` lies wholly inside the archive coverage.
    pub fn check_covers(&self, band: &Band) -> Result<(), SolarSpectrumError> {
        let (cover_lo_nm, cover_hi_nm) = self.table.range_nm();
        if band.lower_nm >= cover_lo_nm
            && band.upper_nm <= cover_hi_nm
            && band.upper_nm > band.lower_nm
        {
            Ok(())
        } else {
            Err(SolarSpectrumError::OutsideCoverage {
                lo_nm: band.lower_nm,
                hi_nm: band.upper_nm,
                cover_lo_nm,
                cover_hi_nm,
            })
        }
    }

    /// Irradiance over `band` at 1 AU, W m⁻².
    pub fn irradiance_w_m2(&self, band: &Band) -> Result<f64, SolarSpectrumError> {
        self.check_covers(band)?;
        self.table
            .irradiance_over(band.lower_nm, band.upper_nm)
            .ok_or_else(|| SolarSpectrumError::Load("irradiance_over returned None".into()))
    }

    /// Photo-electron rate (e⁻/s) collected through `qe` by `aperture`
    /// from the Sun at 1 AU. Errors if the QE support leaves the archive
    /// coverage.
    pub fn photo_electron_rate(
        &self,
        qe: &QuantumEfficiency,
        aperture: Area,
    ) -> Result<f64, SolarSpectrumError> {
        self.check_covers(&qe.band())?;
        Ok(self.photo_electrons(qe, aperture, &Duration::from_secs(1)))
    }
}

impl Spectrum for TsisSolarSpectrum {
    /// F_ν in erg s⁻¹ cm⁻² Hz⁻¹ at 1 AU: the containing bin's box mean of
    /// F_λ converted with λ²/c at the requested wavelength. Zero outside
    /// coverage; see the module docs.
    fn spectral_irradiance(&self, wavelength: Wavelength) -> f64 {
        self.table
            .f_nu_cgs_at_nm(wavelength.as_nanometers())
            .unwrap_or(0.0)
    }

    /// Band irradiance in erg s⁻¹ cm⁻² at 1 AU. Zero if the band is not
    /// wholly inside coverage.
    fn irradiance(&self, band: &Band) -> f64 {
        self.table
            .irradiance_over(band.lower_nm, band.upper_nm)
            .map_or(0.0, |w_m2| w_m2 * W_M2_TO_ERG_S_CM2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::create_flat_qe;
    use crate::photometry::spectrum::CGS;
    use crate::photometry::stellar::BlackbodyStellarSpectrum;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use shared::units::AreaExt;

    fn sun() -> TsisSolarSpectrum {
        TsisSolarSpectrum::load().expect("embedded table")
    }

    /// The three locks agreed with the datasource: these are the table's
    /// own box means, so a rebuild against a newer HSRS moves them and
    /// this test correctly fails.
    #[test]
    fn irradiance_locks_match_the_pinned_table() {
        let sun = sun();
        for (lo, hi, expected) in [(300.0, 1100.0, 1002.2), (380.0, 750.0, 624.6)] {
            let got = sun.irradiance_w_m2(&Band::from_nm_bounds(lo, hi)).unwrap();
            assert_abs_diff_eq!(got, expected, epsilon = 0.15);
            assert!(
                (got - expected).abs() < 0.15,
                "{lo}-{hi} nm = {got:.2} W/m², expected {expected} from {HSRS_PROVENANCE}"
            );
        }
        let total = sun.table().total_irradiance();
        assert!(
            (total - 1325.8).abs() < 0.15,
            "202-2730 nm = {total:.2} W/m², expected 1325.8 from {HSRS_PROVENANCE}"
        );
    }

    /// AB magnitude of the Sun near V is about −26.7; a wrong power of
    /// ten in the F_ν conversion cannot survive this.
    #[test]
    fn f_nu_gives_the_solar_ab_magnitude() {
        let sun = sun();
        let f_nu = sun.spectral_irradiance(Wavelength::from_nanometers(550.0));
        let ab = -2.5 * (f_nu / CGS::AB_ZERO_POINT_FLUX_DENSITY).log10();
        assert_abs_diff_eq!(ab, -26.81, epsilon = 0.05);
    }

    #[test]
    fn bands_outside_coverage_are_errors_not_zeros() {
        let sun = sun();
        assert!(matches!(
            sun.check_covers(&Band::from_nm_bounds(100.0, 400.0)),
            Err(SolarSpectrumError::OutsideCoverage { .. })
        ));
        assert!(sun
            .irradiance_w_m2(&Band::from_nm_bounds(2000.0, 3000.0))
            .is_err());
        assert!(sun
            .check_covers(&Band::from_nm_bounds(300.0, 1100.0))
            .is_ok());
    }

    #[test]
    fn band_irradiance_trait_matches_direct_integral_in_cgs() {
        let sun = sun();
        let band = Band::from_nm_bounds(400.0, 700.0);
        let direct = sun.irradiance_w_m2(&band).unwrap() * 1.0e3;
        assert_relative_eq!(sun.irradiance(&band), direct, max_relative = 1e-12);
    }

    /// The blackbody that stood in for the Sun before this table should
    /// agree with it to within the shape error of a 5772 K Planck curve.
    #[test]
    fn photo_electron_rate_is_close_to_the_blackbody_stand_in() {
        let sun = sun();
        let qe = create_flat_qe(0.5);
        let aperture = Area::from_square_centimeters(100.0);
        let tsis = sun.photo_electron_rate(&qe, aperture).unwrap();
        let blackbody = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(0.65, -26.90)
            .photo_electrons(&qe, aperture, &Duration::from_secs(1));
        assert_relative_eq!(tsis, blackbody, max_relative = 0.15);
        // ~4×10²¹ solar photons/s/m² across the whole spectrum; 100 cm² is
        // 10⁻² m², the 400–800 nm window holds about half the photons and
        // the flat QE is 0.5, so of order 10¹⁹ e⁻/s.
        assert!(tsis > 3e18 && tsis < 3e19, "rate {tsis:e}");
    }
}

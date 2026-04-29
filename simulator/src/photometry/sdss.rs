//! SDSS+GALEX broad-band photometric spectrum.
//!
//! Builds a wavelength-resolved spectrum (the existing [`Spectrum`] trait
//! shape, in CGS `F_ν` units of erg s⁻¹ cm⁻² Hz⁻¹) from per-band fluxes
//! given in nanomaggies. This is the spectrum source used for catalog
//! entries that publish broadband photometry rather than a temperature
//! or full SED — chiefly the NASA-Sloan Atlas, but also any other
//! SDSS-lineage catalog (DESI Legacy Survey, future LSST broad bands).
//!
//! The interpolation between band pivots is **log-log linear in F_ν vs.
//! wavelength**. This is the right choice for the smooth power-law-like
//! continua that characterize most galaxy SEDs in the optical, but it
//! breaks across the Lyman/Balmer/4000 Å breaks at moderate redshift.
//! For our purposes (rendering catalog galaxies into a single optical
//! telescope's QE band) the assumption is comfortably inside the
//! photometric noise floor.
//!
//! Outside the populated wavelength range the spectrum returns 0 — no
//! extrapolation in v1, since extrapolating galaxy SEDs is a separate
//! problem with multiple defensible answers.

use std::time::Duration;

use shared::units::{Area, LengthExt, Wavelength};

use crate::photometry::quantum_efficiency::QuantumEfficiency;
use crate::photometry::spectrum::{Band, Spectrum, CGS};

/// Number of bands in the canonical SDSS+GALEX layout.
pub const SDSS_BAND_COUNT: usize = 7;

/// Identifies one of the SDSS+GALEX broad bands. Indexed 0..7 in the
/// canonical FUV-through-Z order matching `NsaEntry::sersic_flux` and the
/// rest of the SDSS pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum SdssBand {
    /// GALEX far-UV (~153.9 nm).
    Fuv = 0,
    /// GALEX near-UV (~231.6 nm).
    Nuv = 1,
    /// SDSS u (~354.3 nm).
    U = 2,
    /// SDSS g (~477.0 nm).
    G = 3,
    /// SDSS r (~623.1 nm).
    R = 4,
    /// SDSS i (~762.5 nm).
    I = 5,
    /// SDSS z (~913.4 nm).
    Z = 6,
}

impl SdssBand {
    /// All bands in canonical order.
    pub const ALL: [SdssBand; SDSS_BAND_COUNT] = [
        SdssBand::Fuv,
        SdssBand::Nuv,
        SdssBand::U,
        SdssBand::G,
        SdssBand::R,
        SdssBand::I,
        SdssBand::Z,
    ];

    /// Effective wavelength (nm) for each band — the AB-system pivot at
    /// which the published flux is reported.
    ///
    /// Values: GALEX FUV/NUV from Morrissey et al. (2007); SDSS u/g/r/i/z
    /// from Doi et al. (2010), Table 1, "AB system pivot wavelengths."
    pub const fn effective_wavelength_nm(&self) -> f64 {
        match self {
            SdssBand::Fuv => 153.9,
            SdssBand::Nuv => 231.6,
            SdssBand::U => 354.3,
            SdssBand::G => 477.0,
            SdssBand::R => 623.1,
            SdssBand::I => 762.5,
            SdssBand::Z => 913.4,
        }
    }
}

/// Per-band photometric fluxes on the SDSS+GALEX broad-band system,
/// usable as a [`Spectrum`].
///
/// This is *not* a true SED model — it is a piecewise interpolation
/// between band effective wavelengths, suitable for rendering catalog
/// entries within a single optical telescope's QE band. See module-level
/// docs for the assumptions and where they break.
///
/// # Choice of flux source for galaxies
///
/// For galaxies rendered via a Sérsic surface-brightness profile, prefer
/// building from the catalog's *Sérsic-fit* fluxes (e.g.
/// `NsaEntry::sersic_flux`) so the rendered ellipse conserves the model
/// flux. For galaxies rendered via aperture, use the catalog's NMGY
/// aperture fluxes (exposed via `Photometry::ab_magnitude` upstream).
/// The two sources differ by the residual between fit and aperture, which
/// can reach a few tens of percent for galaxies whose Sérsic fit is poor.
#[derive(Debug, Clone)]
pub struct SDSSSpectrum {
    /// Per-band fluxes in nanomaggies. `None` = band not measured.
    /// Indexed by `SdssBand as usize`.
    fluxes_nmgy: [Option<f64>; SDSS_BAND_COUNT],
}

impl SDSSSpectrum {
    /// Construct from the 5-band SDSS optical layout (u, g, r, i, z) — the
    /// NSA v0_1_2 file format and most other SDSS-only catalogs. UV slots
    /// remain `None`. Fluxes in nanomaggies.
    pub fn from_5_band(u: f64, g: f64, r: f64, i: f64, z: f64) -> Self {
        Self {
            fluxes_nmgy: [
                None,
                None,
                some_if_finite_positive(u),
                some_if_finite_positive(g),
                some_if_finite_positive(r),
                some_if_finite_positive(i),
                some_if_finite_positive(z),
            ],
        }
    }

    /// Construct from the 7-band SDSS+GALEX layout (FUV, NUV, u, g, r,
    /// i, z) — the NSA v1_0_1 file format. Fluxes in nanomaggies. Pass
    /// non-positive or non-finite values for known-missing bands and they
    /// will be treated as "not measured."
    #[allow(clippy::too_many_arguments)]
    pub fn from_7_band(fuv: f64, nuv: f64, u: f64, g: f64, r: f64, i: f64, z: f64) -> Self {
        Self {
            fluxes_nmgy: [
                some_if_finite_positive(fuv),
                some_if_finite_positive(nuv),
                some_if_finite_positive(u),
                some_if_finite_positive(g),
                some_if_finite_positive(r),
                some_if_finite_positive(i),
                some_if_finite_positive(z),
            ],
        }
    }

    /// Construct from a slice of `(band, flux_nmgy)` pairs. Useful for
    /// loaders that don't know at compile time which bands are present
    /// (e.g. an NSA loader that auto-detects v0_1_2 vs. v1_0_1). Bands
    /// not present in the slice remain `None`.
    pub fn from_band_slice(samples: &[(SdssBand, f64)]) -> Self {
        let mut fluxes_nmgy = [None; SDSS_BAND_COUNT];
        for (band, flux) in samples {
            fluxes_nmgy[*band as usize] = some_if_finite_positive(*flux);
        }
        Self { fluxes_nmgy }
    }

    /// Returns the per-band flux in nanomaggies, or `None` if the band
    /// is not measured.
    pub fn flux_nmgy(&self, band: SdssBand) -> Option<f64> {
        self.fluxes_nmgy[band as usize]
    }

    /// Convert a per-band nanomaggie flux to spectral flux density in
    /// CGS `F_ν` (erg s⁻¹ cm⁻² Hz⁻¹). Per the AB system definition,
    /// `m_AB = -2.5 log10(F_ν / 3631 Jy)`, so
    /// `1 nMgy = 10^-9 maggy = 3.631 × 10⁻³² erg s⁻¹ cm⁻² Hz⁻¹`.
    fn nmgy_to_fnu_cgs(flux_nmgy: f64) -> f64 {
        // 1 nMgy = 10^-9 × 3631 Jy = 10^-9 × 3631e-23 erg s⁻¹ cm⁻² Hz⁻¹
        flux_nmgy * CGS::AB_ZERO_POINT_FLUX_DENSITY * 1e-9
    }

    /// Returns `(λ₁, F_ν₁, λ₂, F_ν₂)` for the two populated bands that
    /// bracket `wavelength_nm`, or `None` if no bracket exists (i.e. the
    /// wavelength is outside the populated range).
    fn bracketing_pivots(&self, wavelength_nm: f64) -> Option<(f64, f64, f64, f64)> {
        let mut lower: Option<(f64, f64)> = None;
        let mut upper: Option<(f64, f64)> = None;
        for band in SdssBand::ALL {
            let Some(flux) = self.flux_nmgy(band) else {
                continue;
            };
            let lam = band.effective_wavelength_nm();
            let fnu = Self::nmgy_to_fnu_cgs(flux);
            if lam <= wavelength_nm {
                // Track the *largest* pivot at-or-below the query — that
                // gives the tightest bracket on the left side.
                lower = match lower {
                    Some((l, _)) if l >= lam => lower,
                    _ => Some((lam, fnu)),
                };
            }
            if lam >= wavelength_nm {
                // Track the *smallest* pivot at-or-above the query — the
                // tightest bracket on the right side.
                upper = match upper {
                    Some((l, _)) if l <= lam => upper,
                    _ => Some((lam, fnu)),
                };
            }
        }
        match (lower, upper) {
            (Some((l1, f1)), Some((l2, f2))) => Some((l1, f1, l2, f2)),
            _ => None,
        }
    }
}

fn some_if_finite_positive(x: f64) -> Option<f64> {
    if x.is_finite() && x > 0.0 {
        Some(x)
    } else {
        None
    }
}

impl Spectrum for SDSSSpectrum {
    /// Spectral flux density `F_ν` at `wavelength` in CGS units (erg s⁻¹
    /// cm⁻² Hz⁻¹). Linear interpolation in `(log λ, log F_ν)` between
    /// the populated bands bracketing `wavelength`. Returns 0 outside
    /// the populated range.
    fn spectral_irradiance(&self, wavelength: Wavelength) -> f64 {
        let lam = wavelength.as_nanometers();
        if !lam.is_finite() || lam <= 0.0 {
            return 0.0;
        }
        let Some((l1, f1, l2, f2)) = self.bracketing_pivots(lam) else {
            return 0.0;
        };
        // Exact pivot match (or both pivots coincide because only one
        // band is populated and the query happens to land on it).
        if (l1 - l2).abs() < f64::EPSILON {
            return f1;
        }
        // Log-log linear interpolation. Both flux values are positive
        // (they came from `some_if_finite_positive`), so log is safe.
        let log_lam = lam.ln();
        let log_l1 = l1.ln();
        let log_l2 = l2.ln();
        let log_f1 = f1.ln();
        let log_f2 = f2.ln();
        let t = (log_lam - log_l1) / (log_l2 - log_l1);
        (log_f1 + t * (log_f2 - log_f1)).exp()
    }

    /// Band-integrated irradiance in erg s⁻¹ cm⁻². Trapezoidal
    /// integration over the band's 1 nm sub-bands of `spectral_irradiance
    /// × dν`. The conversion `dν = c · dλ / λ²` is folded in per
    /// sub-band.
    fn irradiance(&self, band: &Band) -> f64 {
        if band.lower_nm <= 0.0 || band.lower_nm >= band.upper_nm {
            return 0.0;
        }
        // Trapezoidal integration in frequency space. For each sub-band,
        // evaluate F_ν at the centre and multiply by Δν.
        let mut total = 0.0;
        for sub in band.sub_nm_bands() {
            let center = sub.center();
            let fnu = self.spectral_irradiance(center);
            let (lo_freq, hi_freq) = sub.frequency_bounds();
            total += fnu * (hi_freq - lo_freq);
        }
        total
    }

    fn photons(&self, band: &Band, aperture: Area, duration: Duration) -> f64
    where
        Self: Sized,
    {
        crate::photometry::photoconversion::photons(self, band, aperture, &duration)
    }

    fn photo_electrons(&self, qe: &QuantumEfficiency, aperture: Area, duration: &Duration) -> f64
    where
        Self: Sized,
    {
        crate::photometry::photoconversion::photo_electrons(self, qe, aperture, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photometry::stellar::FlatStellarSpectrum;
    use approx::assert_relative_eq;
    use shared::units::LengthExt;

    /// Anchor: 1 nMgy converted to F_ν must match the AB-system zero-point
    /// scaled by 1e-9 (since 1 maggy = 3631 Jy). This is a direct check of
    /// the only unit conversion in the file.
    #[test]
    fn nmgy_to_fnu_matches_ab_zero_point() {
        let one_nmgy_in_cgs = SDSSSpectrum::nmgy_to_fnu_cgs(1.0);
        let expected = CGS::AB_ZERO_POINT_FLUX_DENSITY * 1e-9;
        assert_relative_eq!(one_nmgy_in_cgs, expected, epsilon = 1e-30);
    }

    /// Anchor: AB magnitude 0 corresponds to F_ν = 3631 Jy by definition.
    /// `m = -2.5 log10(F / 3631 Jy)` ⇒ for F = 3631 Jy, m = 0. In
    /// nanomaggies, 3631 Jy = 1 maggy = 10^9 nMgy. So an SDSS spectrum
    /// with all bands at 10^9 nMgy must produce the same monochromatic
    /// F_ν as a `FlatStellarSpectrum::from_ab_mag(0.0)` at any in-band
    /// wavelength.
    #[test]
    fn ab_magnitude_zero_matches_flat_spectrum_reference() {
        let one_maggy_per_band = 1e9; // nMgy
        let s = SDSSSpectrum::from_5_band(
            one_maggy_per_band,
            one_maggy_per_band,
            one_maggy_per_band,
            one_maggy_per_band,
            one_maggy_per_band,
        );
        let reference = FlatStellarSpectrum::from_ab_mag(0.0);
        let test_lambda = SdssBand::G.effective_wavelength_nm();
        let f_nu_sdss = s.spectral_irradiance(Wavelength::from_nanometers(test_lambda));
        let f_nu_ref = reference.spectral_irradiance(Wavelength::from_nanometers(test_lambda));
        assert_relative_eq!(f_nu_sdss, f_nu_ref, epsilon = 1e-12);
    }

    /// At the band pivot wavelengths, the spectrum value must equal the
    /// per-band F_ν exactly (not interpolated). This locks the
    /// `bracketing_pivots` boundary handling.
    #[test]
    fn spectral_irradiance_at_pivot_returns_exact_band_flux() {
        let s = SDSSSpectrum::from_5_band(1.0, 2.5, 7.0, 12.0, 18.0);
        for (band, expected_nmgy) in [
            (SdssBand::U, 1.0),
            (SdssBand::G, 2.5),
            (SdssBand::R, 7.0),
            (SdssBand::I, 12.0),
            (SdssBand::Z, 18.0),
        ] {
            let lam = Wavelength::from_nanometers(band.effective_wavelength_nm());
            let expected = SDSSSpectrum::nmgy_to_fnu_cgs(expected_nmgy);
            assert_relative_eq!(s.spectral_irradiance(lam), expected, epsilon = 1e-15);
        }
    }

    /// Outside the populated range, spectral_irradiance must be zero —
    /// we don't extrapolate galaxy SEDs in v1.
    #[test]
    fn spectral_irradiance_outside_populated_range_is_zero() {
        let s = SDSSSpectrum::from_5_band(1.0, 1.0, 1.0, 1.0, 1.0);
        // Below SDSS u (354.3 nm) — only the SDSS bands are populated
        assert_eq!(
            s.spectral_irradiance(Wavelength::from_nanometers(300.0)),
            0.0
        );
        // Above SDSS z (913.4 nm)
        assert_eq!(
            s.spectral_irradiance(Wavelength::from_nanometers(1000.0)),
            0.0
        );
    }

    /// Continuity check: at a wavelength midway (in log-λ) between two
    /// pivots, the interpolated F_ν should be the geometric mean of the
    /// two pivots' F_ν values (since log-log linear interpolation
    /// becomes geometric in linear space).
    #[test]
    fn log_log_interpolation_is_geometric_mean_at_log_midpoint() {
        let s = SDSSSpectrum::from_5_band(0.0, 0.0, 1.0, 0.0, 4.0);
        // Only g (477.0 nm) and z (913.4 nm) are populated. Geometric
        // mean of their wavelengths is sqrt(477.0 * 913.4) ≈ 660.4 nm.
        let l1 = SdssBand::G.effective_wavelength_nm();
        let l2 = SdssBand::Z.effective_wavelength_nm();
        let lam_geom = (l1 * l2).sqrt();
        let f_at_geom = s.spectral_irradiance(Wavelength::from_nanometers(lam_geom));
        let f1 = SDSSSpectrum::nmgy_to_fnu_cgs(1.0);
        let f2 = SDSSSpectrum::nmgy_to_fnu_cgs(4.0);
        // Geometric mean of F_ν values: sqrt(1 * 4) = 2 in nMgy.
        let expected = (f1 * f2).sqrt();
        assert_relative_eq!(f_at_geom, expected, epsilon = 1e-12);
    }

    /// 7-band builder exercises the UV slots — verify FUV pivot reads
    /// back the input flux.
    #[test]
    fn from_7_band_populates_uv() {
        let s = SDSSSpectrum::from_7_band(0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0);
        let fuv_lambda = Wavelength::from_nanometers(SdssBand::Fuv.effective_wavelength_nm());
        assert_relative_eq!(
            s.spectral_irradiance(fuv_lambda),
            SDSSSpectrum::nmgy_to_fnu_cgs(0.5),
            epsilon = 1e-15
        );
    }

    /// Slice builder must populate exactly the bands provided, leave
    /// others as `None`.
    #[test]
    fn from_band_slice_populates_only_requested_bands() {
        let s = SDSSSpectrum::from_band_slice(&[(SdssBand::G, 5.0), (SdssBand::I, 10.0)]);
        assert_eq!(s.flux_nmgy(SdssBand::G), Some(5.0));
        assert_eq!(s.flux_nmgy(SdssBand::I), Some(10.0));
        assert_eq!(s.flux_nmgy(SdssBand::U), None);
        assert_eq!(s.flux_nmgy(SdssBand::R), None);
        assert_eq!(s.flux_nmgy(SdssBand::Z), None);
    }

    /// Builders treat non-positive or non-finite fluxes as missing data.
    /// This is what NSA's "zero or NaN means unmeasured" convention maps
    /// to upstream.
    #[test]
    fn nonpositive_fluxes_are_treated_as_missing() {
        let s = SDSSSpectrum::from_5_band(0.0, 1.0, f64::NAN, -3.0, f64::INFINITY);
        assert_eq!(s.flux_nmgy(SdssBand::U), None); // 0
        assert_eq!(s.flux_nmgy(SdssBand::G), Some(1.0));
        assert_eq!(s.flux_nmgy(SdssBand::R), None); // NaN
        assert_eq!(s.flux_nmgy(SdssBand::I), None); // negative
        assert_eq!(s.flux_nmgy(SdssBand::Z), None); // inf — not finite-positive
    }

    /// Anchor: integrating a flat-in-F_ν spectrum (all SDSS bands at
    /// 1 maggy = 10^9 nMgy) over a narrow band centred at 550 nm must
    /// reproduce the analytic answer F_ν · Δν, where F_ν is the AB
    /// zero-point and Δν is the band's frequency width.
    #[test]
    fn band_integration_recovers_fnu_times_bandwidth_for_flat_input() {
        // Build an SDSS spectrum that's effectively flat across all five
        // optical bands at AB mag 0 (1 maggy = 1e9 nMgy each).
        let s = SDSSSpectrum::from_5_band(1e9, 1e9, 1e9, 1e9, 1e9);
        // Narrow band well inside the SDSS coverage where interpolation
        // is between two pivots with equal F_ν → result is constant.
        let band = Band::from_nm_bounds(540.0, 560.0);
        let integrated = s.irradiance(&band);
        let (lo, hi) = band.frequency_bounds();
        let expected = CGS::AB_ZERO_POINT_FLUX_DENSITY * (hi - lo);
        // Tolerance is loose because trapezoid rule on a narrow band
        // hits sub-band rounding; still anchored to the analytic form.
        assert_relative_eq!(integrated, expected, epsilon = 1e-3);
    }

    /// Anchor: photon count for a flat-F_ν AB-mag-0 source through a
    /// notch QE band of width 88 nm centred at 551 nm (Johnson V) is
    /// known to be ≈ 1000 photons s⁻¹ cm⁻² Å⁻¹ (Vega rule of thumb,
    /// shifted by the 0.026 mag AB-vs-Vega offset for V band). This
    /// echoes the existing `test_photoelectron_math_vega` blackbody
    /// test but now anchors *our* spectrum class against the same
    /// reference.
    #[test]
    fn photon_flux_through_v_band_matches_vega_rule_of_thumb() {
        use crate::photometry::QuantumEfficiency;
        use shared::units::AreaExt;
        let aperture = Area::from_square_centimeters(1.0);
        let duration = Duration::from_secs(1);
        let v_band = Band::from_nm_bounds(551.0 - 44.0, 551.0 + 44.0);
        let qe = QuantumEfficiency::from_notch(&v_band, 1.0).unwrap();
        // AB mag = 0 ⇒ 1 maggy = 1e9 nMgy in every band.
        let s = SDSSSpectrum::from_5_band(1e9, 1e9, 1e9, 1e9, 1e9);
        let n_e = s.photo_electrons(&qe, aperture, &duration);

        // Vega rule of thumb: ≈ 1000 photons s⁻¹ cm⁻² Å⁻¹ at V. The
        // Vega-to-AB offset at V is +0.026 mag, so an AB-mag-0 source
        // is brighter by a factor of 10^(0.4 * 0.026) ≈ 1.0242 in F_ν.
        let angstroms = v_band.width() * 10.0;
        let expected = 1000.0 * angstroms * aperture.as_square_centimeters() * 1.0242;
        let rel_err = (n_e - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "AB-flat source at V: got {n_e} photons, expected ~{expected} (Vega rule of thumb)"
        );
    }
}

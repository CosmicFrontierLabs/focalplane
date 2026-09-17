//! What a point on a body's surface reflects, textured or not.
//!
//! The stamp rasteriser asks one question per sub-sample: given the
//! local illumination and viewing cosines, the phase angle, the
//! body-fixed direction of the surface point and the footprint of the
//! sub-sample, what is the bidirectional reflectance? [`SurfaceRadiance`]
//! answers it. Every plain [`Brdf`] is a `SurfaceRadiance` that ignores
//! the position; [`TexturedSurface`] scales a unit-albedo law by the
//! band albedo of the composition found at that position.
//!
//! # Band albedo of a texel
//!
//! A map sample is an `EndmemberMix`: abundances `a_i` of library
//! endmembers. The albedo the sensor sees is
//!
//! ```text
//! A = Σ_i a_i · w_i,   w_i = ∫ ρ_i(λ) F☉(λ) Q(λ) λ dλ / ∫ F☉(λ) Q(λ) λ dλ
//! ```
//!
//! with `ρ_i` the endmember reflectance, `F☉` the solar spectrum and `Q`
//! the sensor quantum efficiency. Weighting by `F☉·Q·λ` rather than by
//! `Q` alone is what makes the factorisation exact: measured against
//! the TSIS-1 table and the USGS endmembers, response-only weighting is
//! off by up to 7.5 % for green vegetation over 400–1100 nm because the
//! red edge sits where the solar·λ weighting changes fastest. The
//! weights are computed once per (body, sensor) at bind time, so the
//! hot loop is a dot product.
//!
//! Composition mixes are hemispherical (white-sky-like) albedos and are
//! used as the Lambert albedo `A` of the texel; the law supplies the
//! illumination geometry on top. Laws linear in albedo (Lambert,
//! Minnaert) make this exact; for Hapke and Lommel–Seeliger the scaling
//! is an approximation, documented on [`TexturedSurface`].
//!
//! # No-data
//!
//! A sample that returns nothing, or a mix with zero total weight, is
//! no-data (unmapped terrain, swath seams). It reflects nothing and is
//! counted, so a render can report the fraction of its disk that was
//! unmapped rather than passing it off as dark ground.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use nalgebra::Vector3;
use shared::units::{LengthExt, Wavelength};
use starfield_planet_maps::SurfaceSampler;
use starfield_reflectance_library::{Endmember, ReflectanceLibrary};
use thiserror::Error;

use crate::bodies::brdf::Brdf;
use crate::photometry::quantum_efficiency::QuantumEfficiency;
use crate::photometry::solar::TsisSolarSpectrum;

/// Spectral step for the band-weight integrals, nm.
const WEIGHT_STEP_NM: f64 = 1.0;

/// Largest fraction of the sensor's response weight `∫ F☉·Q·λ dλ` that
/// may fall outside an endmember spectrum's coverage. The USGS library
/// starts at 350 nm while silicon sensors respond from ~300 nm; the
/// weight there is small, and inside this limit the endmember's covered
/// band mean stands in for the uncovered sliver.
pub const MAX_UNCOVERED_RESPONSE_FRACTION: f64 = 0.05;

/// One surface sub-sample as the rasteriser sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePoint {
    /// Cosine of the incidence angle.
    pub mu0: f64,
    /// Cosine of the emission angle.
    pub mu: f64,
    /// Sun–surface–observer phase angle, radians.
    pub alpha: f64,
    /// Unit vector from the body centre through the surface point, in
    /// body-fixed coordinates (x toward the prime meridian, z toward the
    /// north pole).
    pub body_fixed: Vector3<f64>,
    /// Angular radius of the sub-sample on the sky, radians.
    pub sky_radius_rad: f64,
}

impl SurfacePoint {
    /// East-positive planetocentric longitude in `[0, 2π)`.
    pub fn lon_rad(&self) -> f64 {
        let lon = self.body_fixed.y.atan2(self.body_fixed.x);
        if lon < 0.0 {
            lon + std::f64::consts::TAU
        } else {
            lon
        }
    }

    /// Planetocentric latitude, radians.
    pub fn lat_rad(&self) -> f64 {
        self.body_fixed.z.clamp(-1.0, 1.0).asin()
    }
}

/// Reflectance of a body's surface at a point.
pub trait SurfaceRadiance: Send + Sync {
    /// Bidirectional reflectance in sr⁻¹ (see [`crate::bodies::brdf`]).
    fn reflectance(&self, point: &SurfacePoint) -> f64;

    /// Sub-samples that fell on unmapped terrain since the last reset.
    fn no_data_samples(&self) -> usize {
        0
    }
}

impl<B: Brdf> SurfaceRadiance for B {
    fn reflectance(&self, point: &SurfacePoint) -> f64 {
        Brdf::reflectance(self, point.mu0, point.mu, point.alpha)
    }
}

/// The sensor band a surface is being rendered into.
pub struct SensorBand<'a> {
    /// Combined telescope × sensor quantum efficiency.
    pub qe: &'a QuantumEfficiency,
    /// Solar spectrum supplying the weighting.
    pub solar: &'a TsisSolarSpectrum,
}

/// A surface description that can be bound to a sensor band.
///
/// Plain BRDFs bind to themselves; textured surfaces compute their
/// per-endmember band weights.
pub trait SurfaceModel: Send + Sync {
    /// Bind to a sensor band, returning the per-sample evaluator.
    fn bind(&self, band: &SensorBand<'_>) -> Result<Arc<dyn SurfaceRadiance>, SurfaceError>;

    /// Short label for logs and metadata.
    fn label(&self) -> String;
}

impl<B: Brdf + Clone + fmt::Debug + 'static> SurfaceModel for B {
    fn bind(&self, _band: &SensorBand<'_>) -> Result<Arc<dyn SurfaceRadiance>, SurfaceError> {
        Ok(Arc::new(self.clone()))
    }

    fn label(&self) -> String {
        format!("{self:?}")
    }
}

/// Errors from binding a textured surface to a sensor band.
#[derive(Debug, Error)]
pub enum SurfaceError {
    /// The reflectance library has no spectrum for an endmember the map
    /// can return.
    #[error("reflectance library has no spectrum for endmember {0:?}")]
    MissingEndmember(Endmember),
    /// An endmember's spectrum, the solar table or the QE support does
    /// not cover the sensor band, so its band weight is undefined.
    #[error(
        "band weight for {endmember:?} over {lo_nm:.0}–{hi_nm:.0} nm is undefined: \
         {uncovered_fraction:.1}% of the response weight lies outside the spectrum's \
         coverage (limit {limit:.1}%) or the response integrates to zero"
    )]
    UncoveredBand {
        endmember: Endmember,
        lo_nm: f64,
        hi_nm: f64,
        uncovered_fraction: f64,
        limit: f64,
    },
}

/// A body surface with spatially varying composition.
///
/// `law` is evaluated as given and scaled by the texel's band albedo,
/// so it must be a unit law in the same albedo convention the tier was
/// built in. Two conventions are in use:
///
/// - texels are Lambert (normal) albedos, as in the Earth tier's
///   per-endmember values: use `Lambert { albedo: 1.0 }` (or
///   `Minnaert { albedo: 1.0, k }`), which scale exactly;
/// - texels are brightness rescaled so the disk mean is the body's
///   geometric albedo, as in the Moon and Mars tiers: use
///   [`crate::bodies::brdf::UnitGeometricAlbedo`] around the body's
///   photometric law, so the disk-integrated brightness and phase curve
///   are the law's and the map only redistributes light across the disk.
///
/// Scaling a Hapke or Lommel–Seeliger law by anything other than a
/// constant factor changes its limb darkening; both wrappers above scale
/// by constants.
pub struct TexturedSurfaceModel {
    sampler: Arc<dyn SurfaceSampler + Send + Sync>,
    law: Arc<dyn Brdf>,
    library: Arc<ReflectanceLibrary>,
    label: String,
}

impl fmt::Debug for TexturedSurfaceModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TexturedSurfaceModel({})", self.label)
    }
}

impl TexturedSurfaceModel {
    /// A textured surface: `sampler` supplies composition per footprint,
    /// `law` the unit-albedo geometry law, `library` the endmember
    /// spectra.
    pub fn new(
        sampler: Arc<dyn SurfaceSampler + Send + Sync>,
        law: Arc<dyn Brdf>,
        library: Arc<ReflectanceLibrary>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            sampler,
            law,
            library,
            label: label.into(),
        }
    }

    /// Band weight `w_i` for one endmember: its reflectance averaged
    /// over the sensor band with weight `F☉·Q·λ`.
    ///
    /// The integral runs over the intersection of the sensor's QE
    /// support and the endmember spectrum's coverage. The response
    /// weight left outside that intersection must stay below
    /// [`MAX_UNCOVERED_RESPONSE_FRACTION`].
    fn band_weight(
        library: &ReflectanceLibrary,
        endmember: Endmember,
        band: &SensorBand<'_>,
    ) -> Result<f64, SurfaceError> {
        let reflectance = library
            .get(endmember)
            .ok_or(SurfaceError::MissingEndmember(endmember))?;
        let support = band.qe.band();
        let (curve_lo, curve_hi) = reflectance.range_nm();
        let lo_nm = support.lower_nm.max(curve_lo);
        let hi_nm = support.upper_nm.min(curve_hi);
        let solar = band.solar.table();
        let qe = band.qe;
        let uncovered = |uncovered_fraction: f64| SurfaceError::UncoveredBand {
            endmember,
            lo_nm: support.lower_nm,
            hi_nm: support.upper_nm,
            uncovered_fraction: 100.0 * uncovered_fraction,
            limit: 100.0 * MAX_UNCOVERED_RESPONSE_FRACTION,
        };
        if lo_nm >= hi_nm {
            return Err(uncovered(1.0));
        }

        let response = |nm: f64| qe.at(Wavelength::from_nanometers(nm)) * nm;
        let total = solar
            .weighted_irradiance(support.lower_nm, support.upper_nm, response)
            .filter(|w| *w > 0.0)
            .ok_or_else(|| uncovered(1.0))?;
        let covered = solar
            .weighted_irradiance(lo_nm, hi_nm, response)
            .ok_or_else(|| uncovered(1.0))?;
        let uncovered_fraction = 1.0 - covered / total;
        if uncovered_fraction > MAX_UNCOVERED_RESPONSE_FRACTION {
            return Err(uncovered(uncovered_fraction));
        }

        reflectance
            .curve()
            .weighted_mean(lo_nm, hi_nm, WEIGHT_STEP_NM, |nm| {
                solar.at_nm(nm).unwrap_or(0.0) * response(nm)
            })
            .ok_or_else(|| uncovered(uncovered_fraction))
    }
}

impl SurfaceModel for TexturedSurfaceModel {
    fn bind(&self, band: &SensorBand<'_>) -> Result<Arc<dyn SurfaceRadiance>, SurfaceError> {
        band.solar
            .check_covers(&band.qe.band())
            .map_err(|_| SurfaceError::UncoveredBand {
                endmember: Endmember::Shade,
                lo_nm: band.qe.band().lower_nm,
                hi_nm: band.qe.band().upper_nm,
                uncovered_fraction: 100.0,
                limit: 100.0 * MAX_UNCOVERED_RESPONSE_FRACTION,
            })?;
        let mut weights = Vec::new();
        for endmember in self.sampler.endmembers() {
            let w = Self::band_weight(&self.library, endmember, band)?;
            weights.push((endmember, w));
        }
        Ok(Arc::new(TexturedSurface {
            sampler: Arc::clone(&self.sampler),
            law: Arc::clone(&self.law),
            weights,
            no_data: AtomicUsize::new(0),
        }))
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

/// A [`TexturedSurfaceModel`] bound to one sensor band.
pub struct TexturedSurface {
    sampler: Arc<dyn SurfaceSampler + Send + Sync>,
    law: Arc<dyn Brdf>,
    weights: Vec<(Endmember, f64)>,
    no_data: AtomicUsize,
}

impl TexturedSurface {
    /// Band weights `w_i` per endmember.
    pub fn weights(&self) -> &[(Endmember, f64)] {
        &self.weights
    }

    /// Band albedo of the composition at a point, or `None` for no-data.
    pub fn band_albedo(&self, point: &SurfacePoint) -> Option<f64> {
        let (mix, _footprint) = self.sampler.sample_area(
            point.lon_rad(),
            point.lat_rad(),
            point.sky_radius_rad,
            point.mu,
        )?;
        if mix.total_weight() <= 0.0 {
            return None;
        }
        let mut albedo = 0.0;
        for &(endmember, abundance) in mix.weights() {
            let w = self
                .weights
                .iter()
                .find(|(e, _)| *e == endmember)
                .map(|(_, w)| *w)
                .unwrap_or(0.0);
            albedo += abundance * w;
        }
        Some(albedo)
    }
}

impl SurfaceRadiance for TexturedSurface {
    fn reflectance(&self, point: &SurfacePoint) -> f64 {
        match self.band_albedo(point) {
            Some(albedo) => albedo * self.law.reflectance(point.mu0, point.mu, point.alpha),
            None => {
                self.no_data.fetch_add(1, Ordering::Relaxed);
                0.0
            }
        }
    }

    fn no_data_samples(&self) -> usize {
        self.no_data.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::brdf::Lambert;
    use crate::hardware::sensor::create_flat_qe;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use starfield_planet_maps::earth_tier;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn point(body_fixed: Vector3<f64>) -> SurfacePoint {
        SurfacePoint {
            mu0: 1.0,
            mu: 1.0,
            alpha: 0.0,
            body_fixed: body_fixed.normalize(),
            sky_radius_rad: 1e-6,
        }
    }

    #[test]
    fn body_fixed_direction_to_lon_lat() {
        let p = point(Vector3::new(1.0, 0.0, 0.0));
        assert_abs_diff_eq!(p.lon_rad(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p.lat_rad(), 0.0, epsilon = 1e-12);
        let p = point(Vector3::new(0.0, 1.0, 0.0));
        assert_abs_diff_eq!(p.lon_rad(), FRAC_PI_2, epsilon = 1e-12);
        let p = point(Vector3::new(0.0, -1.0, 0.0));
        assert_abs_diff_eq!(p.lon_rad(), 1.5 * PI, epsilon = 1e-12);
        let p = point(Vector3::new(1.0, 0.0, 1.0));
        assert_abs_diff_eq!(p.lat_rad(), FRAC_PI_4, epsilon = 1e-12);
    }

    #[test]
    fn plain_brdf_is_a_surface_radiance_and_model() {
        let lambert = Lambert { albedo: 0.4 };
        let mut p = point(Vector3::new(1.0, 0.0, 0.0));
        p.mu0 = 0.5;
        assert_relative_eq!(
            SurfaceRadiance::reflectance(&lambert, &p),
            0.4 * 0.5 / PI,
            max_relative = 1e-12
        );
        let solar = TsisSolarSpectrum::load().unwrap();
        let qe = create_flat_qe(0.5);
        let bound = lambert
            .bind(&SensorBand {
                qe: &qe,
                solar: &solar,
            })
            .unwrap();
        assert_relative_eq!(bound.reflectance(&p), 0.4 * 0.5 / PI, max_relative = 1e-12);
        assert_eq!(bound.no_data_samples(), 0);
    }

    #[test]
    fn earth_tier_binds_and_reads_ocean_and_land() {
        let solar = TsisSolarSpectrum::load().unwrap();
        let qe = create_flat_qe(0.5);
        let library = Arc::new(ReflectanceLibrary::load_embedded().unwrap());
        let tier = Arc::new(earth_tier().unwrap());
        let model = TexturedSurfaceModel::new(
            tier,
            Arc::new(Lambert { albedo: 1.0 }),
            library,
            "Earth MCD12C1 0.25°",
        );
        let band = SensorBand {
            qe: &qe,
            solar: &solar,
        };
        let bound = model.bind(&band).unwrap();

        // Weights are reflectances: in [0, 1], Shade exactly zero, ocean
        // dark, arid soil bright.
        let library = ReflectanceLibrary::load_embedded().unwrap();
        let shade = TexturedSurfaceModel::band_weight(&library, Endmember::Shade, &band).unwrap();
        assert_abs_diff_eq!(shade, 0.0, epsilon = 1e-15);
        let ocean =
            TexturedSurfaceModel::band_weight(&library, Endmember::OpenOcean, &band).unwrap();
        let soil = TexturedSurfaceModel::band_weight(&library, Endmember::AridSoil, &band).unwrap();
        assert!(ocean > 0.0 && ocean < 0.1, "ocean weight {ocean:.3}");
        assert!(soil > 0.2 && soil < 0.6, "arid soil weight {soil:.3}");

        // Mid-Pacific (lat 0, lon 200°E) is ocean: dark.
        let pacific = SurfacePoint {
            mu0: 1.0,
            mu: 1.0,
            alpha: 0.0,
            body_fixed: Vector3::new(200f64.to_radians().cos(), 200f64.to_radians().sin(), 0.0),
            sky_radius_rad: 2e-5,
        };
        // Central Sahara (lat 23°N, lon 10°E) is arid soil: bright.
        let sahara = SurfacePoint {
            mu0: 1.0,
            mu: 1.0,
            alpha: 0.0,
            body_fixed: Vector3::new(
                23f64.to_radians().cos() * 10f64.to_radians().cos(),
                23f64.to_radians().cos() * 10f64.to_radians().sin(),
                23f64.to_radians().sin(),
            ),
            sky_radius_rad: 2e-5,
        };
        let ocean = bound.reflectance(&pacific) * PI;
        let desert = bound.reflectance(&sahara) * PI;
        assert!(ocean > 0.0 && ocean < 0.06, "ocean albedo {ocean:.3}");
        assert!(desert > 0.2 && desert < 0.5, "Sahara albedo {desert:.3}");
        assert!(desert > 4.0 * ocean);
        assert_eq!(bound.no_data_samples(), 0);
    }

    /// Geometric albedo of a bound surface: `∫ r μ dΩ` over the hemisphere
    /// facing an observer on the body-fixed +x axis, at zero phase.
    fn hemisphere_geometric_albedo(surface: &dyn SurfaceRadiance, n: usize) -> f64 {
        let d_theta = FRAC_PI_2 / n as f64;
        let d_phi = 2.0 * PI / (2 * n) as f64;
        let mut p = 0.0;
        for it in 0..n {
            let theta = (it as f64 + 0.5) * d_theta;
            let (sin_t, mu) = theta.sin_cos();
            for ip in 0..2 * n {
                let phi = (ip as f64 + 0.5) * d_phi;
                let point = SurfacePoint {
                    mu0: mu,
                    mu,
                    alpha: 0.0,
                    body_fixed: Vector3::new(mu, sin_t * phi.cos(), sin_t * phi.sin()),
                    sky_radius_rad: 2e-5,
                };
                p += surface.reflectance(&point) * mu * sin_t * d_theta * d_phi;
            }
        }
        p
    }

    /// The one-endmember Moon and Mars tiers are scaled so their disk mean
    /// is the body's geometric albedo (0.12, 0.17) in the band the tier
    /// records (`albedo_band_nm`, 500–600 nm): abundance = p / the
    /// endmember's box mean over that band. A sensor sees the endmember at
    /// its own band mean, so the textured body's albedo is
    /// p · ρ(sensor band) / ρ(tier band): the basalt analogue's colour
    /// carried into the band, a few percent either way for silicon. This
    /// locks the convention (a unit Lambert would put a disk-mean tier at
    /// 2/3 of its target), the recorded band, and the colour term.
    #[test]
    fn disk_mean_tiers_reproduce_their_geometric_albedo_in_the_sensor_band() {
        use crate::bodies::brdf::{Hapke, UnitGeometricAlbedo};
        use crate::hardware::sensor::models::IMX455;
        use starfield_planet_maps::{mars_tier, moon_tier, AbundanceTier, AlbedoConvention};

        let solar = TsisSolarSpectrum::load().unwrap();
        let qe = &IMX455.quantum_efficiency;
        let band = SensorBand { qe, solar: &solar };
        let library = Arc::new(ReflectanceLibrary::load_embedded().unwrap());

        // Ratio of the endmember's albedo in the sensor band to its mean
        // over the band the tier was normalised in.
        let colour_term = |endmember: Endmember, tier: &AbundanceTier| {
            assert_eq!(
                tier.albedo_convention(),
                AlbedoConvention::GeometricDiskMean
            );
            let (lo, hi) = tier.albedo_band_nm();
            let tier_band_mean = library
                .get(endmember)
                .unwrap()
                .curve()
                .mean_over(lo as f64, hi as f64)
                .unwrap();
            let sensor_band =
                TexturedSurfaceModel::band_weight(&library, endmember, &band).unwrap();
            eprintln!(
                "residual {endmember:?} imx455={sensor_band:.4} \
                 tier_band_{lo:.0}_{hi:.0}nm={tier_band_mean:.4} ratio={:.3}",
                sensor_band / tier_band_mean
            );
            sensor_band / tier_band_mean
        };
        let moon_map = moon_tier().unwrap();
        let mars_map = mars_tier().unwrap();
        let moon_colour = colour_term(Endmember::FreshBasalt, &moon_map);
        let mars_colour = colour_term(Endmember::WeatheredBasalt, &mars_map);
        assert!(
            (0.9..1.1).contains(&moon_colour),
            "fresh basalt colour term"
        );
        assert!(
            (0.9..1.1).contains(&mars_colour),
            "weathered basalt colour term"
        );

        let moon = TexturedSurfaceModel::new(
            Arc::new(moon_map),
            Arc::new(UnitGeometricAlbedo::new(Hapke::lunar_average())),
            Arc::clone(&library),
            "Moon",
        )
        .bind(&band)
        .unwrap();
        let p_moon = hemisphere_geometric_albedo(moon.as_ref(), 120);
        let mars = TexturedSurfaceModel::new(
            Arc::new(mars_map),
            Arc::new(UnitGeometricAlbedo::new(Lambert { albedo: 1.0 })),
            Arc::clone(&library),
            "Mars",
        )
        .bind(&band)
        .unwrap();
        let p_mars = hemisphere_geometric_albedo(mars.as_ref(), 120);
        eprintln!(
            "residual moon_nearside_p={p_moon:.4} target=0.12 mars_p={p_mars:.4} target=0.17 \
             moon_nodata={} mars_nodata={}",
            moon.no_data_samples(),
            mars.no_data_samples()
        );
        // The hemisphere facing +x is the Moon's mare-rich near side, a
        // fifth darker than the global mean; Mars's prime-meridian
        // hemisphere is close to its mean. Both carry the colour term.
        assert!((0.085..0.135).contains(&p_moon), "Moon p = {p_moon}");
        assert!((0.15..0.20).contains(&p_mars), "Mars p = {p_mars}");
        // No-data swath seams are a small fraction of the near side.
        assert!(moon.no_data_samples() < 120 * 240 / 20);

        // The quantity the tiers actually guarantee is the area-weighted
        // global mean of the texel albedo: the target times the colour
        // term, less the no-data fraction counted as zero here (2.47 % of
        // the Moon's area, none of Mars's).
        let global_mean = |tier: Arc<dyn SurfaceSampler + Send + Sync>| {
            let bound = TexturedSurfaceModel::new(
                tier,
                Arc::new(UnitGeometricAlbedo::new(Lambert { albedo: 1.0 })),
                Arc::clone(&library),
                "mean",
            )
            .bind(&band)
            .unwrap();
            let (n_lat, n_lon) = (180, 360);
            let (mut sum, mut area) = (0.0, 0.0);
            for i in 0..n_lat {
                let lat = (-90.0 + (i as f64 + 0.5) * 180.0 / n_lat as f64).to_radians();
                let w = lat.cos();
                for j in 0..n_lon {
                    let lon = ((j as f64 + 0.5) * 360.0 / n_lon as f64).to_radians();
                    let point = SurfacePoint {
                        mu0: 1.0,
                        mu: 1.0,
                        alpha: 0.0,
                        body_fixed: Vector3::new(
                            lat.cos() * lon.cos(),
                            lat.cos() * lon.sin(),
                            lat.sin(),
                        ),
                        sky_radius_rad: 2e-5,
                    };
                    // r(1, 1, 0) of a unit-geometric-albedo Lambert is 1.5/π
                    // per unit texel albedo.
                    sum += w * bound.reflectance(&point) * PI / 1.5;
                    area += w;
                }
            }
            sum / area
        };
        let moon_mean = global_mean(Arc::new(moon_tier().unwrap()));
        let mars_mean = global_mean(Arc::new(mars_tier().unwrap()));
        let moon_expected = 0.12 * moon_colour;
        let mars_expected = 0.17 * mars_colour;
        eprintln!(
            "residual moon_global_mean_albedo={moon_mean:.4} expected={moon_expected:.4} \
             nodata_offset={:.4} mars_global_mean_albedo={mars_mean:.4} \
             expected={mars_expected:.4}",
            1.0 - moon_mean / moon_expected
        );
        // Moon: the no-data area (2.47 %) counts as zero, so the mean sits
        // that far below the target and nowhere else.
        assert_relative_eq!(
            moon_mean,
            moon_expected * (1.0 - 0.0247),
            max_relative = 0.01
        );
        assert_relative_eq!(mars_mean, mars_expected, max_relative = 0.01);
    }

    /// A unit Lambert is the wrong law for a disk-mean tier: it would
    /// render the Moon at two thirds of its geometric albedo.
    #[test]
    fn unit_lambert_understates_a_disk_mean_tier() {
        use crate::bodies::brdf::UnitGeometricAlbedo;
        use starfield_planet_maps::mars_tier;

        let solar = TsisSolarSpectrum::load().unwrap();
        let qe = create_flat_qe(0.5);
        let band = SensorBand {
            qe: &qe,
            solar: &solar,
        };
        let library = Arc::new(ReflectanceLibrary::load_embedded().unwrap());
        let tier = Arc::new(mars_tier().unwrap());
        let lambert = TexturedSurfaceModel::new(
            Arc::clone(&tier) as Arc<dyn SurfaceSampler + Send + Sync>,
            Arc::new(Lambert { albedo: 1.0 }),
            Arc::clone(&library),
            "Mars, unit Lambert",
        )
        .bind(&band)
        .unwrap();
        let unit = TexturedSurfaceModel::new(
            tier,
            Arc::new(UnitGeometricAlbedo::new(Lambert { albedo: 1.0 })),
            library,
            "Mars, unit geometric albedo",
        )
        .bind(&band)
        .unwrap();
        let ratio = hemisphere_geometric_albedo(lambert.as_ref(), 60)
            / hemisphere_geometric_albedo(unit.as_ref(), 60);
        assert_relative_eq!(ratio, 2.0 / 3.0, max_relative = 1e-3);
    }

    #[test]
    fn uncovered_sensor_band_is_an_error_at_bind_time() {
        let solar = TsisSolarSpectrum::load().unwrap();
        // A QE support reaching into the ultraviolet beyond TSIS-1.
        let qe = QuantumEfficiency::from_table(
            vec![100.0, 150.0, 600.0, 700.0],
            vec![0.0, 0.5, 0.5, 0.0],
        )
        .unwrap();
        let library = Arc::new(ReflectanceLibrary::load_embedded().unwrap());
        let model = TexturedSurfaceModel::new(
            Arc::new(earth_tier().unwrap()),
            Arc::new(Lambert { albedo: 1.0 }),
            library,
            "Earth",
        );
        assert!(matches!(
            model.bind(&SensorBand {
                qe: &qe,
                solar: &solar
            }),
            Err(SurfaceError::UncoveredBand { .. })
        ));
    }
}

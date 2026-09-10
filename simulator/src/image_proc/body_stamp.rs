//! Rasterise one opaque body into a coverage stamp and an electron stamp.
//!
//! A [`BodyStamp`] is the per-body building block of the second
//! compositing pass (see [`super::compose`]). It ray-casts the body's
//! disk on an oversampled pixel grid, evaluates a surface reflectance law
//! at every sub-sample, and blurs both the geometric coverage and the
//! radiance with the sensor's PSF so that the limb, the terminator and
//! the occultation of background sources all share one blur.
//!
//! # Radiometry
//!
//! For a grey surface with bidirectional reflectance `r(μ₀, μ, α)` (see
//! [`crate::bodies::brdf`]) the mean electrons a pixel of solid angle
//! `Ω_px` collects over an exposure `T` are
//!
//! ```text
//! e = T · Ė☉(1 AU) / d☉² · Σ_sub r_i · Ω_px / s²
//! ```
//!
//! where `Ė☉(1 AU)` is the photo-electron rate the aperture would collect
//! from the Sun at 1 AU through the sensor band, `d☉` the body's
//! heliocentric distance in AU and `s` the oversampling factor. The
//! caller supplies `T · Ė☉ / d☉²` as [`StampGeometry::electrons_per_sr`].
//!
//! Sky geometry enters through a 2×2 Jacobian from angular offsets
//! (east, north) to pixel offsets, measured by the caller from the real
//! projection so that roll, axis flips and anisotropic plate scale need
//! no special handling here.

use std::f64::consts::PI;

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use ndarray::Array2;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::image_proc::convolve2d::{convolve2d, ConvolveMode, ConvolveOptions};

use crate::atmosphere::{AtmosphereRadiance, SkyRay};
use crate::bodies::surface::{SurfacePoint, SurfaceRadiance};
use crate::image_proc::compose::BodyComposite;

/// Default sub-samples per pixel edge.
pub const DEFAULT_OVERSAMPLING: usize = 4;

/// Atmosphere table resolution: radial samples across the disk, across
/// the shell, and azimuth samples from the Sun's direction to its
/// antipode.
const ATMOSPHERE_LUT_GROUND: usize = 256;
const ATMOSPHERE_LUT_LIMB: usize = 128;
const ATMOSPHERE_LUT_AZIMUTH: usize = 64;

/// Where and how a body appears on one sensor.
#[derive(Clone, Debug, PartialEq)]
pub struct StampGeometry {
    /// Sub-pixel position of the body centre on the sensor `(x, y)`.
    pub center_px: (f64, f64),
    /// Jacobian `d(px_x, px_y) / d(east_rad, north_rad)` at the body.
    pub jacobian: Matrix2<f64>,
    /// Angular semi-diameter of the disk, radians.
    pub semi_diameter_rad: f64,
    /// Unit vector toward the Sun in the body's sky frame
    /// `(east, north, toward observer)`.
    pub sun_direction_sky: Vector3<f64>,
    /// Sun–body–observer phase angle, radians.
    pub phase_angle: f64,
    /// `T · Ė☉(1 AU) / d☉²`: electrons per steradian per unit
    /// reflectance over the exposure.
    pub electrons_per_sr: f64,
    /// Rotation from the sky frame `(east, north, toward observer)` to
    /// body-fixed coordinates, for texture lookup.
    pub sky_to_body_fixed: Matrix3<f64>,
    /// Body radius in km, the length scale of the atmosphere shell.
    pub radius_km: f64,
}

/// Per-frame table of atmospheric terms over (impact parameter, azimuth
/// from the Sun). The single-scattering integrals depend only on the
/// ray's distance from the body centre and its angle from the Sun's
/// projection on the sky, so a few tens of thousands of rays cover a
/// disk of a million sub-samples.
struct AtmosphereLut {
    top_ratio: f64,
    /// Sun azimuth on the sky, radians from +east toward +north.
    sun_azimuth: f64,
    n_phi: usize,
    /// Ground rays, ρ ∈ [0, 1): `(view T, sun T, path radiance)`.
    ground: Vec<[f64; 3]>,
    n_ground: usize,
    /// Limb rays, ρ ∈ [1, top_ratio): `(transmittance, path radiance)`.
    limb: Vec<[f64; 2]>,
    n_limb: usize,
}

impl AtmosphereLut {
    fn build(
        atmosphere: &dyn AtmosphereRadiance,
        radius_km: f64,
        sun_direction_sky: Vector3<f64>,
        n_ground: usize,
        n_limb: usize,
        n_phi: usize,
    ) -> Self {
        let top_ratio = atmosphere.top_radius_km() / radius_km;
        let sun_azimuth = sun_direction_sky.y.atan2(sun_direction_sky.x);
        let mut ground = Vec::with_capacity(n_ground * n_phi);
        let mut limb = Vec::with_capacity(n_limb * n_phi);
        for i in 0..n_ground {
            // Sample ρ at bin centres, staying strictly inside the disk.
            let rho = (i as f64 + 0.5) / n_ground as f64;
            for j in 0..n_phi {
                let dphi = PI * j as f64 / (n_phi - 1).max(1) as f64;
                let ray = SkyRay {
                    offset_km: Self::offset_km(rho, dphi, sun_azimuth, radius_km),
                    sun_dir: sun_direction_sky,
                };
                let g = atmosphere.ground_ray(&ray);
                ground.push([g.view_transmittance, g.sun_transmittance, g.path_radiance]);
            }
        }
        for i in 0..n_limb {
            let rho = 1.0 + (top_ratio - 1.0) * (i as f64 + 0.5) / n_limb as f64;
            for j in 0..n_phi {
                let dphi = PI * j as f64 / (n_phi - 1).max(1) as f64;
                let ray = SkyRay {
                    offset_km: Self::offset_km(rho, dphi, sun_azimuth, radius_km),
                    sun_dir: sun_direction_sky,
                };
                let l = atmosphere.limb_ray(&ray);
                limb.push([l.transmittance, l.path_radiance]);
            }
        }
        Self {
            top_ratio,
            sun_azimuth,
            n_phi,
            ground,
            n_ground,
            limb,
            n_limb,
        }
    }

    fn offset_km(rho: f64, dphi: f64, sun_azimuth: f64, radius_km: f64) -> (f64, f64) {
        let phi = sun_azimuth + dphi;
        (rho * radius_km * phi.cos(), rho * radius_km * phi.sin())
    }

    /// Azimuth index coordinate for a sky offset `(xi, eta)` in body radii.
    fn phi_coord(&self, xi: f64, eta: f64) -> f64 {
        let mut dphi = (eta.atan2(xi) - self.sun_azimuth).rem_euclid(2.0 * PI);
        if dphi > PI {
            dphi = 2.0 * PI - dphi;
        }
        dphi / PI * (self.n_phi - 1) as f64
    }

    fn bilinear<const N: usize>(
        table: &[[f64; N]],
        n_rho: usize,
        n_phi: usize,
        rho_coord: f64,
        phi_coord: f64,
    ) -> [f64; N] {
        let rc = rho_coord.clamp(0.0, (n_rho - 1) as f64);
        let pc = phi_coord.clamp(0.0, (n_phi - 1) as f64);
        let r0 = rc.floor() as usize;
        let p0 = pc.floor() as usize;
        let r1 = (r0 + 1).min(n_rho - 1);
        let p1 = (p0 + 1).min(n_phi - 1);
        let fr = rc - r0 as f64;
        let fp = pc - p0 as f64;
        let mut out = [0.0; N];
        for (k, slot) in out.iter_mut().enumerate() {
            let v00 = table[r0 * n_phi + p0][k];
            let v01 = table[r0 * n_phi + p1][k];
            let v10 = table[r1 * n_phi + p0][k];
            let v11 = table[r1 * n_phi + p1][k];
            *slot = (1.0 - fr) * ((1.0 - fp) * v00 + fp * v01) + fr * ((1.0 - fp) * v10 + fp * v11);
        }
        out
    }

    /// `(view T, sun T, path radiance)` for a ground ray at `(xi, eta)`.
    fn ground(&self, xi: f64, eta: f64) -> [f64; 3] {
        let rho = (xi * xi + eta * eta).sqrt();
        let rho_coord = rho * self.n_ground as f64 - 0.5;
        Self::bilinear(
            &self.ground,
            self.n_ground,
            self.n_phi,
            rho_coord,
            self.phi_coord(xi, eta),
        )
    }

    /// `(transmittance, path radiance)` for a limb ray at `(xi, eta)`.
    fn limb(&self, xi: f64, eta: f64) -> [f64; 2] {
        let rho = (xi * xi + eta * eta).sqrt();
        let rho_coord = (rho - 1.0) / (self.top_ratio - 1.0) * self.n_limb as f64 - 0.5;
        Self::bilinear(
            &self.limb,
            self.n_limb,
            self.n_phi,
            rho_coord,
            self.phi_coord(xi, eta),
        )
    }
}

/// Coverage and electron stamps for one body, PSF-blurred.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyStamp {
    /// Sensor pixel `(x, y)` of the stamp's `[0, 0]` element.
    pub origin_px: (i64, i64),
    /// Fraction of each pixel hidden by the body, in `[0, 1]`.
    pub coverage: Array2<f64>,
    /// Mean electrons per pixel from the body over the exposure.
    pub electrons: Array2<f64>,
}

impl BodyStamp {
    /// Rasterise `surface` over the disk described by `geometry`, blur
    /// with `psf`, and return the stamps. `oversampling` sub-samples per
    /// pixel edge (at least 2; 4 is a good default).
    pub fn build(
        geometry: &StampGeometry,
        surface: &dyn SurfaceRadiance,
        atmosphere: Option<&dyn AtmosphereRadiance>,
        psf: &PixelScaledAiryDisk,
        oversampling: usize,
    ) -> Self {
        let s = oversampling.max(2);
        let inv_j = geometry
            .jacobian
            .try_inverse()
            .expect("projection Jacobian must be invertible");
        let sr_per_px = inv_j.determinant().abs();
        let sr_per_sub = sr_per_px / (s * s) as f64;
        // Sky footprint of one sub-sample as an equivalent-area disk.
        let sub_radius_rad = (sr_per_sub / PI).sqrt();

        // Atmosphere table for this frame's Sun direction; the shell
        // extends the footprint beyond the solid disk.
        let lut = atmosphere.map(|atm| {
            AtmosphereLut::build(
                atm,
                geometry.radius_km,
                geometry.sun_direction_sky,
                ATMOSPHERE_LUT_GROUND,
                ATMOSPHERE_LUT_LIMB,
                ATMOSPHERE_LUT_AZIMUTH,
            )
        });
        let top_ratio = lut.as_ref().map_or(1.0, |l| l.top_ratio);
        let top_ratio2 = top_ratio * top_ratio;

        // Disk radius in pixels along the larger axis, for the footprint.
        let px_per_rad = geometry.jacobian.norm();
        let radius_px = geometry.semi_diameter_rad * px_per_rad;
        let kernel = psf_kernel(psf);
        let kernel_half = (kernel.dim().0 / 2) as i64;
        let half = (radius_px * top_ratio).ceil() as i64 + kernel_half + 1;
        let size = (2 * half + 1) as usize;

        let cx = geometry.center_px.0;
        let cy = geometry.center_px.1;
        let ox = cx.round() as i64 - half;
        let oy = cy.round() as i64 - half;

        let mut coverage = Array2::zeros((size, size));
        let mut electrons = Array2::zeros((size, size));
        let theta = geometry.semi_diameter_rad;
        let sun = geometry.sun_direction_sky;
        let sub_weight = 1.0 / (s * s) as f64;

        for row in 0..size {
            let py = (oy + row as i64) as f64;
            for col in 0..size {
                let px = (ox + col as i64) as f64;
                let mut cov = 0.0;
                let mut e = 0.0;
                for sy in 0..s {
                    let dy = py + (sy as f64 + 0.5) / s as f64 - 0.5 - cy;
                    for sx in 0..s {
                        let dx = px + (sx as f64 + 0.5) / s as f64 - 0.5 - cx;
                        let sky = inv_j * Vector2::new(dx, dy);
                        let xi = sky.x / theta;
                        let eta = sky.y / theta;
                        let rho2 = xi * xi + eta * eta;
                        if rho2 > top_ratio2 {
                            continue;
                        }
                        if rho2 > 1.0 {
                            // Through the atmosphere shell only: the sky
                            // behind shows through (1 − T) and the air
                            // adds its single-scattered path radiance.
                            if let Some(lut) = &lut {
                                let [t, path] = lut.limb(xi, eta);
                                cov += (1.0 - t) * sub_weight;
                                e += path * geometry.electrons_per_sr * sr_per_sub;
                            }
                            continue;
                        }
                        cov += sub_weight;
                        let mu = (1.0 - rho2).sqrt();
                        let normal = Vector3::new(xi, eta, mu);
                        let mu0 = normal.dot(&sun);
                        let point = SurfacePoint {
                            mu0,
                            mu,
                            alpha: geometry.phase_angle,
                            body_fixed: geometry.sky_to_body_fixed * normal,
                            sky_radius_rad: sub_radius_rad,
                        };
                        let r = surface.reflectance(&point);
                        let r_seen = match &lut {
                            // Surface light crosses the air twice; the air
                            // between ground and observer glows on top.
                            Some(lut) => {
                                let [t_view, t_sun, path] = lut.ground(xi, eta);
                                r * t_sun * t_view + path
                            }
                            None => r,
                        };
                        e += r_seen * geometry.electrons_per_sr * sr_per_sub;
                    }
                }
                coverage[[row, col]] = cov;
                electrons[[row, col]] = e;
            }
        }

        let options = ConvolveOptions {
            mode: ConvolveMode::Same,
        };
        let mut coverage = convolve2d(&coverage.view(), &kernel.view(), Some(options));
        let electrons = convolve2d(&electrons.view(), &kernel.view(), Some(options));
        coverage.mapv_inplace(|c| c.clamp(0.0, 1.0));

        Self {
            origin_px: (ox, oy),
            coverage,
            electrons,
        }
    }

    /// Total mean electrons in the stamp.
    pub fn total_electrons(&self) -> f64 {
        self.electrons.sum()
    }

    /// Convert to a compositing layer relative to a buffer whose
    /// `[0, 0]` element is sensor pixel `roi_origin` `(col, row)`.
    pub fn into_composite(self, roi_origin: (usize, usize)) -> BodyComposite {
        BodyComposite::new(
            (
                self.origin_px.0 - roi_origin.0 as i64,
                self.origin_px.1 - roi_origin.1 as i64,
            ),
            self.coverage,
            self.electrons,
        )
    }
}

/// Unit-sum PSF kernel sampled the same way star deposits are (3×3
/// Simpson rule per pixel), sized to twice the first zero.
pub fn psf_kernel(psf: &PixelScaledAiryDisk) -> Array2<f64> {
    let half = (psf.first_zero() * 2.0).ceil().max(1.0) as i64;
    let size = (2 * half + 1) as usize;
    let mut kernel = Array2::zeros((size, size));
    for row in 0..size {
        for col in 0..size {
            let dx = col as f64 - half as f64;
            let dy = row as f64 - half as f64;
            kernel[[row, col]] = psf.pixel_flux_simpson(dx, dy, 1.0);
        }
    }
    let sum = kernel.sum();
    if sum > 0.0 {
        kernel /= sum;
    }
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::brdf::{disk_integrated_reflectance, Lambert};
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use shared::units::{LengthExt, Wavelength};

    const RAD_PER_PX: f64 = 1e-6;

    fn geometry(radius_px: f64, phase_deg: f64) -> StampGeometry {
        let alpha = phase_deg.to_radians();
        StampGeometry {
            center_px: (40.3, 39.6),
            // Rows grow south: north maps to −y.
            jacobian: Matrix2::new(1.0 / RAD_PER_PX, 0.0, 0.0, -1.0 / RAD_PER_PX),
            semi_diameter_rad: radius_px * RAD_PER_PX,
            sun_direction_sky: Vector3::new(alpha.sin(), 0.0, alpha.cos()),
            phase_angle: alpha,
            electrons_per_sr: 1e15,
            sky_to_body_fixed: Matrix3::identity(),
            radius_km: 6378.1366,
        }
    }

    fn psf() -> PixelScaledAiryDisk {
        PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0))
    }

    #[test]
    fn psf_kernel_sums_to_one_and_is_centred() {
        let k = psf_kernel(&psf());
        assert_relative_eq!(k.sum(), 1.0, max_relative = 1e-12);
        let (rows, cols) = k.dim();
        assert_eq!(rows, cols);
        let c = rows / 2;
        assert!(k[[c, c]] > k[[c, c + 1]]);
        assert_relative_eq!(k[[c, c + 1]], k[[c, c - 1]], max_relative = 1e-12);
    }

    #[test]
    fn coverage_conserves_disk_area_and_saturates_inside() {
        let radius_px = 12.0;
        let stamp = BodyStamp::build(
            &geometry(radius_px, 0.0),
            &Lambert { albedo: 1.0 },
            None,
            &psf(),
            4,
        );
        let area = stamp.coverage.sum();
        assert_relative_eq!(area, PI * radius_px * radius_px, max_relative = 0.01);
        let (rows, cols) = stamp.coverage.dim();
        let centre = stamp.coverage[[rows / 2, cols / 2]];
        assert_abs_diff_eq!(centre, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(stamp.coverage[[0, 0]], 0.0, epsilon = 1e-9);
        assert!(stamp.coverage.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }

    #[test]
    fn electrons_match_disk_integrated_reflectance() {
        // Σ e = electrons_per_sr · θ² · ∫ r μ dΩ for any BRDF.
        let lambert = Lambert { albedo: 0.3 };
        for &phase_deg in &[0.0_f64, 60.0, 120.0] {
            let g = geometry(15.0, phase_deg);
            let stamp = BodyStamp::build(&g, &lambert, None, &psf(), 4);
            let expected = g.electrons_per_sr
                * g.semi_diameter_rad.powi(2)
                * disk_integrated_reflectance(&lambert, g.phase_angle, 300);
            assert_relative_eq!(stamp.total_electrons(), expected, max_relative = 0.01);
        }
    }

    #[test]
    fn crescent_is_brighter_on_the_sunward_side() {
        // Sun toward +east (+x pixel with this Jacobian): the eastern
        // half of the stamp carries the light at 90° phase.
        let g = geometry(15.0, 90.0);
        let stamp = BodyStamp::build(&g, &Lambert { albedo: 0.5 }, None, &psf(), 4);
        let cols = stamp.electrons.dim().1;
        let west: f64 = stamp.electrons.slice(ndarray::s![.., ..cols / 2]).sum();
        let east: f64 = stamp.electrons.slice(ndarray::s![.., cols / 2 + 1..]).sum();
        assert!(east > 10.0 * west, "east {east} west {west}");
    }

    #[test]
    fn stamp_origin_places_centre_at_requested_pixel() {
        let g = geometry(6.0, 0.0);
        let stamp = BodyStamp::build(&g, &Lambert { albedo: 1.0 }, None, &psf(), 4);
        let (rows, cols) = stamp.coverage.dim();
        let centre_x = stamp.origin_px.0 as f64 + (cols / 2) as f64;
        let centre_y = stamp.origin_px.1 as f64 + (rows / 2) as f64;
        assert_eq!(centre_x, g.center_px.0.round());
        assert_eq!(centre_y, g.center_px.1.round());
        // Brightness-weighted centroid recovers the sub-pixel centre.
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut total = 0.0;
        for ((r, c), &e) in stamp.electrons.indexed_iter() {
            sx += e * (stamp.origin_px.0 + c as i64) as f64;
            sy += e * (stamp.origin_px.1 + r as i64) as f64;
            total += e;
        }
        assert_abs_diff_eq!(sx / total, g.center_px.0, epsilon = 0.02);
        assert_abs_diff_eq!(sy / total, g.center_px.1, epsilon = 0.02);
    }

    fn bound_earth_air() -> crate::atmosphere::BoundRayleigh {
        use crate::atmosphere::{BoundRayleigh, RayleighAtmosphere};
        let solar = crate::photometry::solar::TsisSolarSpectrum::load().unwrap();
        let qe = crate::hardware::sensor::create_flat_qe(0.5);
        BoundRayleigh::bind(RayleighAtmosphere::earth(6378.1366), &qe, &solar, 6).unwrap()
    }

    #[test]
    fn atmosphere_extends_coverage_past_the_solid_disk() {
        let g = geometry(60.0, 30.0);
        let air = bound_earth_air();
        let bare = BodyStamp::build(&g, &Lambert { albedo: 0.3 }, None, &psf(), 3);
        let hazy = BodyStamp::build(&g, &Lambert { albedo: 0.3 }, Some(&air), &psf(), 3);
        // The shell (100 km on a 6378 km radius at 60 px) adds ~1 px of
        // partial coverage all round: total coverage grows, but by less
        // than a full extra ring.
        let extra = hazy.coverage.sum() - bare.coverage.sum();
        assert!(extra > 0.0, "shell should add coverage");
        assert!(
            extra < 2.0 * PI * 62.0 * 2.0,
            "shell coverage too large: {extra}"
        );
        // Just outside the geometric limb the bare stamp is dark and the
        // hazy one is not (limb glow), before PSF blur spreads both.
        let (rows, cols) = hazy.coverage.dim();
        let (r0, c0) = (rows / 2, cols / 2);
        assert!(hazy.electrons[[r0, c0 + 61]] > 0.0);
        assert!(hazy.electrons[[r0, c0 + 61]] > bare.electrons[[r0, c0 + 61]]);
    }

    #[test]
    fn twilight_glow_reaches_past_the_terminator() {
        // Thin crescent: Sun 150° from the observer, toward +east. On the
        // east–west axis the terminator sits at ξ = 0.867 (60° from
        // disk centre); air 100 km up stays sunlit to ~10° past it.
        let g = geometry(200.0, 150.0);
        let air = bound_earth_air();
        let bare = BodyStamp::build(&g, &Lambert { albedo: 0.3 }, None, &psf(), 3);
        let hazy = BodyStamp::build(&g, &Lambert { albedo: 0.3 }, Some(&air), &psf(), 3);
        let (rows, cols) = hazy.electrons.dim();
        // 4° past the terminator on the night side (ξ = 0.83): the ground
        // is dark and beyond the PSF's reach from the lit crescent, but
        // the air above it is still sunlit and scatters toward us.
        let twilight = [rows / 2, cols / 2 + 166];
        assert!(bare.electrons[twilight] < 1e-7 * bare.total_electrons());
        assert!(
            hazy.electrons[twilight] > 100.0 * bare.electrons[twilight].max(1e-30),
            "hazy {} bare {}",
            hazy.electrons[twilight],
            bare.electrons[twilight]
        );
        // Deep in the night, opposite the Sun, both are dark.
        let midnight = [rows / 2, cols / 2 - 100];
        assert!(hazy.electrons[midnight] < 1e-7 * hazy.total_electrons());
        // For a thin crescent the sunlit air (crescent, twilight band and
        // limb ring) rivals or exceeds the dark-albedo surface, but single
        // scattering cannot add more than a few times the surface light.
        let ratio = hazy.total_electrons() / bare.total_electrons();
        assert!((1.0..6.0).contains(&ratio), "ratio {ratio}");
    }

    #[test]
    fn into_composite_shifts_into_roi_coordinates() {
        let g = geometry(4.0, 0.0);
        let stamp = BodyStamp::build(&g, &Lambert { albedo: 1.0 }, None, &psf(), 2);
        let origin = stamp.origin_px;
        let layer = stamp.into_composite((10, 20));
        assert_eq!(layer.origin, (origin.0 - 10, origin.1 - 20));
    }
}

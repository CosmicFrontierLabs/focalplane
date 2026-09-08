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

use nalgebra::{Matrix2, Vector2, Vector3};
use ndarray::Array2;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::image_proc::convolve2d::{convolve2d, ConvolveMode, ConvolveOptions};

use crate::bodies::brdf::Brdf;
use crate::image_proc::compose::BodyComposite;

/// Default sub-samples per pixel edge.
pub const DEFAULT_OVERSAMPLING: usize = 4;

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
    /// Rasterise `brdf` over the disk described by `geometry`, blur with
    /// `psf`, and return the stamps. `oversampling` sub-samples per pixel
    /// edge (at least 2; 4 is a good default).
    pub fn build(
        geometry: &StampGeometry,
        brdf: &dyn Brdf,
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

        // Disk radius in pixels along the larger axis, for the footprint.
        let px_per_rad = geometry.jacobian.norm();
        let radius_px = geometry.semi_diameter_rad * px_per_rad;
        let kernel = psf_kernel(psf);
        let kernel_half = (kernel.dim().0 / 2) as i64;
        let half = radius_px.ceil() as i64 + kernel_half + 1;
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
                        if rho2 > 1.0 {
                            continue;
                        }
                        cov += sub_weight;
                        let mu = (1.0 - rho2).sqrt();
                        let normal = Vector3::new(xi, eta, mu);
                        let mu0 = normal.dot(&sun);
                        let r = brdf.reflectance(mu0, mu, geometry.phase_angle);
                        e += r * geometry.electrons_per_sr * sr_per_sub;
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
    use std::f64::consts::PI;

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
            let stamp = BodyStamp::build(&g, &lambert, &psf(), 4);
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
        let stamp = BodyStamp::build(&g, &Lambert { albedo: 0.5 }, &psf(), 4);
        let cols = stamp.electrons.dim().1;
        let west: f64 = stamp.electrons.slice(ndarray::s![.., ..cols / 2]).sum();
        let east: f64 = stamp.electrons.slice(ndarray::s![.., cols / 2 + 1..]).sum();
        assert!(east > 10.0 * west, "east {east} west {west}");
    }

    #[test]
    fn stamp_origin_places_centre_at_requested_pixel() {
        let g = geometry(6.0, 0.0);
        let stamp = BodyStamp::build(&g, &Lambert { albedo: 1.0 }, &psf(), 4);
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

    #[test]
    fn into_composite_shifts_into_roi_coordinates() {
        let g = geometry(4.0, 0.0);
        let stamp = BodyStamp::build(&g, &Lambert { albedo: 1.0 }, &psf(), 2);
        let origin = stamp.origin_px;
        let layer = stamp.into_composite((10, 20));
        assert_eq!(layer.origin, (origin.0 - 10, origin.1 - 20));
    }
}

//! Spacecraft orientation utilities built on [`nalgebra::UnitQuaternion`].
//!
//! The body frame used throughout the simulator is:
//! - `+Z` points along the telescope boresight (outward).
//! - `+X` points toward local "east": perpendicular to the celestial pole
//!   vector and oriented in the direction of increasing right ascension at
//!   the current pointing.
//! - `+Y` points toward local "north": toward increasing declination at the
//!   current pointing.
//!
//! The roll angle is a rotation about the body `+Z` axis following the
//! right-hand rule; a positive roll rotates the body `+X` axis toward the
//! body `+Y` axis in the body frame.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use starfield::Equatorial;

/// Numerical threshold (in radians) below which a pointing is considered to
/// be "at the pole" for roll extraction purposes.
const POLE_EPSILON: f64 = 1e-9;

/// Build the world-from-body orientation quaternion for a given pointing
/// (boresight direction) and roll.
///
/// The base (roll = 0) rotation is constructed so that the body frame
/// columns map directly onto the world-frame local triad at the pointing:
/// body `+X` -> local east, body `+Y` -> local north, body `+Z` ->
/// boresight. A final rotation about the body `+Z` axis applies the roll
/// using the right-hand rule.
pub fn orientation_from_pointing(eq: &Equatorial, roll_rad: f64) -> UnitQuaternion<f64> {
    let ra = eq.ra;
    let dec = eq.dec;
    let (sin_ra, cos_ra) = ra.sin_cos();
    let (sin_dec, cos_dec) = dec.sin_cos();

    // World-frame local triad at the pointing.
    let east = Vector3::new(-sin_ra, cos_ra, 0.0);
    let north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);
    let bore = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);

    // Columns are (east, north, bore) so body +X -> east, +Y -> north, +Z -> bore.
    let m = Matrix3::from_columns(&[east, north, bore]);
    let base = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(m));
    let twist = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), roll_rad);

    base * twist
}

/// Extract the boresight (pointing) direction of an orientation quaternion
/// as an [`Equatorial`] coordinate.
pub fn boresight_of(q: &UnitQuaternion<f64>) -> Equatorial {
    let z = q * Vector3::z();
    let ra = z.y.atan2(z.x);
    let r_xy = (z.x * z.x + z.y * z.y).sqrt();
    let dec = z.z.atan2(r_xy);
    Equatorial::new(ra, dec)
}

/// Extract the roll angle (about the body `+Z` axis) from an orientation
/// quaternion. Returns `0.0` when the boresight is within [`POLE_EPSILON`]
/// of a celestial pole, where the east direction degenerates.
///
/// Uses the convention that the "canonical no-roll" body `+X` vector at the
/// current boresight is the local east direction
/// `(-sin(ra), cos(ra), 0)`; the returned roll is the signed angle (about
/// the boresight) from that canonical vector to the body `+X` vector rotated
/// into the world frame by `q`.
pub fn roll_of(q: &UnitQuaternion<f64>) -> f64 {
    let bore = boresight_of(q);
    if (std::f64::consts::FRAC_PI_2 - bore.dec.abs()).abs() < POLE_EPSILON {
        return 0.0;
    }

    // Body +X rotated to world frame.
    let ex_world = q * Vector3::x();

    // Canonical no-roll east direction at the current boresight.
    let east = Vector3::new(-bore.ra.sin(), bore.ra.cos(), 0.0);

    // Boresight direction for signing the angle via right-hand rule.
    let bore_vec = Vector3::new(
        bore.dec.cos() * bore.ra.cos(),
        bore.dec.cos() * bore.ra.sin(),
        bore.dec.sin(),
    );

    let cos_roll = east.dot(&ex_world).clamp(-1.0, 1.0);
    let cross = east.cross(&ex_world);
    let sin_roll = cross.dot(&bore_vec);
    sin_roll.atan2(cos_roll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn eq_deg(ra_deg: f64, dec_deg: f64) -> Equatorial {
        Equatorial::from_degrees(ra_deg, dec_deg)
    }

    #[test]
    fn test_orientation_from_pointing_roundtrip() {
        let samples = [
            eq_deg(0.0, 0.0),
            eq_deg(45.0, 30.0),
            eq_deg(123.5, -42.0),
            eq_deg(359.9, 89.9),
            eq_deg(10.0, -89.9),
            eq_deg(270.0, 0.0),
        ];
        for eq in samples.iter() {
            let q = orientation_from_pointing(eq, 0.0);
            let out = boresight_of(&q);
            // Compare via cartesian vectors so RA wrap at the pole is a non-issue.
            let expected = Vector3::new(
                eq.dec.cos() * eq.ra.cos(),
                eq.dec.cos() * eq.ra.sin(),
                eq.dec.sin(),
            );
            let actual = Vector3::new(
                out.dec.cos() * out.ra.cos(),
                out.dec.cos() * out.ra.sin(),
                out.dec.sin(),
            );
            assert_relative_eq!(expected.x, actual.x, epsilon = 1e-9);
            assert_relative_eq!(expected.y, actual.y, epsilon = 1e-9);
            assert_relative_eq!(expected.z, actual.z, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_roll_of_recovers_roll() {
        let samples = [
            eq_deg(0.0, 0.0),
            eq_deg(45.0, 30.0),
            eq_deg(180.0, -15.0),
            eq_deg(300.0, 60.0),
        ];
        let rolls = [-PI * 0.9, -FRAC_PI_2, -0.5, 0.0, 0.5, FRAC_PI_2, PI * 0.9];
        for eq in samples.iter() {
            for &r in rolls.iter() {
                let q = orientation_from_pointing(eq, r);
                let r_out = roll_of(&q);
                assert_relative_eq!(r_out, r, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_roll_of_near_pole_returns_zero() {
        let eq = Equatorial::new(1.2, FRAC_PI_2 - 1e-12);
        let q = orientation_from_pointing(&eq, 0.7);
        // Pole guard returns 0 regardless of constructed roll.
        assert_eq!(roll_of(&q), 0.0);
    }

    #[test]
    fn test_body_x_is_local_east_at_zero_roll() {
        // With roll=0, body +X should map to the local east vector.
        let eq = eq_deg(45.0, 20.0);
        let q = orientation_from_pointing(&eq, 0.0);
        let ex_world = q * Vector3::x();
        let east = Vector3::new(-eq.ra.sin(), eq.ra.cos(), 0.0);
        assert_relative_eq!(ex_world.x, east.x, epsilon = 1e-9);
        assert_relative_eq!(ex_world.y, east.y, epsilon = 1e-9);
        assert_relative_eq!(ex_world.z, east.z, epsilon = 1e-9);
    }
}

//! Absolute time for a rendered scene.
//!
//! The renderer measures everything else in [`Duration`] from the start
//! of a trajectory. Solar-system bodies need an absolute epoch so their
//! ephemerides can be evaluated. [`Epoch`] wraps starfield's [`Time`]
//! (which carries the full UTC/TAI/TT/TDB conversion chain) and adds the
//! small amount of plumbing the simulator needs: parsing from the
//! command line, offsetting by a trajectory duration, and serialising
//! into `metadata.json`.
//!
//! Epochs serialise as `{ "jd_tdb": <f64>, "utc": "<iso>" }`; only
//! `jd_tdb` is read back. A Julian date in `f64` resolves to about
//! 0.1 ms, far finer than any light-time or aberration term the
//! renderer evaluates.

use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use starfield::time::{Time, Timescale};
use thiserror::Error;

/// Seconds in one Julian day.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// Errors from parsing an [`Epoch`] out of user input.
#[derive(Debug, Error)]
pub enum EpochError {
    /// The string was neither RFC 3339 nor a `JD <number>` literal.
    #[error(
        "cannot parse epoch {input:?}: expected RFC 3339 (2027-06-01T00:00:00Z) or 'JD 2461558.5'"
    )]
    Unparseable {
        /// Offending input.
        input: String,
    },
}

/// An absolute instant, carried as a starfield [`Time`].
#[derive(Clone, Debug)]
pub struct Epoch {
    time: Time,
}

impl Epoch {
    /// Epoch from a UTC calendar instant.
    pub fn from_utc(dt: DateTime<Utc>) -> Self {
        Self {
            time: Timescale::default().from_datetime(dt),
        }
    }

    /// Epoch from a Barycentric Dynamical Time Julian date (the time
    /// argument JPL ephemerides use).
    pub fn from_jd_tdb(jd: f64) -> Self {
        Self {
            time: Timescale::default().tdb_jd(jd),
        }
    }

    /// Parse either an RFC 3339 UTC string (`2027-06-01T00:00:00Z`) or
    /// a Julian date literal prefixed with `JD` (`JD 2461558.5`, read as
    /// TDB).
    pub fn parse(input: &str) -> Result<Self, EpochError> {
        let trimmed = input.trim();
        if let Some(rest) = trimmed
            .strip_prefix("JD")
            .or_else(|| trimmed.strip_prefix("jd"))
        {
            if let Ok(jd) = rest.trim().parse::<f64>() {
                return Ok(Self::from_jd_tdb(jd));
            }
        }
        DateTime::parse_from_rfc3339(trimmed)
            .map(|dt| Self::from_utc(dt.with_timezone(&Utc)))
            .map_err(|_| EpochError::Unparseable {
                input: input.to_string(),
            })
    }

    /// The underlying starfield time.
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Julian date in Barycentric Dynamical Time.
    pub fn jd_tdb(&self) -> f64 {
        self.time.tdb()
    }

    /// This epoch shifted later by `offset`.
    pub fn offset(&self, offset: Duration) -> Self {
        self.offset_secs(offset.as_secs_f64())
    }

    /// This epoch shifted by a signed number of seconds.
    pub fn offset_secs(&self, seconds: f64) -> Self {
        Self {
            time: self.time.clone() + seconds / SECONDS_PER_DAY,
        }
    }

    /// Signed seconds from `self` to `other`.
    pub fn seconds_until(&self, other: &Epoch) -> f64 {
        (other.jd_tdb() - self.jd_tdb()) * SECONDS_PER_DAY
    }

    /// UTC calendar string rounded to the millisecond, or the TDB Julian
    /// date if the UTC conversion is unavailable.
    pub fn utc_string(&self) -> String {
        // starfield's `utc_iso` truncates the fractional second, so bias
        // by half a millisecond to round to the nearest one.
        let rounded = self.time.clone() + 0.5e-3 / SECONDS_PER_DAY;
        rounded
            .utc_iso('T', 3)
            .unwrap_or_else(|_| format!("JD {:.8} TDB", self.jd_tdb()))
    }
}

impl PartialEq for Epoch {
    fn eq(&self, other: &Self) -> bool {
        self.jd_tdb() == other.jd_tdb()
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.utc_string())
    }
}

impl Serialize for Epoch {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Epoch", 2)?;
        state.serialize_field("jd_tdb", &self.jd_tdb())?;
        state.serialize_field("utc", &self.utc_string())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Epoch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EpochVisitor;

        impl<'de> Visitor<'de> for EpochVisitor {
            type Value = Epoch;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object with a numeric jd_tdb field")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Epoch, M::Error> {
                let mut jd_tdb: Option<f64> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "jd_tdb" => jd_tdb = Some(map.next_value()?),
                        _ => {
                            let _ignored: de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                jd_tdb
                    .map(Epoch::from_jd_tdb)
                    .ok_or_else(|| de::Error::missing_field("jd_tdb"))
            }
        }

        deserializer.deserialize_struct("Epoch", &["jd_tdb", "utc"], EpochVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// J2000.0 is 2000-01-01T12:00:00 TT, JD 2451545.0 TT; TDB differs
    /// from TT by under 2 ms.
    #[test]
    fn j2000_round_trips_through_jd_tdb() {
        let epoch = Epoch::from_jd_tdb(2_451_545.0);
        assert_abs_diff_eq!(epoch.jd_tdb(), 2_451_545.0, epsilon = 1e-9);
    }

    #[test]
    fn utc_parse_matches_known_julian_date() {
        // 2000-01-01T12:00:00 UTC is JD 2451545.0 UTC; TT is 64.184 s
        // ahead, so JD(TDB) ≈ 2451545.0 + 64.184 / 86400.
        let epoch = Epoch::parse("2000-01-01T12:00:00Z").unwrap();
        let expected = 2_451_545.0 + 64.184 / SECONDS_PER_DAY;
        assert_abs_diff_eq!(epoch.jd_tdb(), expected, epsilon = 2e-3 / SECONDS_PER_DAY);
    }

    #[test]
    fn jd_literal_parses_with_and_without_space() {
        let a = Epoch::parse("JD 2461558.5").unwrap();
        let b = Epoch::parse("jd2461558.5").unwrap();
        assert_abs_diff_eq!(a.jd_tdb(), 2_461_558.5, epsilon = 1e-9);
        assert_eq!(a, b);
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(matches!(
            Epoch::parse("next tuesday"),
            Err(EpochError::Unparseable { .. })
        ));
    }

    #[test]
    fn offsets_advance_by_the_requested_seconds() {
        let base = Epoch::from_jd_tdb(2_461_558.5);
        let later = base.offset(Duration::from_secs(3_600));
        assert_abs_diff_eq!(base.seconds_until(&later), 3_600.0, epsilon = 1e-4);
        let earlier = base.offset_secs(-90.0);
        assert_abs_diff_eq!(base.seconds_until(&earlier), -90.0, epsilon = 1e-4);
    }

    #[test]
    fn serde_round_trip_preserves_jd() {
        let epoch = Epoch::parse("2027-06-01T00:00:00Z").unwrap();
        let json = serde_json::to_string(&epoch).unwrap();
        assert!(json.contains("jd_tdb"));
        assert!(json.contains("utc"));
        let back: Epoch = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.jd_tdb(), epoch.jd_tdb(), epsilon = 1e-9);
    }

    #[test]
    fn display_is_utc_calendar() {
        let epoch = Epoch::parse("2027-06-01T00:00:00Z").unwrap();
        assert!(epoch.to_string().starts_with("2027-06-01T00:00:00"));
    }

    #[test]
    fn utc_round_trips_across_leap_second_history() {
        for iso in [
            "2007-10-03T05:30:00Z",
            "1999-12-31T23:59:59Z",
            "2016-12-31T12:00:00Z",
        ] {
            let epoch = Epoch::parse(iso).unwrap();
            let shown = epoch.utc_string();
            assert!(shown.starts_with(&iso[..19]), "{iso} displayed as {shown}");
        }
    }
}

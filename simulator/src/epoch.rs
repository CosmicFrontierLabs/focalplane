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
//! Parsing, offsets and serialisation use starfield's own `Time`
//! support. Epochs serialise in starfield's lossless form, which carries
//! the split Julian day exactly plus informational `jd_tdb` and `utc`
//! fields; on input, that form, a bare `{ "jd_tdb": ... }` map (as older
//! `metadata.json` files hold) or a time string are all accepted.

use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use starfield::time::{Time, Timescale};
use thiserror::Error;

/// Errors from parsing an [`Epoch`] out of user input.
#[derive(Debug, Error)]
pub enum EpochError {
    /// The string was neither an ISO 8601 UTC instant nor a Julian date
    /// literal.
    #[error(
        "cannot parse epoch {input:?}: expected RFC 3339 (2027-06-01T00:00:00Z) or 'JD 2461558.5'"
    )]
    Unparseable {
        /// Offending input.
        input: String,
    },
}

/// An absolute instant, carried as a starfield [`Time`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
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

    /// Parse an ISO 8601 / RFC 3339 UTC instant (`2027-06-01T00:00:00Z`)
    /// or a Julian date literal (`JD 2461558.5 TDB`; with no timescale,
    /// `JD 2461558.5` is read as TDB). See `starfield::time` for the full
    /// grammar, including leap-second validation of `23:59:60`.
    pub fn parse(input: &str) -> Result<Self, EpochError> {
        let ts = Timescale::default();
        let trimmed = input.trim();
        ts.parse(trimmed)
            .or_else(|err| {
                let is_bare_jd = trimmed
                    .get(..2)
                    .is_some_and(|p| p.eq_ignore_ascii_case("jd"))
                    && trimmed[2..].trim().parse::<f64>().is_ok();
                if is_bare_jd {
                    ts.parse(&format!("{trimmed} TDB"))
                } else {
                    Err(err)
                }
            })
            .map(|time| Self { time })
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

    /// This epoch shifted by a signed number of SI seconds.
    pub fn offset_secs(&self, seconds: f64) -> Self {
        Self {
            time: self.time.add_seconds(seconds),
        }
    }

    /// Signed SI seconds (elapsed TT) from `self` to `other`.
    pub fn seconds_until(&self, other: &Epoch) -> f64 {
        other.time.seconds_since(&self.time)
    }

    /// UTC calendar string rounded to the millisecond, or the TDB Julian
    /// date if the UTC conversion is unavailable.
    pub fn utc_string(&self) -> String {
        self.time
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use starfield::constants::DAY_S;

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
        let expected = 2_451_545.0 + 64.184 / DAY_S;
        assert_abs_diff_eq!(epoch.jd_tdb(), expected, epsilon = 2e-3 / DAY_S);
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
        assert_eq!(back.jd_tdb().to_bits(), epoch.jd_tdb().to_bits());
    }

    #[test]
    fn older_metadata_with_only_jd_tdb_still_loads() {
        let back: Epoch =
            serde_json::from_str(r#"{"jd_tdb": 2461558.5, "utc": "2027-06-01T23:58:50.816Z"}"#)
                .unwrap();
        assert_abs_diff_eq!(back.jd_tdb(), 2_461_558.5, epsilon = 1e-9);
    }

    #[test]
    fn explicit_timescale_literals_parse() {
        let tt = Epoch::parse("JD 2451545.0 TT").unwrap();
        assert_abs_diff_eq!(tt.time().tt(), 2_451_545.0, epsilon = 1e-12);
        assert!(Epoch::parse("2027-06-01T23:59:60Z").is_err());
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

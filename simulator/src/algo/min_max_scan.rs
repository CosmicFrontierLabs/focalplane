//! MinMaxScan - A struct for scanning minimum and maximum values in floating point data
//!
//! This struct computes and stores the minimum and maximum values from a vector of floating
//! point numbers. It handles NaN detection and returns errors when NaN values are encountered.

use num_traits::float::Float;
use std::fmt;
use thiserror::Error;

/// Error types for MinMaxScan operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MinMaxError {
    #[error("NaN value encountered at index {0}")]
    NaNEncountered(usize),
    #[error("No data provided (empty slice)")]
    NoData,
}

/// A scanner for minimum and maximum values in floating point data
#[derive(Debug, Clone)]
pub struct MinMaxScan<T: Float> {
    min_value: Option<T>,
    max_value: Option<T>,
    nan_index: Option<usize>,
}

impl<T: Float + fmt::Debug> MinMaxScan<T> {
    /// Create a new MinMaxScan by computing min and max from a slice of values
    ///
    /// If any NaN values are encountered, the has_nan flag will be set internally.
    ///
    /// # Arguments
    /// * `data` - A slice of floating point values
    ///
    /// # Example
    /// ```
    /// use simulator::algo::min_max_scan::MinMaxScan;
    ///
    /// let scanner = MinMaxScan::<f64>::new(&[1.0, 5.0, 3.0, 2.0]);
    /// assert_eq!(scanner.min().unwrap(), 1.0);
    /// assert_eq!(scanner.max().unwrap(), 5.0);
    /// ```
    pub fn new(data: &[T]) -> Self {
        let mut min_value = None;
        let mut max_value = None;
        let mut nan_index = None;

        for (index, &value) in data.iter().enumerate() {
            if value.is_nan() {
                if nan_index.is_none() {
                    nan_index = Some(index);
                }
                // Continue processing to find valid min/max if they exist
                continue;
            }

            match (min_value, max_value) {
                (None, None) => {
                    min_value = Some(value);
                    max_value = Some(value);
                }
                (Some(min), Some(max)) => {
                    if value < min {
                        min_value = Some(value);
                    }
                    if value > max {
                        max_value = Some(value);
                    }
                }
                _ => unreachable!("min and max should always be in sync"),
            }
        }

        Self {
            min_value,
            max_value,
            nan_index,
        }
    }

    /// Get the minimum value
    ///
    /// # Returns
    /// * `Ok(T)` - The minimum value if data has been computed and no NaN was found
    /// * `Err(MinMaxError::NaNEncountered(index))` - If NaN values were encountered
    /// * `Err(MinMaxError::NoData)` - If no data has been computed yet
    pub fn min(&self) -> Result<T, MinMaxError> {
        if let Some(index) = self.nan_index {
            Err(MinMaxError::NaNEncountered(index))
        } else {
            self.min_value.ok_or(MinMaxError::NoData)
        }
    }

    /// Get the maximum value
    ///
    /// # Returns
    /// * `Ok(T)` - The maximum value if data has been computed and no NaN was found
    /// * `Err(MinMaxError::NaNEncountered(index))` - If NaN values were encountered
    /// * `Err(MinMaxError::NoData)` - If no data has been computed yet
    pub fn max(&self) -> Result<T, MinMaxError> {
        if let Some(index) = self.nan_index {
            Err(MinMaxError::NaNEncountered(index))
        } else {
            self.max_value.ok_or(MinMaxError::NoData)
        }
    }

    /// Get both min and max values as a tuple
    ///
    /// # Returns
    /// * `Ok((T, T))` - A tuple of (min, max) if data has been computed and no NaN was found
    /// * `Err(MinMaxError::NaNEncountered(index))` - If NaN values were encountered
    /// * `Err(MinMaxError::NoData)` - If no data has been computed yet
    pub fn min_max(&self) -> Result<(T, T), MinMaxError> {
        Ok((self.min()?, self.max()?))
    }

    /// Check if NaN values were encountered during computation
    pub fn has_nan(&self) -> bool {
        self.nan_index.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_min_max_f64() {
        let scanner = MinMaxScan::<f64>::new(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);

        assert_eq!(scanner.min().unwrap(), 1.0);
        assert_eq!(scanner.max().unwrap(), 9.0);
        assert_eq!(scanner.min_max().unwrap(), (1.0, 9.0));
    }

    #[test]
    fn test_basic_min_max_f32() {
        let scanner = MinMaxScan::<f32>::new(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);

        assert_eq!(scanner.min().unwrap(), 1.0);
        assert_eq!(scanner.max().unwrap(), 9.0);
    }

    #[test]
    fn test_nan_handling() {
        let scanner = MinMaxScan::<f64>::new(&[1.0, 2.0, f64::NAN, 3.0, 4.0]);

        assert!(scanner.has_nan());
        assert_eq!(scanner.min(), Err(MinMaxError::NaNEncountered(2)));
        assert_eq!(scanner.max(), Err(MinMaxError::NaNEncountered(2)));
    }

    #[test]
    fn test_all_nan() {
        let scanner = MinMaxScan::<f64>::new(&[f64::NAN, f64::NAN, f64::NAN]);

        assert!(scanner.has_nan());
        assert_eq!(scanner.min(), Err(MinMaxError::NaNEncountered(0)));
        assert_eq!(scanner.max(), Err(MinMaxError::NaNEncountered(0)));
    }

    #[test]
    fn test_no_data() {
        let scanner = MinMaxScan::<f64>::new(&[]);

        assert_eq!(scanner.min(), Err(MinMaxError::NoData));
        assert_eq!(scanner.max(), Err(MinMaxError::NoData));
    }

    #[test]
    fn test_empty_slice() {
        let scanner = MinMaxScan::<f64>::new(&[]);

        assert_eq!(scanner.min(), Err(MinMaxError::NoData));
        assert_eq!(scanner.max(), Err(MinMaxError::NoData));
    }

    #[test]
    fn test_single_value() {
        let scanner = MinMaxScan::<f64>::new(&[42.0]);

        assert_eq!(scanner.min().unwrap(), 42.0);
        assert_eq!(scanner.max().unwrap(), 42.0);
    }

    #[test]
    fn test_negative_values() {
        let scanner = MinMaxScan::<f64>::new(&[-5.0, -1.0, -10.0, -3.0]);

        assert_eq!(scanner.min().unwrap(), -10.0);
        assert_eq!(scanner.max().unwrap(), -1.0);
    }

    #[test]
    fn test_infinity_values() {
        let scanner = MinMaxScan::<f64>::new(&[1.0, f64::INFINITY, -f64::INFINITY, 5.0]);

        assert_eq!(scanner.min().unwrap(), -f64::INFINITY);
        assert_eq!(scanner.max().unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_mixed_valid_and_nan() {
        let scanner = MinMaxScan::<f64>::new(&[1.0, f64::NAN, 5.0, 3.0, f64::NAN]);

        // Even though we have valid values, we still error on NaN
        assert!(scanner.has_nan());
        assert_eq!(scanner.min(), Err(MinMaxError::NaNEncountered(1)));
        assert_eq!(scanner.max(), Err(MinMaxError::NaNEncountered(1)));
    }
}

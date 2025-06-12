use thiserror::Error;

#[derive(Error, Debug)]
pub enum InterpError {
    #[error("Value {0} is out of bounds for interpolation range [{1}, {2}]")]
    OutOfBounds(f64, f64, f64),
    #[error("Input vectors must have at least 2 points")]
    InsufficientData,
    #[error("Input vectors must have the same length")]
    MismatchedLengths,
    #[error("X values must be sorted in ascending order")]
    UnsortedData,
}

pub fn interp(x: f64, xs: &[f64], ys: &[f64]) -> Result<f64, InterpError> {
    if xs.len() != ys.len() {
        return Err(InterpError::MismatchedLengths);
    }

    if xs.len() < 2 {
        return Err(InterpError::InsufficientData);
    }

    // Check if xs is sorted
    for i in 1..xs.len() {
        if xs[i] <= xs[i - 1] {
            return Err(InterpError::UnsortedData);
        }
    }

    let min_x = xs[0];
    let max_x = xs[xs.len() - 1];

    if x < min_x || x > max_x {
        return Err(InterpError::OutOfBounds(x, min_x, max_x));
    }

    // Binary search for the correct interval
    let idx = match xs.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
        Ok(exact_idx) => return Ok(ys[exact_idx]), // Exact match
        Err(insert_idx) => insert_idx,
    };

    // Linear interpolation between points
    let i1 = idx - 1;
    let i2 = idx;

    let x1 = xs[i1];
    let x2 = xs[i2];
    let y1 = ys[i1];
    let y2 = ys[i2];

    let t = (x - x1) / (x2 - x1);
    Ok(y1 + t * (y2 - y1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(interp(2.0, &xs, &ys).unwrap(), 20.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert_eq!(interp(1.5, &xs, &ys).unwrap(), 15.0);
        assert_eq!(interp(2.5, &xs, &ys).unwrap(), 25.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert!(matches!(
            interp(0.5, &xs, &ys),
            Err(InterpError::OutOfBounds(_, _, _))
        ));
        assert!(matches!(
            interp(3.5, &xs, &ys),
            Err(InterpError::OutOfBounds(_, _, _))
        ));
    }

    #[test]
    fn test_mismatched_lengths() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0];
        assert!(matches!(
            interp(1.5, &xs, &ys),
            Err(InterpError::MismatchedLengths)
        ));
    }

    #[test]
    fn test_insufficient_data() {
        let xs = vec![1.0];
        let ys = vec![10.0];
        assert!(matches!(
            interp(1.0, &xs, &ys),
            Err(InterpError::InsufficientData)
        ));
    }

    #[test]
    fn test_unsorted_data() {
        let xs = vec![2.0, 1.0, 3.0];
        let ys = vec![20.0, 10.0, 30.0];
        assert!(matches!(
            interp(1.5, &xs, &ys),
            Err(InterpError::UnsortedData)
        ));
    }
}

//! Space telescope optical and sensor simulation
//!
//! This crate provides functionality for simulating the optical system
//! and sensor hardware of a space telescope.

pub mod image_proc;

/// Placeholder function to satisfy the compiler
pub fn placeholder() {
    println!("Simulator placeholder");
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

pub use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, PI, SQRT_2};

pub const TWO_PI: f64 = 2.0 * PI;
pub const FRAC_1_2_PI: f64 = 1.0 / TWO_PI;
pub const SQRT_2_PI: f64 = SQRT_2 * 2.0 / FRAC_2_SQRT_PI;

// When std is available, the built-in f64::exp uses intrinsics and is like 5 nanos
#[cfg(feature = "std")]
#[inline(always)]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[cfg(feature = "std")]
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

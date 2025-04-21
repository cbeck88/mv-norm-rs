//! This crate implements functions related to bivariate and multivariate normal distributions.
//!
//! Bivariate normal distribution CDF:
//!
//! * [`tvpack`] is a straight port of Alan Genz' "tvpack" fortran code, and uses no SIMD.
//! * [`BatchBvnd`] extends that to do batch evaluation significantly faster, and uses SIMD operations
//! * [`bvnd`] is a quick-start helper that you can use to evaluate the CDF at a single point, and uses SIMD operations.

#![cfg_attr(not(feature = "std"), no_std)]

mod batch_bvnd;
mod util;

pub mod tvpack;

#[cfg(test)]
mod test_utils;

pub use batch_bvnd::{BatchBvnd, bvnd};

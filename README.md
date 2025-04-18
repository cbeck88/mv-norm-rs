mvnorm
======

This rust crate contains functions related to multivariate normal distributions, especially,
computing the CDF.

It ports a subset of the R package `mvtnorm`, which is
widely used for these purposes.

Additionally, for some functions, this crate provides an alternative "batch evaluation" API.

When the `sleef` dependency is enabled, batch evaluation is "vectorized", using SIMD operations
to compute multiple evaluations simultaneously, based on rust's `core::simd` abstraction.

`sleef` requires the nightly compiler, and is off by default.

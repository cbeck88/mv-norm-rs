# mv-norm

[![Crates.io](https://img.shields.io/crates/v/mv-norm?style=flat-square)](https://crates.io/crates/mv-norm)
[![Crates.io](https://img.shields.io/crates/d/mv-norm?style=flat-square)](https://crates.io/crates/mv-norm)
[![License](https://img.shields.io/badge/license-GPL%202.0-blue?style=flat-square)](LICENSE-GPLv2)

[API Docs](https://docs.rs/mv-norm/latest)

*Fast* and *accurate* calcluations related to [multivariate normal distributions](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), in pure rust. (Note: Right now we only have the bivariate normal CDF.)

This rust crate ports a subset of the [R package `mvtnorm`](https://cran.r-project.org/web/packages/mvtnorm/mvtnorm.pdf), which is
widely used for these purposes.

Additionally, this crate provides "batch evaluation" APIs which may be much faster if you need to evaluate many points. These APIs are designed to allow easy precomputation based on some of the parameters, so that work is shared across many evaluations, and to take advantage of SIMD.

## Why?

A common practice for statistics, numerical integration, modeling, etc. is:

* Use R, or Python with Numpy, which has many high-level functions you need and a nice repl with plotting.
* Low-level statistical primitives, which are perf critical, are obtained by binding to existing C or Fortran code. These are fast, accurate, and [widely](https://cran.r-project.org/web/packages/mvtnorm/index.html) [used](https://github.com/SebastienMarmin/torch-mvnorm).

In this crate, we ported fortran code such as [Alan Genz' `tvpack` algorithm](https://github.com/cran/mvtnorm/blob/67d734c947eb10fbfa9d3431ba6a7d47241be58c/src/tvpack.f#L514), and tested against the original for fidelity.

Then we used [`wide`](https://crates.io/crates/wide) and precomputation tricks to make it significantly faster, especially in a batch evaluation, where we get more then 5x improved throughput (but run benchmarks to see if you can repro this.)

This greatly accelerated a numerical integration routine for a statistical model.

## Future Directions

* Port more of the `mvtnorm` sources, and use Rust's nice SIMD facilities to optimize them. (Eventually, the `core::simd` stuff when it is stabilized.) For example support for Genz-Bretz, or the `tvpack` trivariate normal CDF routine, would be great.
* Provide an `f32` version of algorithms, especially if it can be significantly faster.
* Allow the user to choose larger error tolerances and get faster algorithms when possible.
  Whether this should be done by providing differently named functions (see [SLEEF](https://docs.rs/sleef/latest/sleef/f64x/index.html) for example) or a "policy object" like in `boost::math`, I'm not sure.
* Expose [python](pyo3.rs/v0.24.2/) bindings and publish them.

## Licensing and Distribution

GPLv2, because we ported code from `mvtnorm` which is GPLv2 licensed.

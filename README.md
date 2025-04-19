# mv-norm

*Fast* and *accurate* calcluations related to [multivariate normal distributions](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), in pure rust. (Note: Right now we only have the bivariate normal CDF.)

This rust crate ports a subset of the [R package `mvtnorm`](https://cran.r-project.org/web/packages/mvtnorm/mvtnorm.pdf), which is
widely used for these purposes.

Additionally, this crate provides "batch evaluation" APIs which may be much faster if you need to evaluate many points. These APIs are designed to allow easy precomputation based on some of the parameters, so that work is shared across many evaluations, and to take advantage of SIMD.

## Why?

A common practice for statistics, numerical integration, modeling, etc. is:

* Use R, or Python with Numpy, which has many high-level functions you need and a nice repl with plotting.
* Low-level statistical primitives, which are perf critical, are obtained by binding to existing C or Fortran code. These are fast, accurate, and [widely](https://cran.r-project.org/web/packages/mvtnorm/index.html) [used](https://github.com/SebastienMarmin/torch-mvnorm).

Numpy itself can often vectorize things just fine.

In this crate, we've taken such C or Fortran code and ported it to pure rust. There's a few advantages:

* It is easier to write safe, portable, and fast SIMD code using existing rust abstractions (e.g. [`wide`](https://crates.io/crates/wide) and the unstable `core::simd`), with significant performance gains. In the harder cases where the C or Fortran code doesn't automatically vectorize, rust gets the job done.
* It is easier to modify these algorithms to precompute or cache data across calls, in a thread-safe way, without making unnecessary allocations, in rust, than it is to do so in C or Fortran.
* Using rust also allows you to use [`rayon`](https://docs.rs/rayon/latest/rayon/). Rayon is excellent, and far superior to alternatives like python multiprocessing or threading.
* Existing crates like [`gauss_quad`](https://crates.io/crates/gauss-quad) or [`integrate`](https://crates.io/crates/integrate) make it easy to get started, try several integration strategies, and evaluate the best one, and often themselves have an API that uses `rayon`.
* You can still easily expose bindings to python using [`pyo3`](https://docs.rs/pyo3/latest/pyo3/), and you can directly return `numpy` arrays for further processing using the rust [`numpy` crate](https://docs.rs/numpy/latest/numpy/).

I have been experimenting with workflows where, a large chunk of my model (e.g. a part large enough that it takes > 0.01s to compute) is actually written in rust. Then python code calls out to this, evaluates it with several parameters, plots results, and also manages fetching any data needed to drive it and storing the results. You can also easily play with the model and visualize results in a high-level repl after that point. The main drawback is, at the beginning you do end up going back and forth between rust and python often, but once it's stable I'm happier with the performance and the results. (Plus, if you then want to evaluate your model on a cluster of machines in the cloud, it's easy to write a pure-rust backend that does that, which IMO is more attractive than a python backend for many reasons.)

Even if you don't go as far as that, and continue to do your gaussian quadrature and integration in python instead, you can make use of lower-level bindings to crates like this one. Especially if you make use of the batch APIs, it may significantly improve the performance of your model compared to what you were doing before.

### Case study: Bivariate normal cdf

The `mv_norm::BatchBvnd` object enables evaluating the bivariate normal CDF with as much as ten times higher
throughput, through the use of precomputation and SIMD optimizations. This implementation was derived
from the `bvnd` function in [Alan Genz' `tvpack` fortran algorithm](https://github.com/cran/mvtnorm/blob/67d734c947eb10fbfa9d3431ba6a7d47241be58c/src/tvpack.f#L514), and is tested against it for fidelity.

#### Comparisons

At time of writing, I do not believe that [`statsrs`](https://crates.io/crates/statrs) or [`nalgebra-mvn`](https://crates.io/crates/nalgebra-mvn) contain implementations of the bivariate normal CDF,
and I could not find any other open source crates that do this.

I was not interested in [monte-carlo approaches](https://github.com/scipy/scipy/blob/6dbfa8c1463e33129cff2dabb01b67174a9bdf32/scipy/stats/_qmvnt.py#L146) to computing the CDF, or naive approaches based on computing the PDF.
In my motivating use-case, I wanted to use `bvnd` or similar in the integrand of an already complicated numerical integral,
and I wanted it to be very fast and accurate, so that the larger integral does not become prohibitive.

In my first attempt, I decided to use the [Owen's T function](https://en.wikipedia.org/wiki/Owen%27s_T_function) approach to computing the bivariate normal distribution.
This is the [`owens-t` crate](https://crates.io/crates/owens-t) published on crates.io -- in that case, we ported
C++ code from `boost::math`, and then implemented the bivariate normal CDF on top of the `owens_t` implementation.

The Owen's T method, using the [Patefield-Tandy algorithm](https://www.jstatsoft.org/article/view/v005i05), is competitive with the `tvpack` algorithm for computing the bivariate normal CDF.

* Patefield-Tandy is a hybrid approach that partitions the input domain for Owen's T function, and adaptively chooses one of 6 different series to evaluate based on all of the input parameters that are passed in.
* Genz' approach in `tvpack` selects one of two algorithms, and one of three quadratures, depending only on the value of `|rho|`. Performance is very sensitive to `|rho|`, and otherwise pretty consistent.
* For one-off, single point evaluations, both methods give comparable results, and which is better will depend mostly on what values of `rho` you have in your application.
* When we try to apply SIMD optimizations, however, the `tvpack` algorithm has an advantage.
   * It is difficult to use SIMD optimizations in the context of Patefield-Tandy because the most important routines are based on evaluating some number of terms of an alternating series, and the later terms depend on the values computed for the earlier terms. This computation is inherently sequential -- we can't easily compute 4 or 8 terms simultaneously. We could instead try to batch evaluate 4 or 8 different points of Owen's T simultaneously, but then we have the problem that they might fall in different regions and the Patefield-Tandy classifier might select different series for all of them.
   * By contrast, `tvpack` only contains two algorithms, and is always quadrature-based. [Quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature) is more amenable to SIMD since we can try to evaluate multiple points of the quadrature simultaneously, and this will even improve the performance of single-point evaluations.

The `tvpack` algorithm can also benefit significantly from precomputation if `rho` is known in advance and, e.g., we will evaluate `bvnd` at all `(x,y)` points in a grid. In these cases, several functions of `rho` can be computed only once and cached, and the univariate normal CDF can be computed for each `x` and `y` once, and then reused. Evaluating `(x,y)` pairs from a grid, across only a few values of `rho`, fits my motivating use-case exactly.

With these optimizations, throughput increases significantly, and 100 x 100 grids of normal CDF values can be evaluated rapidly on a single core. For smaller values of rho, we get an amortized 10-30 nanoseconds per evaluation in my testing, which is comparable to the cost of evaluating highly optimized functions like `erfc` from [`libm`](https://crates.io/crates/libm). Even for worst-case values of `rho`, my tests show that throughput approaches 20M evaluations per second.

(Try running the benchmarks on your hardware to see if you observe a similar speed up.)

TODO: Post comparisons with R and [scipy](https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/stats/_multivariate.py#L597) timings.

Compared to the `mvtnorm` R package and the existing `numpy` translations of the original fortran code, I believe that this implementation has some key advantages.

* Precomputing and re-using values that depend only on `rho` turns out to be very effective in the case of the `tvpack` algorithm.
* The hot loop on the slowest path performs a test to decide if it should bother evaluating a particular quadrature point, or skip it.
  * By sorting the quadrature points, we are able to do the most important ones first and bail out of the loop entirely once the test fails once, rather than continuing and applying the test to more points.
  * We can pre-compute exactly how many passes through the loop we will make, and then avoid additional branching in the loop at all. Branching within the loop can prevent the compiler from vectorizing the loop automatically, but this is what most implementations are doing.
* The hot loop evaluates `exp` on several arguments, and compilers appear to run away from this and not attempt to vectorize it. By using a SIMD helper crate ([`wide`](https://crates.io/crates/wide)) we can ensure that `exp` is computed in a vectorized fashion and that the whole hot loop works as intended.

## Future Directions

It would be interesting to:

* Port more of the `mvtnorm` sources, and use Rust's nice SIMD facilities to optimize them. (Eventually, the `core::simd` stuff when it is stabilized.)
* Expose python and/or R bindings and publish them.
* Provide an `f32` version of algorithms, if it can be significantly faster.
* Allow the user to choose larger error tolerances and get faster algorithms when possible.
  Whether this should be done by providing differently named functions (see [SLEEF](https://docs.rs/sleef/latest/sleef/f64x/index.html) for example) or a "policy object" like in `boost::math`, I'm not sure.

## Licensing and Distribution

MIT or Apache 2 at your option.

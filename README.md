mv-norm
=======

This rust crate contains functions related to multivariate normal distributions, especially,
computing the CDF. (Note: Right now we only have the bivariate normal CDF.)

It ports a subset of the [R package `mvtnorm`](https://cran.r-project.org/web/packages/mvtnorm/mvtnorm.pdf), which is
widely used for these purposes.

Additionally, for some functions, this crate provides an alternative "batch evaluation" API.

For example, the `BatchBvnd` context object enables evaluating the bivariate normal CDF with up to ten times higher
throughput, through the use of precomputation and SIMD optimizations. This implementation was derived
from the `bvnd` function in [Alan Genz' tvpack fortran algorithm](https://github.com/cran/mvtnorm/blob/67d734c947eb10fbfa9d3431ba6a7d47241be58c/src/tvpack.f#L514), and is tested against it for fidelity.

Comparisons
-----------

At time of writing, I do not believe that `stats-rs` or `nalgebra-mvn` contain implementations of the bivariate normal CDF,
and I could not find any other open source crates that do this.

I was not interested in monte-carlo approaches to computing the CDF, or naive approaches based on computing the PDF.
In my motivating use-case, I wanted to use `bvnd` or similar in the integrand of an already complicated numerical integral,
and I wanted it to be fast and accurate, so that the larger integral does not become prohibitive. (I was already committed
to using rust because other parts of my program are using [`rayon`](https://docs.rs/rayon/latest/rayon/), which is excellent.)

In my first attempt, I decided to use the Owen's T function approach to computing the bivariate normal distribution.
This is the [`owens-t` crate](https://crates.io/crates/owens-t) published on crates.io -- in that case, we ported
code from `boost::math`, and then implemented the bivariate normal CDF on top of the `owens_t` implementation.

The Owen's T method, using the Patefield-Tandy algorithm, is competitive with the `tvpack` algorithm method for computing the bivariate normal CDF.

* Patefield-Tandy is a hybrid approach that partitions the input domain for Owen's T function, and adaptively chooses one of 6 different series to evaluate based on all of the input parameters that are passed in.
* Genz' `tvpack` approach selects one of two algorithms, and one of three quadratures, depending only on the value of `|rho|`. Performance is very sensitive to `|rho|`, and otherwise pretty consistent.
* For one-off, single point evaluations, both methods are competitive, and which is better will depend mostly on what values of `rho` you have in your application.
* When we try to apply SIMD optimizations, however, the `tvpack` algorithm has an advantage.
   * It is difficult to use SIMD optimizations in the context of Patefield-Tandy because the most important routines are based on evaluating some number of terms of an alternating series, and the later terms depend on the values computed for the earlier terms. This computation is inherently sequential-- we can't easily compute 4 or 8 terms simultaneously. We could instead try to batch evaluate 4 or 8 different points of Owen's T simultaneously, but then we have the problem that they might fall in different regions and the Patefield-Tandy classifier might select different series for all of them.
   * By contrast, the `tvpack` approach only has two algorithms, and is always quadrature-based. [Quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature) is more amenable to SIMD since we can try to evaluate multiple points of the quadrature simultaneously, and this will even improve the performance of single-point evaluations.

The `tvpack` algorithm can also benefit significantly from precomputation if `rho` is known in advance and, e.g., we will evaluate `bvnd` at all `(x,y)` points in a grid. In these cases, several transcendental functions of `rho` can be computed only once and cached, and the univariate normal CDF can be computed for each `x` and `y` once, and then reused. Evaluating `(x,y)` pairs from a grid, across only a few values of `rho`, fits my motivating use-case exactly.

With these optimizations, throughput increases significantly, and 100 x 100 grids of normal CDF values can be evaluated rapidly on a single core. For smaller values of rho, we get an amortized 10-30 nanoseconds per evaluation in my testing, which is comparable to the cost of evaluating highly optimized functions like `erfc` from `libm`. Even for worst-case values of `rho`, my tests show that throughput approaches 20M evaluations per second.

(Try running the benchmarks on your hardware to see if you observe a similar speed up.)

TODO: Post comparisons with R and numpy timings.

Compared to the `mvtnorm` R package and the existing `numpy` translations of the original fortran code, I believe that this implementation has some key advantages.

* Precomputing and re-using values that depend only on `rho` turns out to be very effective in the case of the `tvpack` algorithm.
* The hot loop on the slowest path performs a test to decide if it should bother evaluating a particular quadrature point, or skip it.
  * By sorting the quadrature points, we are able to do the most important ones first and bail out of the loop entirely once the test fails once, rather than continuing and applying the test to more points.
  * We can pre-compute exactly how many passes through the loop we will make, and then avoid additional branching in the loop at all. Branching within the loop can prevent the compiler from vectorizing the loop automatically, but this is what most implementations are doing.
* The hot loop evaluates `exp` on several arguments, and compilers appear to run away from this and not attempt to vectorize it. By using a SIMD helper crate (`wide`) we can ensure that `exp` is computed in a vectorized fashion and that the whole hot loop works as intended.

Licensing and Distribution
--------------------------

MIT or Apache 2 at your option.

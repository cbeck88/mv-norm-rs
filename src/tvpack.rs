//! These functions are ported from Alan Genz' fortran code:
//!
//! See cran repository reae-only mirror:
//! https://github.com/cran/mvtnorm/blob/master/src/tvpack.f
#![allow(clippy::excessive_precision)]

use crate::util::*;
use libm::{asin, sin};

// These quadratures from the bvnd fortran sources
// https://github.com/cran/mvtnorm/blob/67d734c947eb10fbfa9d3431ba6a7d47241be58c/src/tvpack.f#L514
//
// NOTE: We added the last entry to quad6 as padding to simplify the simd version of this
//*     Gauss Legendre Points and Weights, N = 6
const QUAD_6: [(f64, f64); 4] = [
    (0.1713244923791705e+00, -0.9324695142031522e+00),
    (0.3607615730481384e+00, -0.6612093864662647e+00),
    (0.4679139345726904e+00, -0.2386191860831970e+00),
    (0.0, 0.0),
];

//*     Gauss Legendre Points and Weights, N = 12
const QUAD_12: [(f64, f64); 6] = [
    (0.4717533638651177e-01, -0.9815606342467191e+00),
    (0.1069393259953183e+00, -0.9041172563704750e+00),
    (0.1600783285433464e+00, -0.7699026741943050e+00),
    (0.2031674267230659e+00, -0.5873179542866171e+00),
    (0.2334925365383547e+00, -0.3678314989981802e+00),
    (0.2491470458134029e+00, -0.1252334085114692e+00),
];

//*     Gauss Legendre Points and Weights, N = 20
const QUAD_20: [(f64, f64); 10] = [
    (0.1761400713915212e-01, -0.9931285991850949e+00),
    (0.4060142980038694e-01, -0.9639719272779138e+00),
    (0.6267204833410906e-01, -0.9122344282513259e+00),
    (0.8327674157670475e-01, -0.8391169718222188e+00),
    (0.1019301198172404e+00, -0.7463319064601508e+00),
    (0.1181945319615184e+00, -0.6360536807265150e+00),
    (0.1316886384491766e+00, -0.5108670019508271e+00),
    (0.1420961093183821e+00, -0.3737060887154196e+00),
    (0.1491729864726037e+00, -0.2277858511416451e+00),
    (0.1527533871307259e+00, -0.7652652113349733e-01),
];

// quadrature selection from tvpack bvnd algorithm
fn select_quadrature(rho_abs: f64) -> &'static [(f64, f64)] {
    if rho_abs < 0.3 {
        &QUAD_6[..3]
    } else if rho_abs < 0.75 {
        &QUAD_12[..]
    } else {
        &QUAD_20[..]
    }
}

// quadrature selection, but we padded the result to be a multiple of two
// which simplified the SIMD version of things.
pub(crate) fn select_quadrature_padded(rho_abs: f64) -> &'static [(f64, f64)] {
    if rho_abs < 0.3 {
        &QUAD_6[..]
    } else if rho_abs < 0.75 {
        &QUAD_12[..]
    } else {
        &QUAD_20[..]
    }
}

/// Rust port of tvpack fortran function bvnd.
/// Note that this is basically a transliteration, and doesn't use SIMD or make
/// semantic changes to the original.
///
/// Note: I believe that the original has a bug when r <= -0.925, and is only one or
/// two decimals accurate in that case. But in other cases it is highly accurate.
/// The mv_norm::bvnd function has corrected the bug.
///
/// Orignal documentation:
///
/// ```ignore
///     A function for computing bivariate normal probabilities.
///
///       Alan Genz
///       Department of Mathematics
///       Washington State University
///       Pullman, WA 99164-3113
///       Email : alangenz@wsu.edu
///
///    This function is based on the method described by
///        Drezner, Z and G.O. Wesolowsky, (1989),
///        On the computation of the bivariate normal integral,
///        Journal of Statist. Comput. Simul. 35, pp. 101-107,
///    with major modifications for double precision, and for |R| close to 1.
///
/// BVND calculates the probability that X > DH and Y > DK.
///      Note: Prob( X < DH, Y < DK ) = BVND( -DH, -DK, R ).
///
/// Parameters
///
///   DH  DOUBLE PRECISION, integration limit
///   DK  DOUBLE PRECISION, integration limit
///   R   DOUBLE PRECISION, correlation coefficient
/// ```
/// https://github.com/cran/mvtnorm/blob/67d734c947eb10fbfa9d3431ba6a7d47241be58c/src/tvpack.f#L514
pub fn bvnd(dh: f64, dk: f64, r: f64) -> f64 {
    let mut h = dh;
    let mut k = dk;
    let hk = h * k;

    // Select quadrature
    let quad = select_quadrature(r.abs());

    let mut bvn = 0.0;

    if r.abs() <= 0.925 {
        if r.abs() > 0.0 {
            let hs = ((h * h) + (k * k)) / 2.0;
            let asr = 0.5 * asin(r);
            for (w, x) in quad {
                // We evaluate at 1-x and 1+x
                for is in [-1.0, 1.0] {
                    let sn = sin(asr * (is * x + 1.0));
                    bvn += w * exp((sn * hk - hs) / (1.0 - sn * sn));
                }
            }
            bvn *= asr * FRAC_1_2_PI;
        }
        bvn += phid(-h) * phid(-k);
        bvn
    } else {
        // r.abs() > 0.925
        if r < 0.0 {
            h = -h;
            k = -k;
        }
        if r.abs() < 1.0 {
            let a_s = (1.0 - r) * (1.0 + r);
            let mut a = sqrt(a_s);
            let b_s = (h - k) * (h - k);
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            let asr = -0.5 * (b_s / a_s + hk);
            if asr > -100.0 {
                bvn = a
                    * exp(asr)
                    * (1.0 - c * (b_s - a_s) * (1.0 - d * b_s / 5.0) / 3.0
                        + c * d * (a_s * a_s) / 5.0);
            }
            if -hk < 100.0 {
                let b = sqrt(b_s); // TODO: Can this not more simply be (h-k).abs()?
                bvn -= exp(-0.5 * hk)
                    * SQRT_2_PI
                    * phid(-b / a)
                    * b
                    * (1.0 - c * b_s * (1.0 - d * b_s / 5.0) / 3.0);
            }
            a /= 2.0;
            for (w, x) in quad {
                // We evaluate at 1-x and 1+x
                for is in [-1.0, 1.0] {
                    let x = a * (is * x + 1.0);
                    let x_s = x * x;
                    let r_s = sqrt(1.0 - x_s);
                    let asr = -0.5 * (b_s / x_s + hk);
                    if asr > -100.0 {
                        bvn += a
                            * w
                            * exp(asr)
                            * (exp(-hk * (1.0 - r_s) / (2.0 * (1.0 + r_s))) / r_s
                                - (1.0 + c * x_s * (1.0 + d * x_s)));
                    }
                }
            }
            bvn *= -FRAC_1_2_PI;
        }
        if r > 0.0 {
            bvn += phid(-f64::max(h, k));
        } else {
            bvn = -bvn;
            if k > h {
                bvn += phid(k) - phid(h)
            }
        }
        bvn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use assert_within::assert_within;

    // Test against the burkardt test points
    #[test]
    fn spot_check_tvpack_bvnd_burkardt() {
        // FIXME: Double check these test vectors, because we had similar precision
        // limits with the owens-t crate which makes me suspicious of them.
        for (n, BvndTestPoint { x, y, r, expected }) in get_burkardt_nbs_test_points().enumerate() {
            let eps = if x >= 0.0 && y >= 0.0 {
                1e-10
            } else if x == 0.0 || y == 0.0 {
                1e-6
            } else if x <= 0.0 && y <= 0.0 {
                1e-9
            } else {
                1e-6
            };

            let val = bvnd(x, y, r);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, bvnd(y,x,r), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // Test against the burkardt test vectors, but using owens t values.
    #[test]
    fn spot_check_tvpack_bvnd_against_burkardt_owens_t_vals() {
        let eps = 1e-15;
        for (n, BvndTestPoint { x, y, r, expected }) in
            get_owens_t_value_burkardt_test_points().enumerate()
        {
            let val = bvnd(x, y, r);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, bvnd(y,x, r), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // This tests against owens' T, at random points, only with r >= 0 points
    #[test]
    fn spot_check_tvpack_bvnd_random_owens() {
        for (n, BvndTestPoint { x, y, r, expected }) in get_random_owens_t_test_points().enumerate()
        {
            debug_assert!(r >= 0.0);
            if r == 1.0 || r == -1.0 {
                // I think these are buggy cases for tvpack
                continue;
            }
            let eps = 1e-15;
            let val = bvnd(x, y, r);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, bvnd(y, x, r), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }
}

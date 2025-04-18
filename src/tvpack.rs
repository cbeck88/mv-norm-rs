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
//*     Gauss Legendre Points and Weights, N = 6
pub(crate) const QUAD_6: [(f64, f64); 3] = [
    (0.1713244923791705e+00, -0.9324695142031522e+00),
    (0.3607615730481384e+00, -0.6612093864662647e+00),
    (0.4679139345726904e+00, -0.2386191860831970e+00),
];

//*     Gauss Legendre Points and Weights, N = 12
pub(crate) const QUAD_12: [(f64, f64); 6] = [
    (0.4717533638651177e-01, -0.9815606342467191e+00),
    (0.1069393259953183e+00, -0.9041172563704750e+00),
    (0.1600783285433464e+00, -0.7699026741943050e+00),
    (0.2031674267230659e+00, -0.5873179542866171e+00),
    (0.2334925365383547e+00, -0.3678314989981802e+00),
    (0.2491470458134029e+00, -0.1252334085114692e+00),
];

//*     Gauss Legendre Points and Weights, N = 20
pub(crate) const QUAD_20: [(f64, f64); 10] = [
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
    if dh == f64::INFINITY || dk == f64::INFINITY {
        return 0.0;
    }
    if dh == f64::NEG_INFINITY {
        if dk == f64::NEG_INFINITY {
            return 1.0;
        } else {
            return phid(dk);
        }
    }
    // Select quadrature
    let quad = if r.abs() < 0.3 {
        &QUAD_6[..]
    } else if r.abs() < 0.75 {
        &QUAD_12[..]
    } else {
        &QUAD_20[..]
    };

    let mut h = dh;
    let mut k = dk;
    let hk = h * k;
    let mut bvn = 0.0;

    if r.abs() < 0.925 {
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
        // r.abs() >= 0.925
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
    use assert_within::assert_within;

    // These values from: https://people.math.sc.edu/Burkardt/cpp_src/test_values/test_values.cpp
    // who says they come from Mathematica
    //
    //void bivariate_normal_cdf_values ( int &n_data, double &x, double &y,
    //  double &r, double &fxy )
    //
    //****************************************************************************80
    //
    //  Purpose:
    //
    //    BIVARIATE_NORMAL_CDF_VALUES returns some values of the bivariate normal CDF.
    //
    //  Discussion:
    //
    //    FXY is the probability that two variables A and B, which are
    //    related by a bivariate normal distribution with correlation R,
    //    respectively satisfy A <= X and B <= Y.
    //
    //    Mathematica can evaluate the bivariate normal CDF via the commands:
    //
    //      <<MultivariateStatistics`
    //      cdf = CDF[MultinormalDistribution[{0,0}{{1,r},{r,1}}],{x,y}]
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license.
    //
    //  Modified:
    //
    //    23 November 2010
    //
    //  Author:
    //
    //    John Burkardt
    //
    //  Reference:
    //
    //    National Bureau of Standards,
    //    Tables of the Bivariate Normal Distribution and Related Functions,
    //    NBS, Applied Mathematics Series, Number 50, 1959.
    //
    //  Parameters:
    //
    //    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
    //    first call.  On each call, the routine increments N_DATA by 1, and
    //    returns the corresponding data; when there is no more data, the
    //    output value of N_DATA will be 0 again.
    //
    //    Output, double &X, &Y, the parameters of the function.
    //
    //    Output, double &R, the correlation value.
    //
    //    Output, double &FXY, the value of the function.
    //

    const N_MAX: usize = 41;
    const FXY_VEC: [f64; N_MAX] = [
        0.02260327218569867E+00,
        0.1548729518584100E+00,
        0.4687428083352184E+00,
        0.7452035868929476E+00,
        0.8318608306874188E+00,
        0.8410314261134202E+00,
        0.1377019384919464E+00,
        0.1621749501739030E+00,
        0.1827411243233119E+00,
        0.2010067421506235E+00,
        0.2177751155265290E+00,
        0.2335088436446962E+00,
        0.2485057781834286E+00,
        0.2629747825154868E+00,
        0.2770729823404738E+00,
        0.2909261168683812E+00,
        0.3046406378726738E+00,
        0.3183113449213638E+00,
        0.3320262544108028E+00,
        0.3458686754647614E+00,
        0.3599150462310668E+00,
        0.3742210899871168E+00,
        0.3887706405282320E+00,
        0.4032765198361344E+00,
        0.4162100291953678E+00,
        0.6508271498838664E+00,
        0.8318608306874188E+00,
        0.0000000000000000,
        0.1666666666539970,
        0.2500000000000000,
        0.3333333333328906,
        0.5000000000000000,
        0.7452035868929476,
        0.1548729518584100,
        0.1548729518584100,
        0.06251409470431653,
        0.7452035868929476,
        0.1548729518584100,
        0.1548729518584100,
        0.06251409470431653,
        0.6337020457912916,
    ];
    const R_VEC: [f64; N_MAX] = [
        0.500, 0.500, 0.500, 0.500, 0.500, 0.500, -0.900, -0.800, -0.700, -0.600, -0.500, -0.400,
        -0.300, -0.200, -0.100, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800,
        0.900, 0.673, 0.500, -1.000, -0.500, 0.000, 0.500, 1.000, 0.500, 0.500, 0.500, 0.500,
        0.500, 0.500, 0.500, 0.500, 0.500,
    ];
    const X_VEC: [f64; N_MAX] = [
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        1.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        0.7071067811865475,
    ];
    const Y_VEC: [f64; N_MAX] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        0.7071067811865475,
    ];

    #[test]
    fn spot_check_phi2() {
        // FIXME: Double check these test vectors, because we had similar precision
        // limits with the owens-t crate which makes me suspicious of them.
        let eps = 1e-6;
        for n in 0..N_MAX {
            let x = X_VEC[n];
            let y = Y_VEC[n];
            let r = R_VEC[n];
            let fxy = FXY_VEC[n];
            let val = bvnd(x, y, r);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, bvnd(y,x,r), val);
            assert_within!(+eps, val, fxy, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }
}

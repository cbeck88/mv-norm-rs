use crate::tvpack::select_quadrature_padded;
use crate::util::*;
use libm::{asin, sin};
use wide::f64x4;

/// Context for quickly evaluating bvnd at many points with a single value of rho
#[derive(Clone, Debug)]
pub struct BatchBvnd {
    inner: BatchBvndInner,
}

impl BatchBvnd {
    /// Precompute for the evaluation of bivariate normal CDF at several points
    /// with this value of rho (correlation coefficient)
    pub fn new(rho: f64) -> Self {
        Self {
            inner: BatchBvndInner::new(rho),
        }
    }

    /// Evaluate Pr[ X > x, Y > y ] for X, Y standard normals of covariance rho
    pub fn bvnd(&self, x: f64, y: f64) -> f64 {
        self.inner.bvnd(x, y)
    }

    /// Same as bvnd, but faster if you already know values of phid(-x), phid(-y).
    /// Note that no checking of the values you provide is performed.
    ///
    /// Here phid(z) := 0.5 erfc(z/sqrt(2))
    ///
    /// If you know standard normal CDF of z or -z already then you probably have
    /// a good value for phid.
    pub fn bvnd_with_precomputed_phid(
        &self,
        x: f64,
        y: f64,
        phid_minus_x: f64,
        phid_minus_y: f64,
    ) -> f64 {
        self.inner
            .bvnd_with_precomputed_phid(x, y, phid_minus_x, phid_minus_y)
    }

    /// Batch evaluate bvnd at all points of a grid.
    /// This computes and reuses values of phid, which improves performance noticeably for large grids,
    /// and is relatively easy to use from the caller's perspective.
    ///
    /// This routine does not allocate, but uses an output parameter to store results.
    ///
    /// Input:
    ///   `xs`: A slice of f64 values
    ///   `ys`: A slice of f64 values
    ///   `out`: A mutable slice of f64 values.
    ///
    /// Pre-conditions:
    ///   `out` has length `(xs.len() + 1) * (ys.len() + 1)`, and will be interpreted as a `(Y+1) * (X+1)` matrix.
    ///   In the following, we use the notation `out[y_idx][x_idx] := out[y_idx * (xs.len() + 1) + x_idx ]`.
    ///
    ///   None of the values you pass in should be infinity or nan.
    ///
    /// Post-conditions:
    ///   The routine adds an "imaginary" value of -∞ to the beginning of your `xs` array and `ys` array.
    ///   Then, `out[y_idx][x_idx] = bvnd(xs[x_idx], ys[y_idx])`, with those imaginary entries.
    ///
    ///   When one of the arguments would be -∞, bivariate normal cdf degenerates to univariate normal cdf,
    ///   so you will get `phid` of the other argument.
    ///
    ///   `out[0][0]` will be `1.0` always.
    pub fn grid_bvnd(&self, xs: &[f64], ys: &[f64], out: &mut [f64]) {
        let xn = xs.len();
        let yn = ys.len();

        debug_assert!(
            out.len() == (xn + 1) * (yn + 1),
            "Invalid output buffer length: {} != {} = ({xn} + 1) * ({yn} + 1)",
            out.len(),
            (xn + 1) * (yn + 1)
        );

        let stride = xn + 1;

        out[0] = 1.0;
        for i in 0..xn {
            out[1 + i] = phid(-xs[i]);
        }
        for j in 0..yn {
            let phid_y = phid(-ys[j]);
            out[stride * j] = phid_y;

            for i in 0..xn {
                out[stride * j + i + 1] =
                    self.bvnd_with_precomputed_phid(xs[i], ys[j], out[1 + i], phid_y);
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum BatchBvndInner {
    // Rho = -1
    RhoMinus1,
    // Rho = 0
    Rho0,
    // Rho = 1
    Rho1,
    // Rho in (-0.925, .925), not zero
    RhoMiddle {
        // Precomputed quadrature values, dependent on value of rho
        // <= 20 points, in groups of 4
        quadrature: [Quad1; 5],
        // Number of chunks from the quadrature that are actually populated
        n: usize,
    },
    // Rho in (-1, -.925) or (.925, 1)
    RhoOther {
        // Value of rho provided by user
        rho: f64,
        // a_s = (1.0 - rho)*(1.0 + rho)
        a_s: f64,
        // a = sqrt(a_s)
        a: f64,
        // a_inv = 1.0 / a
        a_inv: f64,
        // Precomputed quadrature values, dependent on the value of rho
        // =20 points, in chunks of 4
        quadrature: [Quad2; 5],
    },
}

// Values associated to Rho middle quadrature
// Over-aligned to encourage vectorization
#[derive(Copy, Clone, Debug, Default)]
#[repr(align(32))]
struct Quad1 {
    // sn (sine value) from tvpack algorithm
    sn: f64x4,
    // (1 - sn^2)^{-1}, the reciprocal of denominator from tv_pack algo
    denom_inv: f64x4,
    // weight from tvpack quadrature, times asr / 2pi
    w: f64x4,
}

// Values associated to Rho other quadrature
#[derive(Copy, Clone, Debug, Default)]
#[repr(align(32))]
struct Quad2 {
    // x_s (x square) from tvpack algorithm
    // Note: tvpack mutates a before computing x, so we have to divide by two,
    // relative to the a that we record.
    x_s: f64x4,
    // x_s inverse
    x_s_inv: f64x4,
    // 1.0/r_s from tvpack algorithm
    r_s_inv: f64x4,
    // rational expression of r_s used in tvpack: (1.0 - r_s) / (2.0 * (1.0 + r_s))
    r_s_ratio: f64x4,
    // w * a/2 from tvpack algorithm
    w: f64x4,
}

impl BatchBvndInner {
    fn new(rho: f64) -> Self {
        assert!(rho.abs() <= 1.0, "rho must be between -1.0 and 1.0: {rho}");

        if rho == -1.0 {
            Self::RhoMinus1
        } else if rho == 0.0 {
            Self::Rho0
        } else if rho == 1.0 {
            Self::Rho1
        } else if rho.abs() <= 0.925 {
            let asr = asin(rho) * 0.5;

            let tv_pack_quad = select_quadrature_padded(rho.abs());
            debug_assert!(tv_pack_quad.len() % 2 == 0);
            let n = tv_pack_quad.len() / 2;

            let mut quadrature = [Quad1::default(); 5];
            for quad_idx in 0..n {
                let quad = &mut quadrature[quad_idx];
                for pair_idx in 0..2 {
                    let (w, x) = tv_pack_quad[2 * quad_idx + pair_idx];
                    for (sign_idx, is) in [-1.0, 1.0].iter().enumerate() {
                        let sn = sin(asr * (is * x + 1.0));
                        let denom_inv = (1.0 - sn * sn).recip();
                        let w = w * asr * FRAC_1_2_PI;

                        let idx = 2 * pair_idx + sign_idx;
                        quad.sn.as_array_mut()[idx] = sn;
                        quad.denom_inv.as_array_mut()[idx] = denom_inv;
                        quad.w.as_array_mut()[idx] = w;
                    }
                }
            }

            Self::RhoMiddle { quadrature, n }
        } else {
            let a_s = (1.0 + rho) * (1.0 - rho);
            let a = sqrt(a_s);
            let a_inv = a.recip();

            let tv_pack_quad = select_quadrature_padded(rho.abs());
            debug_assert!(tv_pack_quad.len() == 10);
            // Note: We want to generate the x's in monotonically
            // decreasing order because it simplifies loop exit criteria
            // So we don't use the tvpack ordering
            // for (w, x) in select_quadrature(rho.abs()) {
            //    for is in [-1.0, 1.0] {
            // The x's are negative and start close to -.99, and a is positive and close to 1.
            // So starting with is = -1.0 and iterating will start with the largest x and go to smallest.
            // When we then do is = 1.0, we go in reverse order.
            let temp: [(f64, f64); 20] = core::array::from_fn(|idx| {
                if idx < 10 {
                    let (w, x) = tv_pack_quad[idx];
                    (w, -x)
                } else {
                    tv_pack_quad[19 - idx]
                }
            });

            temp.windows(2).for_each(|w| {
                debug_assert!(
                    w[0].1 >= w[1].1,
                    "should be sorted so that x is decreasing: {w:?}"
                )
            });

            let quadrature: [Quad2; 5] = core::array::from_fn(|idx| {
                let mut quad = Quad2::default();
                for idx2 in 0..4 {
                    let (w, x) = temp[idx * 4 + idx2];
                    let a = a * 0.5; // See tvpack before quadrature starts
                    let x = a * (x + 1.0);
                    let x_s = x * x;
                    let r_s = sqrt(1.0 - x_s);
                    let w = w * a;

                    let x_s_inv = x_s.recip();
                    let r_s_ratio = (1.0 - r_s) / (2.0 * (1.0 + r_s));
                    let r_s_inv = r_s.recip();

                    quad.x_s.as_array_mut()[idx2] = x_s;
                    quad.x_s_inv.as_array_mut()[idx2] = x_s_inv;
                    quad.r_s_inv.as_array_mut()[idx2] = r_s_inv;
                    quad.r_s_ratio.as_array_mut()[idx2] = r_s_ratio;
                    quad.w.as_array_mut()[idx2] = w;
                }

                quad
            });

            Self::RhoOther {
                rho,
                a_s,
                a,
                a_inv,
                quadrature,
            }
        }
    }

    fn bvnd(&self, dh: f64, dk: f64) -> f64 {
        /* Note: I don't think we need to do this, and it won't really be possible in the batched api
        if dh == f64::INFINITY || dk == f64::INFINITY {
            return 0.0;
        }
        if dh == f64::NEG_INFINITY {
            if dk == f64::NEG_INFINITY {
                return 1.0;
            } else {
                return phid(dk);
            }
        } else if dk == f64::NEG_INFINITY {
            return phid(dh);
        }
        */
        self.bvnd_with_precomputed_phid(dh, dk, phid(-dh), phid(-dk))
    }

    // Comptue bvnd, using precomputed values of phid(-dh), phid(-dk)
    fn bvnd_with_precomputed_phid(
        &self,
        dh: f64,
        dk: f64,
        phid_minus_dh: f64,
        phid_minus_dk: f64,
    ) -> f64 {
        match self {
            Self::RhoMinus1 => {
                if dk > dh {
                    phid_minus_dk - phid_minus_dh
                } else {
                    0.0
                } // TODO: looks a bit funky, but matches tvpack... maybe there's a bug
            }
            Self::Rho0 => phid_minus_dh * phid_minus_dk,
            Self::Rho1 => {
                if dh > dk {
                    phid_minus_dh
                } else {
                    phid_minus_dk
                }
            }
            Self::RhoMiddle { quadrature, n } => {
                let h = dh;
                let k = dk;
                let hk = h * k;
                let hs = ((h * h) + (k * k)) * 0.5;

                let hk = f64x4::splat(hk);
                let hs = f64x4::splat(hs);

                let mut bvn = 0.0;
                for Quad1 { sn, denom_inv, w } in quadrature.iter().take(*n) {
                    bvn += (*w * ((*sn * hk - hs) * denom_inv).exp()).reduce_add();
                }
                // Note: bvn *= asr * FRAC_1_2_PI was folded into w
                bvn += phid_minus_dh * phid_minus_dk;
                bvn
            }
            Self::RhoOther {
                rho,
                a_s,
                a,
                a_inv,
                quadrature,
            } => {
                let hk = dh * dk;

                let mut bvn = 0.0;

                let b = dh - dk;
                let b_s = b * b;
                let c = (4.0 - hk) / 8.0;
                let d = (12.0 - hk) / 16.0;
                let asr = -0.5 * (b_s * (a_inv * a_inv) + hk);

                let common = c * (1.0 - d * b_s / 5.0) / 3.0;
                if asr > -100.0 {
                    bvn = a * exp(asr) * (1.0 - (b_s - a_s) * common + c * d * (a_s * a_s) / 5.0);
                }
                if -hk < 100.0 {
                    let b = b.abs();
                    bvn -= exp(-0.5 * hk) * SQRT_2_PI * phid(-b * a_inv) * b * (1.0 - b_s * common);
                }

                // This threshold comes from tvpack algorithm
                // Performance improves if we hoist it out of the loop and precompute the stopping point
                {
                    let limit = quadrature
                        .partition_point(|q2| b_s * q2.x_s_inv.as_array_ref()[0] <= (200.0 - hk));

                    // Hoisting these out manually in case the compiler cannot see that limit > 0
                    let minus_hk = f64x4::splat(-hk);
                    let b_s = f64x4::splat(b_s);
                    let c = f64x4::splat(c);
                    let d = f64x4::splat(d);
                    let one = f64x4::ONE;
                    let minus_half = f64x4::new([-0.5; 4]);

                    for Quad2 {
                        x_s,
                        x_s_inv,
                        r_s_inv,
                        r_s_ratio,
                        w,
                    } in quadrature.iter().take(limit)
                    {
                        // This all ideally gets vectorized, but the "exp" function seems to scare the compiler away,
                        // so we use one of the simd helper libraries that supports exp.
                        let asr = minus_half * (b_s * x_s_inv - minus_hk);
                        bvn += (*w
                            * asr.exp()
                            * ((minus_hk * r_s_ratio).exp() * r_s_inv
                                - (one + c * x_s * (one + d * x_s))))
                            .reduce_add();
                    }
                }
                bvn *= -FRAC_1_2_PI;
                // These h,k declarations are a bit silly but we're trying to preserve
                // it for now to make it easier to match it up to the fortran sources.
                if *rho > 0.0 {
                    let h = dh;
                    let k = dk;
                    if h > k {
                        bvn += phid_minus_dh;
                    } else {
                        bvn += phid_minus_dk;
                    }
                } else {
                    bvn = -bvn;
                    let h = -dh;
                    let k = -dk;
                    if k > h {
                        bvn += phid_minus_dk - phid_minus_dh
                    }
                }
                bvn
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        BvndTestPoint, get_additional_test_points, get_burkardt_nbs_test_points,
    };
    use assert_within::assert_within;

    #[test]
    fn spot_check_phi2() {
        // FIXME: Double check these test vectors, because we had similar precision
        // limits with the owens-t crate which makes me suspicious of them.
        let eps = 1e-6;
        for (n, BvndTestPoint { x, y, r, expected }) in get_burkardt_nbs_test_points().enumerate() {
            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    #[test]
    fn spot_check_phi2_additional() {
        let eps = 1e-6;
        for (n, BvndTestPoint { x, y, r, expected }) in get_additional_test_points().enumerate() {
            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }
}

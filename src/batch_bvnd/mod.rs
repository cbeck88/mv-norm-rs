use crate::tvpack::select_quadrature_padded;
use crate::util::*;
use libm::asin;
use wide::f64x4;

// Return phi(-x) = 1 - phi(x) = 0.5 erfc (x / sqrt(2))
// to double precision, and check for x = infinity.
#[inline(always)]
fn checked_phid_minus(x: f64) -> f64 {
    if x == f64::INFINITY {
        0.0
    } else {
        0.5 * libm::erfc(x * FRAC_1_SQRT_2)
    }
}

/// Evaluate Pr[ X > x, Y > y ] for X, Y standard normals of correlation coefficient `rho`.
///
/// `rho` must be in the range [-1.0, 1.0].
///
/// If `x` or `y` equals +∞, the result will be `0.0`.
/// If `x` or `y` equals -∞, the result is unspecified.
///
/// *Note*: This function uses just under 1kb of stack memory.
/// Typical programs have 4-8kb of stack, so this should usually be fine.
///
/// If that's too much for your application, you can alternatively:
///
/// * allocate a `BatchBvnd` object somewhere else and call `bvnd` on it
/// * use `tvpack::bvnd` instead, but it won't have SIMD optimizations then.
/// * use `owens_t::biv_norm` instead
#[inline]
pub fn bvnd(x: f64, y: f64, rho: f64) -> f64 {
    BatchBvnd::new(rho).bvnd(x, y)
}

/// Context for quickly evaluating bvnd at many points with a single value of rho
#[derive(Clone, Debug)]
pub struct BatchBvnd {
    inner: BatchBvndInner,
}

impl BatchBvnd {
    /// Precompute for the evaluation of bivariate normal CDF at several points
    /// with this value of rho (correlation coefficient)
    ///
    /// `rho` must be in the range [-1.0, 1.0].
    #[inline]
    pub fn new(rho: f64) -> Self {
        Self {
            inner: BatchBvndInner::new(rho),
        }
    }

    /// Evaluate Pr[ X > x, Y > y ] for X, Y standard normals of correlation coefficient `rho`
    ///
    /// If `x` or `y` = +∞, the result will be 0.0.
    /// If `x` or `y` = -∞, the result is unspecified.
    #[inline]
    pub fn bvnd(&self, x: f64, y: f64) -> f64 {
        self.inner.bvnd(x, y)
    }

    /// Same as bvnd, but faster if you already know values of phi(-x), phi(-y).
    /// Note that no checking of the values you provide is performed.
    ///
    /// Here phi(-z) := 0.5 erfc(z/sqrt(2))
    ///
    /// If you know the standard normal CDF value of z or -z already then you probably have
    /// a decent value for phid.
    pub fn bvnd_with_precomputed_phid(
        &self,
        x: f64,
        y: f64,
        phi_minus_x: f64,
        phi_minus_y: f64,
    ) -> f64 {
        self.inner
            .bvnd_with_precomputed_phid(x, y, phi_minus_x, phi_minus_y)
    }

    /// Batch evaluate bvnd at all points of a grid.
    /// This computes and reuses values of phi, which improves performance noticeably for large grids,
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
    ///   You may pass +∞ as an element of `xs` or `ys`, and that row or column will be entirely 0.0.
    ///   If you pass -∞, the behavior in that row or column is unspecified.
    ///
    /// Post-conditions:
    ///   The routine adds an "imaginary" value of -∞ to the beginning of your `xs` array and `ys` array.
    ///   Then, `out[y_idx][x_idx] = bvnd(xs[x_idx], ys[y_idx])`, when `xs` and `ys` are thought of *with those imaginary entries*.
    ///
    ///   Here, `bvnd(x,y) := Pr[ X > x, Y > y]` for X and Y standard normal of correlation coefficient `rho`.
    ///
    ///   In other words, to find the answer to the query corresponding to `(x_idx, y_idx)` in the input slices,
    ///   you have to add 1 to `x_idx` and to `y_idx` when you go to the output. The entries in row 0 or column 0 of
    ///   the output are special, and correspond to `x` or `y` being -∞.
    ///
    ///   When `x` or `y` is -∞, the bivariate normal cdf degenerates to the univariate normal cdf.
    ///   So  out[0][x_idx] = Pr[ X > xs[x_idx] ] = phi(-xs[x_idx])
    ///   and out[y_idx][0] = Pr[ Y > ys[y_idx] ] = phi(-ys[y_idx])
    ///
    ///   `out[0][0]` will always be `1.0`.
    ///
    ///   When `x` or `y` is ∞, the result will be `0.0`.
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
            out[1 + i] = checked_phid_minus(xs[i]);
        }
        for j in 0..yn {
            let phid_y = checked_phid_minus(ys[j]);
            out[stride * (j + 1)] = phid_y;

            for i in 0..xn {
                let phid_x = out[1 + i];
                out[stride * (j + 1) + (i + 1)] =
                    self.bvnd_with_precomputed_phid(xs[i], ys[j], phid_x, phid_y);
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
    // Note: for actual batch evaluation, this doesn't really need to be fast.
    // For the bvnd helper function, you would want this to be fast since it won't
    // really be amortized, but for now I'm not bothering.
    // Benchmarks show that BvndBatch::new(rho).bvnd(x,y) is still faster than
    // tvpack::bvnd(x,y,rho) anyways.
    #[inline]
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

            // This is a bit messy -- what's happening here is, we went from
            // a list of static (x, w) points for a quadrature, to a collection of
            // precomputed "data" depending on rho, needed to evaluate one point
            // of the quadrature. At that point, it's still an array of structs,
            // containing named f64 elements.
            //
            // When we incorporated simd, we now evaluate multiple quadrature points
            // simultaneously, so the Quad1 structure now contains f64x4 simd vectors.
            // So now it's more like an array of structs of arrays.
            //
            // Additionally, in `tvpack` algorithm every entry x in the quadrature array
            // is actually evaluated at 1-x and 1+x, with the same weight. So the `tvpack`
            // array is length 10, but implies 20 quadrature points.
            //
            // Depeding on the value of rho, we get at most a 20 point quadrature,
            // and tv_pack_quad has length 4, or 6, or 10.
            //
            // We iterate over tv_pack_quad in pairs, and then for each of those entries,
            // there is the 1-x and the 1+x version. That makes four quadrature points,
            // which are processed and stored in simd vectors for a single Quad1 object.
            // There are either 2, 3, or 5 Quad1 objects produced, depending on the value of n.
            let asr = f64x4::splat(asr);
            let quadrature: [Quad1; 5] = core::array::from_fn(|quad_idx| {
                if quad_idx >= n { return Quad1::default(); }
                // Expanded (with signs) weights and positions from tvpack quad, in simd register
                let (w,x) = {
                    let (w0, x0) = tv_pack_quad[2 * quad_idx];
                    let (w1, x1) = tv_pack_quad[2 * quad_idx + 1];
                    let w = f64x4::new([w0, w0, w1, w1]);
                    let x = f64x4::new([-x0, x0, -x1, x1]);
                    (w,x)
                };
                let sn = (asr * (x + f64x4::ONE)).sin();
                let denom_inv = f64x4::ONE / (f64x4::ONE - (sn * sn));
                let w = w * asr * f64x4::new([FRAC_1_2_PI; 4]);
                Quad1 {
                    sn,
                    denom_inv,
                    w,
                }
            });

            Self::RhoMiddle { quadrature, n }
        } else {
            let a_s = (1.0 + rho) * (1.0 - rho);
            let a = sqrt(a_s);
            let a_inv = a.recip();

            let tv_pack_quad = select_quadrature_padded(rho.abs());
            assert!(tv_pack_quad.len() == 10);

            // Expand tv_pack_quad to a full list of 20 points, in sorted order of x.
            //
            // We want to generate the x's in monotonically
            // decreasing order because it simplifies loop exit criteria
            // So we don't use the `tvpack` ordering:
            //
            // for (w, x) in select_quadrature(rho.abs()) {
            //    for is in [-1.0, 1.0] {
            //
            // The x's are negative and start close to -.99, and a is positive and close to 1.
            // So starting with is = -1.0 and iterating in order will start with the largest x.
            // Once those are exhausted, we do is = 1.0, and iterate in reverse order.
            let expanded_sorted_quad = |idx: usize| -> (f64, f64) {
                assert!(idx < 20);
                if idx < 10 {
                    let (w, x) = tv_pack_quad[idx];
                    (w, -x)
                } else {
                    tv_pack_quad[19 - idx]
                }
            };

            // Check that `expanded_sorted_quad` actually has decreasing values of x
            for i in 0..19 {
                let a = expanded_sorted_quad(i);
                let b = expanded_sorted_quad(i + 1);
                debug_assert!(
                    a.1 >= b.1,
                    "should be sorted so that x is decreasing: {a:?}, {b:?}"
                );
            }

            let quadrature: [Quad2; 5] = core::array::from_fn(|idx| {
                let mut quad = Quad2::default();
                for idx2 in 0..4 {
                    let (w, x) = expanded_sorted_quad(idx * 4 + idx2);
                    // Note: a_s = 1 - rho^2, and 0.925 <= |rho| < 1.
                    // So 0 <= a_s <= 0.144375
                    // and 0 <= a <= 0.37996
                    let a = a * 0.5; // See tvpack before quadrature starts
                    // Quadrature points are (-0.993, 0.993) so after this line, x in [0, ~0.37)
                    let x = a * (x + 1.0);
                    // 0 <= x_s <= ~0.142
                    let x_s = x * x;
                    let r_s = sqrt(1.0 - x_s);
                    let w = w * a;

                    // x_s_inv is at least 7.0, and can get arbitrarily large as rho approaches 1.
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
        self.bvnd_with_precomputed_phid(dh, dk, checked_phid_minus(dh), checked_phid_minus(dk))
    }

    // Compute bvnd, using precomputed values of phid(-dh), phid(-dk)
    #[inline(always)]
    fn bvnd_with_precomputed_phid(
        &self,
        dh: f64,
        dk: f64,
        phid_minus_dh: f64,
        phid_minus_dk: f64,
    ) -> f64 {
        if dh == f64::INFINITY || dk == f64::INFINITY {
            return 0.0;
        }
        if dh == f64::NEG_INFINITY {
            return phid_minus_dk;
        }
        if dk == f64::NEG_INFINITY {
            return phid_minus_dh;
        }

        match self {
            Self::RhoMinus1 => f64::max(phid_minus_dh + phid_minus_dk - 1.0, 0.0),
            Self::Rho0 => phid_minus_dh * phid_minus_dk,
            Self::Rho1 => {
                if dh < dk {
                    phid_minus_dk
                } else {
                    phid_minus_dh
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
                    bvn += (*w * ((sn.mul_sub(hk, hs) * denom_inv).exp())).reduce_add();
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
                // We use the following identity to reduce rho <= -0.925 inputs to rho <= 0.925:
                //
                // Pr[ X > x, Y > y] + Pr [ X > x, Y <= y] = Pr [ X > x ]
                // Pr[ X > x, Y > y] + Pr [ X > x, -Y >= -y] = Pr [ X > x ]
                // bvnd(x,y,r) + bvnd(x,-y,-r) = phid(-x)
                //
                // So instead we will flip the sign of r implicitly, and flip one of x or y.
                // Note that, the quadrature only depends on (1-r)(1+r), so flipping the sign doesn't
                // change any of that stuff.
                let (h, k) = if rho.is_sign_positive() {
                    (dh, dk)
                } else {
                    // Arbitrarily, we will flip the sign of h. It turns out it doesn't really matter which one we flip.
                    (-dh, dk)
                };
                let hk = h * k;

                let mut bvn = 0.0;

                let b = h - k;
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

                // This threshold comes from tvpack algorithm:
                //
                // ASR = -( BS/XS + HK )/2
                // IF ( ASR .GT. -100 ) THEN
                //   BVN = BVN + A*W(I,NG)*EXP( ASR )
                //          * ...
                //
                // If asr <= -100, then exp(asr) is very small and we can skip evaluating this point.
                //
                // In our version, we've sorted the points so that 1/x_s is increasing, and asr is decreasing.
                // So once we skip any point, we can skip all remaining points.
                //
                // Once we do simd, we don't evaluate it for every point, we only do it for every fourth point.
                // If we evaluate some extra quadrature points as a byproduct of simd, and their contribution
                // to the sum is negligible, no big deal.
                //
                // Performance improves if we hoist it out of the loop and precompute the stopping point.
                // `quadrature` only has 5 elements, so this `partition_point` is pretty fast.
                //
                // We have:
                // -1/2 * (bs/xs + hk) > -100
                // bs/xs + hk < 200
                // bs/xs < 200 - hk
                // 1/xs < (200 - hk) / bs
                //
                // The last equivalence is valid because b_s is a square, so dividing it doesn't change signs.
                // Even if it is zero, ieee requires that the result is +infinity or -infinity
                // However, it's probably not faster to do it that way, since fp division can be like 20-30 cycles,
                // while fp multiplication is typically 1-2 cycles.
                {
                    let limit = quadrature
                        .iter()
                        .position(|q2| b_s * q2.x_s_inv.as_array_ref()[0] >= (200.0 - hk))
                        .unwrap_or(5);
                    // let limit = quadrature
                    //    .partition_point(|q2| b_s * q2.x_s_inv.as_array_ref()[0] < (200.0 - hk));
                    /* Experiment:
                    // Actually only do the first 4 checks, and if all 4 are required, assume the last is as well,
                    // because the marginal cost to do the 5th is not that much, but going from 5 to 4 for binary search
                    // avoids an entire comparison.

                    let mut limit = quadrature[..4]
                        .partition_point(|q2| b_s * q2.x_s_inv.as_array_ref()[0] < (200.0 - hk));
                    limit += limit >> 2; // If the result was 4, then make it 5, otherwise leave it alone
                    */

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
                        let asr = minus_half * x_s_inv.mul_sub(b_s, minus_hk);
                        bvn += (*w
                            * asr.exp()
                            * (minus_hk * r_s_ratio).exp().mul_sub(*r_s_inv
                                , x_s.mul_add(d, one).mul_add(c * x_s, one)))
                            .reduce_add();
                    }
                }
                bvn *= -FRAC_1_2_PI;

                if rho.is_sign_positive() {
                    // bvn += phid(-f64::max(h, k));
                    if h > k {
                        bvn += phid_minus_dh;
                    } else {
                        bvn += phid_minus_dk;
                    }
                    bvn
                } else {
                    // Note: We have the following identity:
                    // bvnd(dh, dk, r) + bvnd(-dh, dk, -r) = phid(-dk)
                    //
                    // bvn variable currently holds what we would return for
                    // bvnd(h, k, -r), but missing the - phid(-max(h, k)) part.
                    //
                    // So bvn = bvnd(h,k,-r) - phid(-max(h,k))
                    // or bvnd(h,k,-r) = bvn + phid(-max(h,k))
                    // where h = -dh, k = dk
                    //
                    // We have
                    //
                    // bvnd(dh, dk, r) = phid(-dk) - bvnd(-dh, dk, -r)
                    //                 = phid(-dk) - phid(-max(h,k)) - bvn
                    //
                    // If dk >= -dh, then this is:
                    //
                    // bvnd(dh, dk, r) = phid(-dk) - phid(-dk) - bvn = -bvn
                    //
                    // Otherwise it is: phid(-dk) - phid(dh) - bvn
                    // = phid(-dk) + phid(-dh) - 1 - bvn
                    //
                    // So we now need to solve by:
                    // bvnd(dh, dk, r) = phid(-dk) + phid(-max(-dh, dk)) - bvn
                    //
                    // This is basically the same as the fortran code, if you take
                    // into account the mutability of the variables.
                    //
                    // BVN = -BVN
                    // IF ( K .GT. H ) BVN = BVN + PHID(K) - PHID(H)
                    //
                    // Testing gives strong evidence that both versions are correct,
                    // since it now satisfies all identifies to 15 decimals, and agrees
                    // with owen's t results everywhere to 15 decimals.
                    // But in this version, we can use precomputed values of phid,
                    // so its faster.
                    if k >= h {
                        -bvn
                    } else {
                        phid_minus_dk + phid_minus_dh - 1.0 - bvn
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use assert_within::assert_within;
    use rand::Rng;
    use rand_pcg::{Pcg64Mcg, rand_core::SeedableRng};

    // Check that the batch bvnd implementation matches the tvpack implementation closely
    #[test]
    fn batch_matches_tvpack() {
        let mut rng = Pcg64Mcg::seed_from_u64(9);

        let eps = 1e-15;

        for n in 0..10000 {
            let x = to_three_decimals(8.0 * rng.random::<f64>() - 4.0);
            let y = to_three_decimals(8.0 * rng.random::<f64>() - 4.0);
            //let r = to_three_decimals(rng.random::<f64>());
            let r = to_three_decimals(2.0 * rng.random::<f64>() - 1.0);
            let owens_t_val = owens_t::biv_norm(x, y, r);
            let ctxt = BatchBvnd::new(r);
            let batch_val = ctxt.bvnd(x, y);

            // While we're here, check that the owen's t value very pretty close
            // to the batch value.
            assert_within!(+eps, batch_val, owens_t_val, "n = {n}, x = {x}, y = {y}, rho = {r}");

            // Compare with tvpack value
            let tvpack_val = crate::tvpack::bvnd(x, y, r);

            assert_within!(+eps, batch_val, tvpack_val, "n = {n}, x = {x}, y = {y}, rho = {r}\nowens_t::biv_norm(x,y,rho) = {owens_t_val}")
        }
    }

    // Check against the burkardt test points
    #[test]
    fn spot_check_batch_bvnd_against_burkardt_points() {
        // Note: the burkardt points appear to themselves have fairly low accuracy
        let eps = 1e-6;
        for (n, BvndTestPoint { x, y, r, expected }) in get_burkardt_nbs_test_points().enumerate() {
            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // Check against the burkardt test points, but using the owens-t values
    #[test]
    fn spot_check_batch_bvnd_against_burkardt_owens_t() {
        let eps = 1e-15;
        for (n, BvndTestPoint { x, y, r, expected }) in
            get_owens_t_value_burkardt_test_points().enumerate()
        {
            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // Check against 10,000 random owen's T evaluations, where rho is positive
    #[test]
    fn spot_check_batch_bvnd_against_random_owens_t() {
        for (n, BvndTestPoint { x, y, r, expected }) in get_random_owens_t_test_points().enumerate()
        {
            let eps = 1e-15;

            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val, "n = {n}, x = {x}, y = {y}, rho = {r}");
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // Check against 10,000 random owen's T evaluations, where rho is negative
    #[test]
    fn spot_check_batch_bvnd_against_random_owens_t_negative_rho() {
        for (n, BvndTestPoint { x, y, r, expected }) in
            get_random_owens_t_test_points_negative_rho().enumerate()
        {
            let eps = 1e-15;

            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val, "n = {n}, x = {x}, y = {y}, rho = {r}");
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // Check against known points where x is 0
    #[test]
    fn spot_check_batch_bvnd_against_axis_points() {
        for (n, BvndTestPoint { x, y, r, expected }) in get_axis_test_points().enumerate() {
            let eps = 1e-15;

            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val, "n = {n}, x = {x}, y = {y}, rho = {r}");
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }

    // Check identities like:
    //
    // bvnd(x,y,r) = bvnd(y,x,r)
    // bvnd(x,y,r) + bvnd(x,-y,-r) = phi(-x)
    //
    // on random points
    #[test]
    fn check_symmetry_conditions() {
        let mut rng = Pcg64Mcg::seed_from_u64(9);

        for n in 0..10000 {
            let x = to_three_decimals(2.0 * rng.random::<f64>() - 1.0);
            let y = to_three_decimals(2.0 * rng.random::<f64>() - 1.0);
            let r = to_three_decimals(rng.random::<f64>());

            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            // Phi_2(x,y,r) = Phi_2(y,x,r);
            let eps = 1e-15;
            assert_within!(+eps, val, ctxt.bvnd(y,x), "n = {n}, x = {x}, y = {y}, rho = {r}");

            // Pr[ X > x, Y > y ] = Pr[X > x] - Pr[ X > x, Y < y ]
            // { Y < y } iff { -Y > -y }, and correlation of X and -Y  is -rho.
            // Note: this test fails when r >0.925, because the behavior at -0.925 is wonky
            //if r <= 0.925 {
            assert_within!(+eps, val, checked_phid_minus(x) - bvnd(x,-y,-r), "n = {n}, x = {x}, y = {y}, rho = {r}");
            //}
            assert_within!(+eps, val, checked_phid_minus(x) - checked_phid_minus(-y) + ctxt.bvnd(-x,-y), "n = {n}, x = {x}, y = {y}, rho = {r}");
        }
    }

    // Check identities like:
    //
    // bvnd(x,y,r) = bvnd(y,x,r)
    // bvnd(x,y,r) + bvnd(x,-y,-r) = phi(-x)
    //
    // on a wider range of random points
    #[test]
    fn check_symmetry_conditions_wider_range() {
        let mut rng = Pcg64Mcg::seed_from_u64(9);

        for n in 0..10000 {
            let x = to_three_decimals(8.0 * rng.random::<f64>() - 4.0);
            let y = to_three_decimals(8.0 * rng.random::<f64>() - 4.0);
            let r = to_three_decimals(rng.random::<f64>());

            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x, y);
            assert!(val >= 0.0, "val = {val}");
            assert!(val <= 1.0, "val = {val}");
            // Phi_2(x,y,r) = Phi_2(y,x,r);
            let eps = 1e-15;
            assert_within!(+eps, val, ctxt.bvnd(y,x), "n = {n}, x = {x}, y = {y}, rho = {r}");

            // Pr[ X > x, Y > y ] = Pr[X > x] - Pr[ X > x, Y < y ]
            // { Y < y } iff { -Y > -y }, and correlation of X and -Y  is -rho.
            assert_within!(+eps, val, checked_phid_minus(x) - bvnd(x,-y,-r), "n = {n}, x = {x}, y = {y}, rho = {r}");
            assert_within!(+eps, val, checked_phid_minus(x) - checked_phid_minus(-y) + ctxt.bvnd(-x,-y), "n = {n}, x = {x}, y = {y}, rho = {r}");
        }
    }

    // Check that grid evaluations give expected results
    #[test]
    fn check_grid_values() {
        let mut rng = Pcg64Mcg::seed_from_u64(9);

        let m = 20;

        let r = to_three_decimals(2.0 * rng.random::<f64>() - 1.0);

        let ctxt = BatchBvnd::new(r);

        let mut xs = vec![0.0; m];
        let mut ys = vec![0.0; m];
        let mut out = vec![0.0; (m + 1) * (m + 1)];

        let eps = 1e-15;
        for n in 0..1000 {
            let x1 = to_three_decimals(8.0 * rng.random::<f64>() - 4.0);
            let x2 = to_three_decimals(2.0 * rng.random::<f64>());
            let y1 = to_three_decimals(8.0 * rng.random::<f64>() - 4.0);
            let y2 = to_three_decimals(2.0 * rng.random::<f64>());

            for idx in 0..m {
                let f = (idx as f64) / (m as f64);
                xs[idx] = x1 + f * x2;
                ys[idx] = y1 + f * y2;
            }

            ctxt.grid_bvnd(&xs, &ys, &mut out);

            for i in 0..m {
                for j in 0..m {
                    assert_within!(+eps, out[(j+1)*(m + 1) + (i + 1)], crate::tvpack::bvnd(xs[i],ys[j],r), "n = {n}, i = {i}, j = {j}, xs[i] = {}, ys[j] = {}", xs[i], ys[j]);
                }
            }
        }
    }
}

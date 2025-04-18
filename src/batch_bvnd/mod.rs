use crate::tvpack::select_quadrature;
use crate::util::*;
use heapless::Vec;
use libm::{asin, sin};

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
}

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
        quadrature: Vec<Quad1, 20>,
    },
    // Rho in (-1, -.925) or (.925, 1)
    RhoOther {
        // Value of rho provided by user
        rho: f64,
        // a_s = (1.0 - rho)*(1.0 + rho)
        a_s: f64,
        // a_s_inv = 1.0 / a_s
        a_s_inv: f64,
        // a = sqrt(a_s)
        a: f64,
        // Precomputed quadrature values, dependent on the value of rho
        quadrature: Vec<Quad2, 20>,
    },
}

// Values associated to Rho middle quadrature
#[derive(Copy, Clone, Debug)]
struct Quad1 {
    // sn (sine value) from tvpack algorithm
    sn: f64,
    // weight from tvpack quadrature, times asr / 2pi
    w: f64,
}

// Values associated to Rho other quadrature
#[derive(Copy, Clone, Debug)]
struct Quad2 {
    // x_s (x square) from tvpack algorithm
    // Note: tvpack mutates a before computing x, so we have to divide by two,
    // relative to the a that we record.
    x_s: f64,
    // x_s inverse
    x_s_inv: f64,
    // 1.0/r_s from tvpack algorithm
    r_s_inv: f64,
    // rational expression of r_s used in tvpack: (1.0 - r_s) / (2.0 * (1.0 + r_s))
    r_s_ratio: f64,
    // w * a/2 from tvpack algorithm
    w: f64,
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

            let mut quadrature = Vec::<Quad1, 20>::new();
            for (w, x) in select_quadrature(rho.abs()) {
                for is in [-1.0, 1.0] {
                    quadrature.push(Quad1 {
                        sn: sin(asr * (is * x + 1.0)),
                        w: w * asr * FRAC_1_2_PI,
                    }).unwrap();
                }
            }

            Self::RhoMiddle { quadrature }
        } else {
            let a_s = (1.0 + rho) * (1.0 - rho);
            let a = sqrt(a_s);
            let a_s_inv = a_s.recip();

            let mut quadrature = Vec::<Quad2, 20>::new();
            for (w, x) in select_quadrature(rho.abs()) {
                for is in [-1.0, 1.0] {
                    let a = a * 0.5; // See tvpack before quadrature starts
                    let x = a * (is * x + 1.0);
                    let x_s = x * x;
                    let r_s = sqrt(1.0 - x_s);
                    let w = w * a;

                    let x_s_inv = x_s.recip();
                    let r_s_ratio = (1.0 - r_s) / (2.0 * (1.0 + r_s));
                    let r_s_inv = r_s.recip();

                    quadrature.push(Quad2 { x_s, x_s_inv, r_s_inv, r_s_ratio, w }).unwrap();
                }
            }

            Self::RhoOther {
                rho,
                a_s,
                a_s_inv,
                a,
                quadrature,
            }
        }
    }

    fn bvnd(&self, dh: f64, dk: f64) -> f64 {
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

        match self {
            Self::RhoMinus1 => {
                let h = -dh;
                let k = -dk;
                let mut bvn = 0.0;
                if k > h {
                    bvn += phid(k) - phid(h)
                } // TODO: looks a bit funky, but matches tvpack... maybe there's a bug
                bvn
            }
            Self::Rho0 => {
                let h = dh;
                let k = dk;
                phid(-h) * phid(-k)
            }
            Self::Rho1 => {
                let h = dh;
                let k = dk;
                phid(-f64::max(h, k))
            }
            Self::RhoMiddle { quadrature } => {
                let h = dh;
                let k = dk;
                let hk = h * k;
                let hs = ((h * h) + (k * k)) * 0.5;

                let mut bvn = 0.0;
                for Quad1 { sn, w } in quadrature.iter() {
                    bvn += w * exp((sn * hk - hs) / (1.0 - sn * sn));
                }
                // Note: bvn *= asr * FRAC_1_2_PI was folded into w
                bvn += phid(-h) * phid(-k);
                bvn
            }
            Self::RhoOther {
                rho,
                a_s,
                a_s_inv,
                a,
                quadrature,
            } => {
                let hk = dh * dk;

                let mut bvn = 0.0;

                let b = dh - dk;
                let b_s = b * b;
                let c = (4.0 - hk) / 8.0;
                let d = (12.0 - hk) / 16.0;
                let asr = -0.5 * (b_s * a_s_inv + hk);
                if asr > -100.0 {
                    bvn = a
                        * exp(asr)
                        * (1.0 - c * (b_s - a_s) * (1.0 - d * b_s / 5.0) / 3.0
                            + c * d * (a_s * a_s) / 5.0);
                }
                if -hk < 100.0 {
                    let b = b.abs();
                    bvn -= exp(-0.5 * hk)
                        * SQRT_2_PI
                        * phid(-b / a)
                        * b
                        * (1.0 - c * b_s * (1.0 - d * b_s / 5.0) / 3.0);
                }

                for Quad2 { x_s, x_s_inv, r_s_inv, r_s_ratio, w } in quadrature.iter() {
                    let asr = -0.5 * (b_s * x_s_inv + hk);
                    if asr > -100.0 {
                        bvn += w // note: a* was folded into w
                            * exp(asr)
                            * (exp(-hk * r_s_ratio) * r_s_inv
                                - (1.0 + c * x_s * (1.0 + d * x_s)));
                    }
                }
                bvn *= -FRAC_1_2_PI;
                // These h,k declarations are a bit silly but we're trying to preserve
                // it for now to make it easier to match it up to the fortran sources.
                if *rho > 0.0 {
                    let h = dh;
                    let k = dk;
                    bvn += phid(-f64::max(h, k));
                } else {
                    bvn = -bvn;
                    let h = -dh;
                    let k = -dk;
                    if k > h {
                        bvn += phid(k) - phid(h)
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
    use assert_within::assert_within;
    use crate::test_utils::{get_burkardt_nbs_test_points, BvndTestPoint};

    #[test]
    fn spot_check_phi2() {
        // FIXME: Double check these test vectors, because we had similar precision
        // limits with the owens-t crate which makes me suspicious of them.
        let eps = 1e-6;
        for (n, BvndTestPoint { x, y, r, expected }) in get_burkardt_nbs_test_points().enumerate() {
            let ctxt = BatchBvnd::new(r);
            let val = ctxt.bvnd(x,y);
            //eprintln!("n = {n}: biv_norm({x}, {y}, {r}) = {val}: expected: {fxy}");
            assert_within!(+eps, ctxt.bvnd(y,x), val);
            assert_within!(+eps, val, expected, "n = {n}, x = {x}, y = {y}, rho = {r}")
        }
    }
}

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

pub fn to_three_decimals(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Test point for bivariate normal distribution CDF values.
pub struct BvndTestPoint {
    pub x: f64,
    pub y: f64,
    pub r: f64,
    pub expected: f64,
}

impl From<&(f64, f64, f64, f64)> for BvndTestPoint {
    fn from(src: &(f64, f64, f64, f64)) -> Self {
        let (x, y, r, expected) = *src;
        Self { x, y, r, expected }
    }
}

pub fn get_owens_t_value_burkardt_test_points() -> impl Iterator<Item = BvndTestPoint> {
    (0..N_MAX).map(|i| {
        let x = -X_VEC[i];
        let y = -Y_VEC[i];
        let r = R_VEC[i];
        let expected = owens_t::biv_norm(x, y, r);
        BvndTestPoint { x, y, r, expected }
    })
}

// Ensure test coverage on the part of the algorithm where rho > 0.925
pub fn get_random_owens_t_test_points() -> impl Iterator<Item = BvndTestPoint> {
    let mut rng = Pcg64Mcg::seed_from_u64(9);

    (0..10000)
        .map(move |_| {
            let x = to_three_decimals(4.0 * rng.random::<f64>() - 2.0);
            let y = to_three_decimals(4.0 * rng.random::<f64>() - 2.0);
            let r = to_three_decimals(rng.random::<f64>());
            let expected = owens_t::biv_norm(x, y, r);
            BvndTestPoint { x, y, r, expected }
        })
        .filter(|tp| tp.x != 0.0 && tp.y != 0.0)
}

pub fn get_burkardt_nbs_test_points() -> impl Iterator<Item = BvndTestPoint> {
    // NOTE: Burkardt's vectors are for Pr[ X < x, Y < y], while we are trying to solve
    // Pr [ X > x, Y > y]
    (0..N_MAX).map(|i| BvndTestPoint {
        x: -X_VEC[i],
        y: -Y_VEC[i],
        r: R_VEC[i],
        expected: FXY_VEC[i],
    })
}

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
    -0.300, -0.200, -0.100, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
    0.673, 0.500, -1.000, -0.500, 0.000, 0.500, 1.000, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500,
    0.500, 0.500, 0.500,
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

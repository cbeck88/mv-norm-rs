use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mvnorm::bvnd;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

fn div_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("f64::div", |b| {
        b.iter(|| {
            let arg: f64 = black_box(rng.random::<f64>());
            let arg2: f64 = black_box(rng.random::<f64>());
            arg / arg2
        })
    });
}

fn sqrt_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("f64::sqrt", |b| {
        b.iter(|| {
            let arg: f64 = black_box(4.0 * rng.random::<f64>() - 2.0);
            arg.sqrt()
        })
    });
}

fn exp_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("f64::exp", |b| {
        b.iter(|| {
            let arg: f64 = black_box(4.0 * rng.random::<f64>() - 2.0);
            arg.exp()
        })
    });
}


fn erfc_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("libm::erfc", |b| {
        b.iter(|| {
            let arg: f64 = black_box(4.0 * rng.random::<f64>() - 2.0);
            libm::erfc(arg)
        })
    });
}

fn sin_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("libm::sin", |b| {
        b.iter(|| {
            let arg: f64 = black_box(4.0 * rng.random::<f64>() - 2.0);
            libm::sin(arg)
        })
    });
}

fn asin_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("libm::asin", |b| {
        b.iter(|| {
            let arg: f64 = black_box(4.0 * rng.random::<f64>() - 2.0);
            libm::asin(arg)
        })
    });
}

fn bvnd_bench(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    c.bench_function("bvnd_bench", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(rng.random::<f64>());
            let arg2: f64 = black_box(rng.random::<f64>());
            let arg3: f64 = black_box(rng.random::<f64>());
            bvnd(arg1, arg2, arg3)
        })
    });
}

criterion_group!(
    builtins,
    div_bench,
    sqrt_bench,
    exp_bench
);

criterion_group!(
    libm,
    erfc_bench,
    sin_bench,
    asin_bench,
);

criterion_group!(
    ours,
    bvnd_bench
);

criterion_main!(/*builtins,*/ libm, ours);

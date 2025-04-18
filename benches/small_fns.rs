use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mvnorm::{bvnd, BatchBvnd};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

// Circular buffer of random value to avoid invoking rng during benchmark
struct ValuePool {
    arr: [f64; 512],
    idx: usize,
}

impl ValuePool {
    fn new(mut f: impl FnMut(&mut Pcg64Mcg) -> f64) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(9);
        Self {
            arr: core::array::from_fn(move |_| f(&mut rng)),
            idx: 0,
        }
    }
    fn next(&mut self) -> f64 {
        self.idx += 1;
        self.idx %= 512;
        self.arr[self.idx]
    }
}

fn div_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| rng.random::<f64>());
    c.bench_function("f64::div", |b| {
        b.iter(|| {
            let arg: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            arg / arg2
        })
    });
}

fn sqrt_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| rng.random::<f64>());
    c.bench_function("f64::sqrt", |b| {
        b.iter(|| {
            let arg: f64 = black_box(vp.next());
            arg.sqrt()
        })
    });
}

fn exp_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 4.0 * rng.random::<f64>() - 2.0);
    c.bench_function("f64::exp", |b| {
        b.iter(|| {
            let arg: f64 = black_box(vp.next());
            arg.exp()
        })
    });
}

fn erfc_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 4.0 * rng.random::<f64>() - 2.0);
    c.bench_function("libm::erfc", |b| {
        b.iter(|| {
            let arg: f64 = black_box(vp.next());
            libm::erfc(arg)
        })
    });
}

fn sin_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 4.0 * rng.random::<f64>() - 2.0);
    c.bench_function("libm::sin", |b| {
        b.iter(|| {
            let arg: f64 = black_box(vp.next());
            libm::sin(arg)
        })
    });
}

fn asin_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| rng.random::<f64>());
    c.bench_function("libm::asin", |b| {
        b.iter(|| {
            let arg: f64 = black_box(vp.next());
            libm::asin(arg)
        })
    });
}

fn bvnd_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    c.bench_function("bvnd", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next());
            bvnd(arg1, arg2, arg3)
        })
    });
}

fn bvnd_pos_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| rng.random::<f64>());
    c.bench_function("bvnd (rho > 0)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next());
            bvnd(arg1, arg2, arg3)
        })
    });
}


fn bvnd_batch_025(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    let ctxt = BatchBvnd::new(0.25);
    c.bench_function("BatchBvnd::new(0.25).eval(...)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            ctxt.bvnd(arg1, arg2)
        })
    });
}

fn bvnd_batch_05(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    let ctxt = BatchBvnd::new(0.5);
    c.bench_function("BatchBvnd::new(0.5).eval(...)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            ctxt.bvnd(arg1, arg2)
        })
    });
}

fn bvnd_batch_075(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    let ctxt = BatchBvnd::new(0.75);
    c.bench_function("BatchBvnd::new(0.75).eval(...)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            ctxt.bvnd(arg1, arg2)
        })
    });
}

fn bvnd_batch_099(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    let ctxt = BatchBvnd::new(0.99);
    c.bench_function("BatchBvnd::new(0.99).eval(...)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            ctxt.bvnd(arg1, arg2)
        })
    });
}

criterion_group!(builtins, div_bench, sqrt_bench, exp_bench);

criterion_group!(libm, erfc_bench, sin_bench, asin_bench,);

criterion_group!(ours, bvnd_bench, bvnd_pos_bench, bvnd_batch_025, bvnd_batch_05, bvnd_batch_075, bvnd_batch_099);

criterion_main!(/*builtins, libm,*/ ours);

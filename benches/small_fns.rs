use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use mv_norm::{BatchBvnd, bvnd, tvpack};
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
    let mut vp = ValuePool::new(|rng| 4.0 * rng.random::<f64>() - 2.0);
    c.bench_function("bvnd ([-2,2])", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next() / 2.0);
            bvnd(arg1, arg2, arg3)
        })
    });
}

fn bvnd_bench_narrow(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    c.bench_function("bvnd ([-1,1])", |b| {
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
    c.bench_function("bvnd ([0,1]) (rho > 0)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next());
            bvnd(arg1, arg2, arg3)
        })
    });
}

fn tvpack_bvnd_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 4.0 * rng.random::<f64>() - 2.0);
    c.bench_function("tvpack::bvnd ([-2,2])", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next() / 2.0);
            tvpack::bvnd(arg1, arg2, arg3)
        })
    });
}

fn tvpack_bvnd_bench_narrow(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);
    c.bench_function("tvpack::bvnd ([-1,1])", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next());
            tvpack::bvnd(arg1, arg2, arg3)
        })
    });
}

fn tvpack_bvnd_pos_bench(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| rng.random::<f64>());
    c.bench_function("tvpack::bvnd ([0,1]) (rho > 0)", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            let arg3: f64 = black_box(vp.next());
            tvpack::bvnd(arg1, arg2, arg3)
        })
    });
}

fn bvnd_batch(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 2.0 * rng.random::<f64>() - 1.0);

    let mut group = c.benchmark_group(format!("BatchBvnd::bvnd([-1,1])"));
    for rho in [0.25, 0.50, 0.75, 0.99] {
        let ctxt = BatchBvnd::new(rho);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("rho", rho), &rho, |b, _| {
            b.iter(|| {
                let arg1: f64 = black_box(vp.next());
                let arg2: f64 = black_box(vp.next());
                ctxt.bvnd(arg1, arg2)
            })
        });
    }
}

fn bvnd_batch_wider(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 4.0 * rng.random::<f64>() - 2.0);

    let mut group = c.benchmark_group(format!("BatchBvnd::bvnd([-2,2])"));
    for rho in [0.25, 0.50, 0.75, 0.99] {
        let ctxt = BatchBvnd::new(rho);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("rho", rho), &rho, |b, _| {
            b.iter(|| {
                let arg1: f64 = black_box(vp.next());
                let arg2: f64 = black_box(vp.next());
                ctxt.bvnd(arg1, arg2)
            })
        });
    }
}

fn bvnd_batch_099_2(c: &mut Criterion) {
    let mut vp = ValuePool::new(|rng| 8.0 * rng.random::<f64>() - 4.0);
    let ctxt = BatchBvnd::new(0.99);
    c.bench_function("BatchBvnd::new(0.99).bvnd([-4,4])", |b| {
        b.iter(|| {
            let arg1: f64 = black_box(vp.next());
            let arg2: f64 = black_box(vp.next());
            ctxt.bvnd(arg1, arg2)
        })
    });
}

fn bvnd_grid<const N: usize>(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(9);
    let xs: [f64; N] = core::array::from_fn(|_| 4.0 * rng.random::<f64>() - 2.0);
    let ys: [f64; N] = core::array::from_fn(|_| 4.0 * rng.random::<f64>() - 2.0);
    let mut out = vec![0f64; (N + 1) * (N + 1)];

    let mut group = c.benchmark_group(format!("BatchBvnd::grid_bvnd({N} x {N}, [-2,2])"));
    for rho in [0.25, 0.50, 0.75, 0.99] {
        let ctxt = BatchBvnd::new(rho);
        group.throughput(Throughput::Elements((N * N) as u64));
        group.bench_with_input(BenchmarkId::new("rho", rho), &rho, |b, _| {
            b.iter(|| {
                let xs: &[f64] = black_box(&xs[..]);
                let ys: &[f64] = black_box(&ys[..]);
                ctxt.grid_bvnd(xs, ys, &mut out);
                out[0]
            })
        });
    }
}

criterion_group!(builtins, div_bench, sqrt_bench, exp_bench);

criterion_group!(libm, erfc_bench, sin_bench, asin_bench,);

criterion_group!(
    bvnd_single,
    bvnd_bench,
    bvnd_bench_narrow,
    bvnd_pos_bench,
    tvpack_bvnd_bench,
    tvpack_bvnd_bench_narrow,
    tvpack_bvnd_pos_bench,
);

criterion_group!(our_batch, bvnd_batch, bvnd_batch_wider, bvnd_batch_099_2);

criterion_group!(our_grid_eval, bvnd_grid<50>, bvnd_grid<100>,);

criterion_main!(
    //builtins,
    //libm,
    //bvnd_single,
    //our_batch,
    our_grid_eval
);

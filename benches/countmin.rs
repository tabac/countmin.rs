use std::collections::hash_map::RandomState;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use countminsketch::CountMin;

fn generate_strings(count: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();

    let mut workload: Vec<String> = (0..count)
        .map(|_| format!("- {} - {} -", rng.gen::<u64>(), rng.gen::<u64>()))
        .collect();

    workload.shuffle(&mut rng);

    workload
}

fn bench_update(c: &mut Criterion) {
    let workload = generate_strings(2000);

    macro_rules! bench_impls {
        ($benchname:expr, $width:expr, $depth:expr) => {
            let mut cms: CountMin<String, u32, RandomState> =
                CountMin::with_dimensions($width, $depth, RandomState::new())
                    .unwrap();

            c.bench_function($benchname, |b| {
                b.iter(|| {
                    for val in &workload {
                        cms.update(&val, 1);
                    }
                })
            });
        };
    }

    bench_impls!["countmin_update_w200d4", 200, 4];
    bench_impls!["countmin_update_w2000d4", 2000, 4];
    bench_impls!["countmin_update_w20000d4", 20000, 4];
}

fn bench_query(c: &mut Criterion) {
    let workload = generate_strings(2000);

    macro_rules! bench_impls {
        ($benchname:expr, $width:expr, $depth:expr) => {
            let mut cms: CountMin<String, u32, RandomState> =
                CountMin::with_dimensions($width, $depth, RandomState::new())
                    .unwrap();

            for val in &workload {
                cms.update(&val, 1);
            }

            c.bench_function($benchname, |b| {
                b.iter(|| {
                    for val in &workload {
                        let val = cms.query(val);
                        black_box(val);
                    }
                })
            });
        };
    }

    bench_impls!["countmin_query_w200d4", 200, 4];
    bench_impls!["countmin_query_w2000d4", 2000, 4];
    bench_impls!["countmin_query_w20000d4", 20000, 4];
}

criterion_group!(benches, bench_update, bench_query);

criterion_main!(benches);

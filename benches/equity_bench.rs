use criterion::{criterion_group, criterion_main, Criterion};
use rust_poker::equity_calculator::{exact_equity, exact_equity_seq};
use rust_poker::hand_range::{get_card_mask, HandRange};

// ---------------------------------------------------------------------------
// Heads-up preflop (lookup path: postflop_combos >> 500)
// ---------------------------------------------------------------------------

fn bench_hu_preflop_threaded(c: &mut Criterion) {
    let ranges =
        HandRange::from_strings(["AA".to_string(), "random".to_string()].to_vec());
    let board = get_card_mask("");
    c.bench_function("exact_equity(AA vs random, preflop, 1-thread)", |b| {
        b.iter(|| exact_equity(&ranges, board, 1).unwrap())
    });
}

fn bench_hu_preflop_seq(c: &mut Criterion) {
    let ranges =
        HandRange::from_strings(["AA".to_string(), "random".to_string()].to_vec());
    let board = get_card_mask("");
    c.bench_function("exact_equity_seq(AA vs random, preflop)", |b| {
        b.iter(|| exact_equity_seq(&ranges, board).unwrap())
    });
}

// ---------------------------------------------------------------------------
// Heads-up weighted range (lookup path)
// ---------------------------------------------------------------------------

fn bench_hu_weighted_threaded(c: &mut Criterion) {
    let ranges =
        HandRange::from_strings(["KK".to_string(), "AA@1,QQ".to_string()].to_vec());
    let board = get_card_mask("");
    c.bench_function("exact_equity(KK vs AA@1+QQ, preflop, 1-thread)", |b| {
        b.iter(|| exact_equity(&ranges, board, 1).unwrap())
    });
}

fn bench_hu_weighted_seq(c: &mut Criterion) {
    let ranges =
        HandRange::from_strings(["KK".to_string(), "AA@1,QQ".to_string()].to_vec());
    let board = get_card_mask("");
    c.bench_function("exact_equity_seq(KK vs AA@1+QQ, preflop)", |b| {
        b.iter(|| exact_equity_seq(&ranges, board).unwrap())
    });
}

// ---------------------------------------------------------------------------
// 4-way preflop (two combined-range groups, lookup path)
// ---------------------------------------------------------------------------

fn bench_4way_preflop_threaded(c: &mut Criterion) {
    let ranges = HandRange::from_strings(
        [
            "AKo".to_string(),
            "QJo".to_string(),
            "T9o".to_string(),
            "87o".to_string(),
        ]
        .to_vec(),
    );
    let board = get_card_mask("");
    c.bench_function("exact_equity(AKo/QJo/T9o/87o, preflop, 1-thread)", |b| {
        b.iter(|| exact_equity(&ranges, board, 1).unwrap())
    });
}

fn bench_4way_preflop_seq(c: &mut Criterion) {
    let ranges = HandRange::from_strings(
        [
            "AKo".to_string(),
            "QJo".to_string(),
            "T9o".to_string(),
            "87o".to_string(),
        ]
        .to_vec(),
    );
    let board = get_card_mask("");
    c.bench_function("exact_equity_seq(AKo/QJo/T9o/87o, preflop)", |b| {
        b.iter(|| exact_equity_seq(&ranges, board).unwrap())
    });
}

// ---------------------------------------------------------------------------
// Heads-up flop (intermediate board, no-lookup path)
// ---------------------------------------------------------------------------

fn bench_hu_flop_threaded(c: &mut Criterion) {
    let ranges =
        HandRange::from_strings(["AA".to_string(), "KK".to_string()].to_vec());
    let board = get_card_mask("2h3d4c");
    c.bench_function("exact_equity(AA vs KK, flop, 1-thread)", |b| {
        b.iter(|| exact_equity(&ranges, board, 1).unwrap())
    });
}

fn bench_hu_flop_seq(c: &mut Criterion) {
    let ranges =
        HandRange::from_strings(["AA".to_string(), "KK".to_string()].to_vec());
    let board = get_card_mask("2h3d4c");
    c.bench_function("exact_equity_seq(AA vs KK, flop)", |b| {
        b.iter(|| exact_equity_seq(&ranges, board).unwrap())
    });
}

criterion_group!(
    benches,
    bench_hu_preflop_threaded,
    bench_hu_preflop_seq,
    bench_hu_weighted_threaded,
    bench_hu_weighted_seq,
    bench_4way_preflop_threaded,
    bench_4way_preflop_seq,
    bench_hu_flop_threaded,
    bench_hu_flop_seq,
);
criterion_main!(benches);

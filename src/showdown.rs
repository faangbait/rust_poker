//! Exact showdown-tally kernel for an all-in preflop equity cache (2..=9 players).
//!
//! See the `simd-kernel` plan. This crate ships ONLY the hot kernel
//! ([`tally_showdown`]), its scalar oracle ([`tally_showdown_scalar`]), and parity
//! tests. Matchup enumeration, board enumeration, isomorphism, and the on-disk
//! cache live in the consumer crate.
//!
//! Equity is accumulated in integer units of `D = lcm(1..=9) = 2520` so split pots
//! divide exactly: a board class of weight `w` with `k` tied winners gives each
//! winner `w * 2520 / k` units, and `2520` is divisible by every `k` in `1..=9`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Common denominator: lcm(1..=9). Divisible by every tie count k in 1..=9.
pub const D: u64 = 2520;

/// Max players supported by the kernel.
const MAX_PLAYERS: usize = 9;

/// Flush SIMD partials to u64 every this many batches to keep the per-lane u32
/// accumulators from overflowing. 4096 batches = 32768 boards; safe for modest
/// per-board weights (board-class multiplicities, ~<=24 in practice).
#[cfg(target_arch = "x86_64")]
const FLUSH_BATCHES: usize = 4096;

/// Exact unit accounting for a single board, shared by the scalar oracle, the
/// AVX2 tie fallback, and the AVX2 tail. Adds into `out_units`; caller handles W.
#[inline]
fn board_units(scores: &[&[u16]], b: usize, w: u32, out_units: &mut [u64]) {
    let n = scores.len();
    let mut max = 0u16;
    for p in 0..n {
        let s = scores[p][b];
        if s > max {
            max = s;
        }
    }
    let mut k = 0u64;
    for p in 0..n {
        if scores[p][b] == max {
            k += 1;
        }
    }
    // D is divisible by every k in 1..=9, so this division is exact.
    let units = (w as u64) * D / k;
    for p in 0..n {
        if scores[p][b] == max {
            out_units[p] += units;
        }
    }
}

/// Scalar reference / correctness oracle. Generic over N = 2..=9 players.
///
/// `scores[p][b]` = player p's hand score on board b (higher = better, >= 1).
/// `weights[b]`   = multiplicity of board b (1 for concrete boards, >1 for a class).
/// `valid`        = bitset, len `ceil(B/64)`; bit b set => board b playable here.
/// `out_units[p]` += equity numerator in units of `D`. Caller-zeroed.
///
/// Returns total valid weight `W`. Invariant: `sum_p out_units == D * W`.
pub fn tally_showdown_scalar(
    scores: &[&[u16]],
    weights: &[u32],
    valid: &[u64],
    out_units: &mut [u64],
) -> u64 {
    let b = weights.len();
    let mut w_total: u64 = 0;
    for board in 0..b {
        if (valid[board >> 6] >> (board & 63)) & 1 == 0 {
            continue;
        }
        let w = weights[board];
        w_total += w as u64;
        board_units(scores, board, w, out_units);
    }
    w_total
}

/// Exact showdown tally. AVX2 when available, scalar otherwise. Same contract as
/// [`tally_showdown_scalar`].
pub fn tally_showdown(
    scores: &[&[u16]],
    weights: &[u32],
    valid: &[u64],
    out_units: &mut [u64],
) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: gated on runtime AVX2 detection.
            return unsafe { tally_avx2(scores, weights, valid, out_units) };
        }
    }
    tally_showdown_scalar(scores, weights, valid, out_units)
}

/// Horizontal sum of 8 u32 lanes into u64 (no overflow: small values, rare path).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_epu32(v: __m256i) -> u64 {
    let mut tmp = [0u32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr().cast(), v);
    tmp.iter().map(|&x| x as u64).sum()
}

/// AVX2 hot kernel. 8 boards/batch: u16 scores widened to u32 to match u32 weights.
///
/// Fast path (common): no valid lane in the batch is a tie. Per player, add masked
/// weights where score == lane max into a u32x8 partial; flush to `out_units`
/// (scaled by D) and W periodically. Slow path (rare): a batch with a tie lane is
/// handled lane-by-lane in scalar for exact `w * D / k` accounting.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tally_avx2(
    scores: &[&[u16]],
    weights: &[u32],
    valid: &[u64],
    out_units: &mut [u64],
) -> u64 {
    let n = scores.len();
    let b = weights.len();
    let full = b / 8;

    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi32(1);
    let sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    let mut pacc = [zero; MAX_PLAYERS]; // per-player weight-sum partials (pre-D)
    let mut wacc = zero; // valid-weight partials
    let mut w_total: u64 = 0;
    let mut since_flush = 0usize;

    let mut eqs = [zero; MAX_PLAYERS];

    for batch in 0..full {
        let base = batch * 8;

        // valid mask: 8 bits -> 8 all-ones/zero lanes.
        let word = valid[base >> 6];
        let mask8 = ((word >> (base & 63)) & 0xff) as i32;
        let bits = _mm256_set1_epi32(mask8);
        let validmask = _mm256_cmpeq_epi32(_mm256_and_si256(bits, sel), sel);

        // weights, invalid lanes zeroed.
        let wv = _mm256_loadu_si256(weights.as_ptr().add(base).cast());
        let mw = _mm256_and_si256(wv, validmask);

        // lane-wise max over players (scores widened u16 -> u32).
        let mut sp = [zero; MAX_PLAYERS];
        let mut maxv = zero;
        for p in 0..n {
            let s128 = _mm_loadu_si128(scores[p].as_ptr().add(base).cast());
            let s = _mm256_cvtepu16_epi32(s128);
            sp[p] = s;
            maxv = _mm256_max_epu32(maxv, s);
        }

        // tie count k per lane, and remember eq masks for the fast path.
        let mut kvec = zero;
        for p in 0..n {
            let eq = _mm256_cmpeq_epi32(sp[p], maxv);
            eqs[p] = eq;
            kvec = _mm256_add_epi32(kvec, _mm256_and_si256(eq, one));
        }

        // tie lanes that actually matter: k > 1 AND weight != 0.
        let kgt1 = _mm256_cmpgt_epi32(kvec, one);
        let weq0 = _mm256_cmpeq_epi32(mw, zero);
        let tie = _mm256_andnot_si256(weq0, kgt1); // (mw!=0) & (k>1)

        if _mm256_testz_si256(tie, tie) == 1 {
            // fast path: single winner per valid lane.
            for p in 0..n {
                pacc[p] = _mm256_add_epi32(pacc[p], _mm256_and_si256(mw, eqs[p]));
            }
            wacc = _mm256_add_epi32(wacc, mw);
        } else {
            // slow path: exact lane-by-lane (rare).
            for lane in 0..8 {
                if (mask8 >> lane) & 1 == 0 {
                    continue;
                }
                let w = weights[base + lane];
                w_total += w as u64;
                board_units(scores, base + lane, w, out_units);
            }
        }

        since_flush += 1;
        if since_flush >= FLUSH_BATCHES {
            for p in 0..n {
                out_units[p] += D * hsum_epu32(pacc[p]);
                pacc[p] = zero;
            }
            w_total += hsum_epu32(wacc);
            wacc = zero;
            since_flush = 0;
        }
    }

    // final flush of partials.
    for p in 0..n {
        out_units[p] += D * hsum_epu32(pacc[p]);
    }
    w_total += hsum_epu32(wacc);

    // tail (boards not in a full batch of 8).
    for board in (full * 8)..b {
        if (valid[board >> 6] >> (board & 63)) & 1 == 0 {
            continue;
        }
        let w = weights[board];
        w_total += w as u64;
        board_units(scores, board, w, out_units);
    }

    w_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn make_valid(b: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
        let mut v = vec![0u64; (b + 63) / 64];
        for i in 0..b {
            if set(i) {
                v[i >> 6] |= 1u64 << (i & 63);
            }
        }
        v
    }

    #[test]
    fn invariant_and_parity_random() {
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        for _ in 0..200 {
            let n = rng.gen_range(2, 10);
            let b = rng.gen_range(1, 600);

            // Low score range forces frequent ties to exercise the slow path.
            let cols: Vec<Vec<u16>> = (0..n)
                .map(|_| (0..b).map(|_| rng.gen_range(1, 6) as u16).collect())
                .collect();
            let scores: Vec<&[u16]> = cols.iter().map(|c| c.as_slice()).collect();

            let weights: Vec<u32> = (0..b).map(|_| rng.gen_range(1, 25)).collect();
            let valid_bits: Vec<bool> = (0..b).map(|_| rng.gen_bool(0.85)).collect();
            let valid = make_valid(b, |i| valid_bits[i]);

            let mut out_simd = vec![0u64; n];
            let mut out_scalar = vec![0u64; n];

            let w_simd = tally_showdown(&scores, &weights, &valid, &mut out_simd);
            let w_scalar =
                tally_showdown_scalar(&scores, &weights, &valid, &mut out_scalar);

            assert_eq!(w_simd, w_scalar, "W mismatch");
            assert_eq!(out_simd, out_scalar, "SIMD != scalar");

            let sum: u64 = out_simd.iter().sum();
            assert_eq!(sum, D * w_simd, "sum_p out_units != D * W");
        }
    }

    #[test]
    fn wide_scores_no_ties() {
        // Distinct high scores -> always fast path; exercises widening + flush.
        let mut rng = SmallRng::seed_from_u64(42);
        let n = 6;
        let b = 5000;
        let cols: Vec<Vec<u16>> = (0..n)
            .map(|_| (0..b).map(|_| rng.gen_range(1, 60000) as u16).collect())
            .collect();
        let scores: Vec<&[u16]> = cols.iter().map(|c| c.as_slice()).collect();
        let weights = vec![1u32; b];
        let valid = make_valid(b, |_| true);

        let mut out = vec![0u64; n];
        let w = tally_showdown(&scores, &weights, &valid, &mut out);
        let sum: u64 = out.iter().sum();
        assert_eq!(sum, D * w);
    }

    #[test]
    fn forced_tie_split() {
        // Two players, identical scores, one board, weight 1 -> 50/50 split.
        let s: Vec<u16> = vec![7];
        let scores: Vec<&[u16]> = vec![&s, &s];
        let weights = vec![1u32];
        let valid = vec![1u64];
        let mut out = vec![0u64; 2];
        let w = tally_showdown(&scores, &weights, &valid, &mut out);
        assert_eq!(w, 1);
        assert_eq!(out[0], D / 2);
        assert_eq!(out[1], D / 2);
        assert_eq!(out[0] + out[1], D);
    }

    #[test]
    fn invalid_boards_contribute_nothing() {
        let a: Vec<u16> = vec![10, 99];
        let c: Vec<u16> = vec![20, 1];
        let scores: Vec<&[u16]> = vec![&a, &c];
        let weights = vec![5u32, 1000u32];
        let valid = make_valid(2, |i| i == 0); // board 1 invalid
        let mut out = vec![0u64; 2];
        let w = tally_showdown(&scores, &weights, &valid, &mut out);
        assert_eq!(w, 5);
        assert_eq!(out[0], 0); // player 0 lost the only valid board
        assert_eq!(out[1], 5 * D); // player 1 won it outright
    }
}

# RustPoker

[![docs.rs](https://docs.rs/rust_poker/badge.svg)](https://docs.rs/rust_poker)
[![crates.io](https://img.shields.io/crates/v/rust_poker.svg)](https://crates.io/crates/rust_poker)

A poker library written in Rust.

This is a fork of [kmurf1999/rust_poker](https://github.com/kmurf1999/rust_poker) modified for faster resolution of postflop spots.

- Multithreaded range vs range equity calculation
- Fast hand evaluation
- Efficient hand indexing

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
rust_poker = "0.2.0"
```

**Note**: The first build of an application using `rust_poker` will take extra time to generate the hand evaluation table.

## Hand Evaluator

Evaluates the strength of any poker hand using up to 7 cards.

### Usage

```rust
use rust_poker::hand_evaluator::{Hand, CARDS, evaluate};
// cards are indexed 0->51 where index is 4 * rank + suit
let hand = Hand::empty() + CARDS[0] + CARDS[1];
let score = evaluate(&hand);
println!("score: {}", score);
```

## Equity Calculator

Calculates range vs range equities for up to 6 players specified by equilab-style range strings.
Supports both Monte Carlo simulation (`approx_equity`) and exact enumeration (`exact_equity`).

### Usage

```rust
use rust_poker::hand_range::{HandRange, get_card_mask};
use rust_poker::equity_calculator::approx_equity;

let ranges = HandRange::from_strings(["AK,22+".to_string(), "random".to_string()].to_vec());
let board_mask = get_card_mask("2h3d4c");
let stdev_target = 0.01;
let n_threads = 4;
let equities = approx_equity(&ranges, board_mask, n_threads, stdev_target).unwrap();
println!("player 1 equity: {}", equities[0]);
```

```rust
use rust_poker::hand_range::{HandRange, get_card_mask};
use rust_poker::equity_calculator::exact_equity;

let ranges = HandRange::from_strings(["AA".to_string(), "random".to_string()].to_vec());
let board_mask = get_card_mask("2h3d4c");
let n_threads = 4;
let equities = exact_equity(&ranges, board_mask, n_threads).unwrap();
println!("player 1 equity: {}", equities[0]);
```

## Credit

Based on **zekyll's** C++ equity calculator, [OMPEval](https://github.com/zekyll/OMPEval), originally ported to Rust by [kmurf1999](https://github.com/kmurf1999/rust_poker).

## License

This project is MIT Licensed

Copyright (c) 2020 Kyle Murphy

# rl-rs

Reinforcement learning in Rust. From Q-tables to policy gradients.

## Overview

`rl-rs` provides the foundational algorithms of reinforcement learning, from exact dynamic programming (policy evaluation, policy iteration, value iteration) through sample-based methods (TD, Q-learning, REINFORCE) to bandit algorithms (ε-greedy, UCB1, Thompson Sampling).

### What's inside

| Module | What it does |
|---|---|
| `mdp` | Core MDP traits and types |
| `dp` | Dynamic programming: policy evaluation, policy iteration, value iteration |
| `td` | Temporal-difference learning (TD(0), TD(λ)) |
| `q_learning` | Q-learning and extensions |
| `eligibility` | Eligibility traces |
| `policy_gradient` | REINFORCE and policy gradient methods |
| `bandit` | Multi-armed bandits: ε-greedy, UCB1, Thompson Sampling |
| `value` | State/action value function abstractions |
| `gridworld` | Grid world environment for experimentation |

## Quick start

```toml
[dependencies]
rl-rs = "0.1"
```

```sh
cargo add rl-rs
```

## Examples

### Value iteration on a grid world

```rust
use rl_rs::gridworld::GridWorld;
use rl_rs::dp::value_iteration;
use rl_rs::mdp::MDP;

let gw = GridWorld::new(5, 5, vec![(4, 4)], vec![(1, 1), (3, 3)]);
let v = value_iteration(&gw, 0.99, 1e-6, 1000);
println!("{:?}", v);
```

### Q-learning

```rust
use rl_rs::gridworld::GridWorld;
use rl_rs::q_learning::*;
use rl_rs::value::ActionValueFunction;
use rl_rs::mdp::MDP;

let gw = GridWorld::new(5, 5, vec![(4, 4)], vec![(1, 1), (3, 3)]);
let mut q = TabularQ::new(gw.num_states(), gw.num_actions());
q_learning(&gw, &mut q, 0.1, 0.99, 0.1, 10_000, 42);
```

### Multi-armed bandit

```rust
use rl_rs::bandit::*;

let arms = vec![BernoulliArm::new(0.2), BernoulliArm::new(0.5), BernoulliArm::new(0.8)];
let rewards = run_epsilon_greedy(&arms, 1000, 0.1, 42);
```

### Policy gradient (REINFORCE)

```rust
use rl_rs::policy_gradient::*;
use rl_rs::mdp::MDP;
use rl_rs::gridworld::GridWorld;

let gw = GridWorld::new(5, 5, vec![(4, 4)], vec![(1, 1), (3, 3)]);
let policy = reinforce(&gw, 0.01, 500, 100, 42);
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

//! # rl-rs
//!
//! A reinforcement learning library implementing MDP formalism, dynamic programming,
//! temporal difference learning, Q-learning, policy gradient methods, multi-armed bandits,
//! and grid world environments.

pub mod mdp;
pub mod value;
pub mod dp;
pub mod td;
pub mod q_learning;
pub mod policy_gradient;
pub mod bandit;
pub mod eligibility;
pub mod gridworld;

#[cfg(test)]
mod tests;

pub use mdp::*;
pub use value::*;
pub use dp::*;
pub use td::*;
pub use q_learning::*;
pub use policy_gradient::*;
pub use bandit::*;
pub use eligibility::*;
pub use gridworld::*;

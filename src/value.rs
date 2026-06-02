//! Value functions (state-value V, action-value Q).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// State-value function V(s).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateValueFunction<S: Clone + Eq + std::hash::Hash> {
    pub values: HashMap<S, f64>,
    pub default: f64,
}

impl<S: Clone + Eq + std::hash::Hash> StateValueFunction<S> {
    pub fn new(default: f64) -> Self {
        Self {
            values: HashMap::new(),
            default,
        }
    }

    pub fn get(&self, state: &S) -> f64 {
        self.values.get(state).copied().unwrap_or(self.default)
    }

    pub fn set(&mut self, state: S, value: f64) {
        self.values.insert(state, value);
    }

    pub fn update(&mut self, state: &S, delta: f64) {
        let current = self.get(state);
        self.values.insert(state.clone(), current + delta);
    }
}

impl<S: Clone + Eq + std::hash::Hash> Default for StateValueFunction<S> {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Action-value function Q(s, a).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionValueFunction<S: Clone + Eq + std::hash::Hash, A: Clone + Eq + std::hash::Hash> {
    pub values: HashMap<(S, A), f64>,
    pub default: f64,
}

impl<S: Clone + Eq + std::hash::Hash, A: Clone + Eq + std::hash::Hash>
    ActionValueFunction<S, A>
{
    pub fn new(default: f64) -> Self {
        Self {
            values: HashMap::new(),
            default,
        }
    }

    pub fn get(&self, state: &S, action: &A) -> f64 {
        self.values
            .get(&(state.clone(), action.clone()))
            .copied()
            .unwrap_or(self.default)
    }

    pub fn set(&mut self, state: S, action: A, value: f64) {
        self.values.insert((state, action), value);
    }

    pub fn update(&mut self, state: &S, action: &A, delta: f64) {
        let current = self.get(state, action);
        self.values
            .insert((state.clone(), action.clone()), current + delta);
    }

    /// Best action for a given state.
    pub fn greedy_action(&self, state: &S, actions: &[A]) -> Option<A>
    where
        A: Copy,
    {
        actions
            .iter()
            .max_by(|a, b| {
                self.get(state, a)
                    .partial_cmp(&self.get(state, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Max Q value for a given state.
    pub fn max_q(&self, state: &S, actions: &[A]) -> f64
    where
        A: Copy,
    {
        if actions.is_empty() {
            return 0.0;
        }
        actions
            .iter()
            .map(|a| self.get(state, a))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl<S: Clone + Eq + std::hash::Hash, A: Clone + Eq + std::hash::Hash> Default
    for ActionValueFunction<S, A>
{
    fn default() -> Self {
        Self::new(0.0)
    }
}

//! Eligibility traces for TD(λ) and related algorithms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

/// Eligibility traces: tracks which states have been visited recently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityTraces<S: Clone + Eq + Hash> {
    traces: HashMap<S, f64>,
}

impl<S: Clone + Eq + Hash> EligibilityTraces<S> {
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
        }
    }

    /// Get the eligibility trace value for a state.
    pub fn get(&self, state: &S) -> f64 {
        self.traces.get(state).copied().unwrap_or(0.0)
    }

    /// Set the trace value for a state.
    pub fn set(&mut self, state: &S, value: f64) {
        self.traces.insert(state.clone(), value);
    }

    /// Accumulating trace: add value to current trace.
    pub fn update(&mut self, state: &S, increment: f64) {
        let current = self.get(state);
        self.traces.insert(state.clone(), current + increment);
    }

    /// Decay all traces by a factor.
    pub fn decay(&mut self, factor: f64) {
        self.traces.retain(|_, v| {
            *v *= factor;
            *v > 1e-10
        });
    }

    /// Get all non-zero traces.
    pub fn all_traces(&self) -> Vec<(S, f64)> {
        self.traces
            .iter()
            .map(|(s, &v)| (s.clone(), v))
            .collect()
    }

    /// Reset all traces.
    pub fn reset(&mut self) {
        self.traces.clear();
    }

    /// Check if traces are empty.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }
}

impl<S: Clone + Eq + Hash> Default for EligibilityTraces<S> {
    fn default() -> Self {
        Self::new()
    }
}

//! Markov Decision Process formalism.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

/// A state in an MDP.
pub trait State: Clone + Copy + Eq + Hash + Send + Sync + std::fmt::Debug + 'static {}

/// Blanket impl for primitive types that satisfy our constraints.
impl<T: Clone + Copy + Eq + Hash + Send + Sync + std::fmt::Debug + 'static> State for T {}

/// An action in an MDP.
pub trait Action: Clone + Copy + Eq + Hash + Send + Sync + std::fmt::Debug + 'static {}

impl<T: Clone + Copy + Eq + Hash + Send + Sync + std::fmt::Debug + 'static> Action for T {}

/// A transition outcome: (next_state, reward, probability).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition<S: State, A: Action> {
    pub next_state: S,
    pub reward: f64,
    pub probability: f64,
    #[serde(skip)]
    _action: std::marker::PhantomData<A>,
}

impl<S: State, A: Action> Transition<S, A> {
    pub fn new(next_state: S, reward: f64, probability: f64) -> Self {
        Self {
            next_state,
            reward,
            probability,
            _action: std::marker::PhantomData,
        }
    }
}

/// A deterministic transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicTransition<S: State> {
    pub next_state: S,
    pub reward: f64,
}

/// Markov Decision Process trait.
pub trait MDP: Send + Sync {
    type S: State;
    type A: Action;

    /// All states in the MDP.
    fn states(&self) -> Vec<Self::S>;

    /// All actions available in a given state.
    fn actions(&self, state: Self::S) -> Vec<Self::A>;

    /// Transition probabilities for a state-action pair.
    fn transitions(&self, state: Self::S, action: Self::A) -> Vec<Transition<Self::S, Self::A>>;

    /// Discount factor gamma ∈ [0, 1).
    fn discount_factor(&self) -> f64;

    /// Whether a state is terminal.
    fn is_terminal(&self, state: Self::S) -> bool;

    /// All actions in the MDP.
    fn all_actions(&self) -> Vec<Self::A>;
}

/// A simple tabular MDP using state/action indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabularMDP {
    pub num_states: usize,
    pub num_actions: usize,
    /// transitions[s][a] = list of (next_state, reward, probability)
    pub transitions: Vec<Vec<Vec<(usize, f64, f64)>>>,
    pub discount: f64,
    pub terminals: Vec<bool>,
}

impl TabularMDP {
    pub fn new(num_states: usize, num_actions: usize, discount: f64) -> Self {
        Self {
            num_states,
            num_actions,
            transitions: vec![vec![vec![]; num_actions]; num_states],
            discount,
            terminals: vec![false; num_states],
        }
    }

    pub fn add_transition(
        &mut self,
        state: usize,
        action: usize,
        next_state: usize,
        reward: f64,
        prob: f64,
    ) {
        self.transitions[state][action].push((next_state, reward, prob));
    }

    pub fn set_terminal(&mut self, state: usize) {
        self.terminals[state] = true;
    }
}

impl MDP for TabularMDP {
    type S = usize;
    type A = usize;

    fn states(&self) -> Vec<Self::S> {
        (0..self.num_states).collect()
    }

    fn actions(&self, _state: Self::S) -> Vec<Self::A> {
        (0..self.num_actions).collect()
    }

    fn transitions(&self, state: Self::S, action: Self::A) -> Vec<Transition<Self::S, Self::A>> {
        self.transitions[state][action]
            .iter()
            .map(|(ns, r, p)| Transition::new(*ns, *r, *p))
            .collect()
    }

    fn discount_factor(&self) -> f64 {
        self.discount
    }

    fn is_terminal(&self, state: Self::S) -> bool {
        self.terminals[state]
    }

    fn all_actions(&self) -> Vec<Self::A> {
        (0..self.num_actions).collect()
    }
}

/// A policy mapping states to action probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticPolicy<S: State, A: Action> {
    /// action_probs[state] = HashMap<action, probability>
    pub action_probs: HashMap<S, HashMap<A, f64>>,
}

impl<S: State, A: Action> StochasticPolicy<S, A> {
    pub fn new() -> Self {
        Self {
            action_probs: HashMap::new(),
        }
    }

    pub fn set_prob(&mut self, state: S, action: A, prob: f64) {
        self.action_probs
            .entry(state)
            .or_insert_with(HashMap::new)
            .insert(action, prob);
    }

    pub fn get_prob(&self, state: S, action: A) -> f64 {
        self.action_probs
            .get(&state)
            .and_then(|m| m.get(&action))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn sample_action(&self, state: S, rng: &mut impl rand::Rng) -> Option<A> {
        let probs = self.action_probs.get(&state)?;
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (&action, &prob) in probs {
            cumulative += prob;
            if r < cumulative {
                return Some(action);
            }
        }
        probs.keys().next().copied()
    }
}

/// A deterministic policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicPolicy<S: State, A: Action> {
    pub mapping: HashMap<S, A>,
}

impl<S: State, A: Action> DeterministicPolicy<S, A> {
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
        }
    }

    pub fn set(&mut self, state: S, action: A) {
        self.mapping.insert(state, action);
    }

    pub fn get(&self, state: S) -> Option<A> {
        self.mapping.get(&state).copied()
    }
}

impl<S: State, A: Action> Default for DeterministicPolicy<S, A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State, A: Action> Default for StochasticPolicy<S, A> {
    fn default() -> Self {
        Self::new()
    }
}

//! Policy gradient methods: REINFORCE basics.

use crate::mdp::MDP;
use crate::td::sample_transition;
use std::hash::Hash;

/// A simple softmax policy parameterized by preferences (theta) for each state-action pair.
#[derive(Debug, Clone)]
pub struct SoftmaxPolicy<S: Clone + Eq + Hash, A: Clone + Eq + Hash + Copy> {
    /// Preferences: theta[(s, a)]
    pub theta: std::collections::HashMap<(S, A), f64>,
    pub alpha: f64,
}

impl<S: Clone + Eq + Hash, A: Clone + Eq + Hash + Copy> SoftmaxPolicy<S, A> {
    pub fn new(alpha: f64) -> Self {
        Self {
            theta: std::collections::HashMap::new(),
            alpha,
        }
    }

    /// Get preference value.
    pub fn preference(&self, state: &S, action: &A) -> f64 {
        self.theta
            .get(&(state.clone(), *action))
            .copied()
            .unwrap_or(0.0)
    }

    /// Compute action probabilities using softmax.
    pub fn action_probs(&self, state: &S, actions: &[A]) -> std::collections::HashMap<A, f64> {
        let max_pref = actions
            .iter()
            .map(|a| self.preference(state, a))
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_sum: f64 = actions
            .iter()
            .map(|a| (self.preference(state, a) - max_pref).exp())
            .sum();

        let mut probs = std::collections::HashMap::new();
        for &action in actions {
            let p = (self.preference(state, &action) - max_pref).exp() / exp_sum;
            probs.insert(action, p);
        }
        probs
    }

    /// Sample an action.
    pub fn sample(&self, state: &S, actions: &[A], rng: &mut impl rand::Rng) -> A {
        let probs = self.action_probs(state, actions);
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for &action in actions {
            cumulative += probs[&action];
            if r < cumulative {
                return action;
            }
        }
        actions[actions.len() - 1]
    }

    /// Update preferences using REINFORCE gradient.
    /// θ(s,a) ← θ(s,a) + α * G * [1 - π(a|s)]  if a == taken action
    /// θ(s,a) ← θ(s,a) - α * G * π(a|s)           otherwise
    pub fn reinforce_update(
        &mut self,
        state: &S,
        action_taken: A,
        return_g: f64,
        actions: &[A],
    ) {
        let probs = self.action_probs(state, actions);
        for &a in actions {
            let prob = probs[&a];
            let gradient = if a == action_taken { 1.0 - prob } else { -prob };
            let current = self.preference(state, &a);
            self.theta.insert(
                (state.clone(), a),
                current + self.alpha * return_g * gradient,
            );
        }
    }
}

/// An episode trajectory for policy gradient methods.
#[derive(Debug, Clone)]
pub struct Trajectory<S: Clone, A: Clone> {
    pub states: Vec<S>,
    pub actions: Vec<A>,
    pub rewards: Vec<f64>,
}

impl<S: Clone, A: Clone> Trajectory<S, A> {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    pub fn push(&mut self, state: S, action: A, reward: f64) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
    }

    /// Compute discounted returns G_t for each timestep.
    pub fn returns(&self, gamma: f64) -> Vec<f64> {
        let mut returns = vec![0.0; self.rewards.len()];
        if self.rewards.is_empty() {
            return returns;
        }

        let mut g = 0.0;
        for t in (0..self.rewards.len()).rev() {
            g = self.rewards[t] + gamma * g;
            returns[t] = g;
        }
        returns
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl<S: Clone, A: Clone> Default for Trajectory<S, A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Collect a trajectory using the given policy on an MDP.
pub fn collect_trajectory<M: MDP, R: rand::Rng>(
    mdp: &M,
    policy: &dyn Fn(M::S, &mut R) -> M::A,
    start_state: M::S,
    max_steps: usize,
    rng: &mut R,
) -> Trajectory<M::S, M::A>
where
    M::S: Hash + Clone,
    M::A: Clone,
{
    let mut traj = Trajectory::new();
    let mut state = start_state;

    for _ in 0..max_steps {
        if mdp.is_terminal(state) {
            break;
        }

        let action = policy(state, rng);
        let transitions = mdp.transitions(state, action);
        if transitions.is_empty() {
            break;
        }

        let (next_state, reward) = sample_transition(&transitions, rng);

        traj.push(state, action, reward);
        state = next_state;
    }

    traj
}

/// REINFORCE algorithm: update policy weights using a full trajectory.
pub fn reinforce_update<S: Clone + Eq + Hash, A: Clone + Eq + Hash + Copy>(
    policy: &mut SoftmaxPolicy<S, A>,
    trajectory: &Trajectory<S, A>,
    gamma: f64,
    actions: &[A],
) {
    let returns = trajectory.returns(gamma);

    for t in 0..trajectory.len() {
        policy.reinforce_update(
            &trajectory.states[t],
            trajectory.actions[t],
            returns[t],
            actions,
        );
    }
}

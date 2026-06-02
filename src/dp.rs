//! Dynamic programming: policy evaluation, policy iteration, value iteration.

use crate::mdp::{DeterministicPolicy, MDP, StochasticPolicy, Transition};
use crate::value::{ActionValueFunction, StateValueFunction};
use std::hash::Hash;

/// Evaluate a stochastic policy using iterative policy evaluation.
///
/// Returns the state-value function V^π for the given policy.
pub fn policy_evaluation_stochastic<M: MDP>(
    mdp: &M,
    policy: &StochasticPolicy<M::S, M::A>,
    theta: f64,
    max_iterations: usize,
) -> StateValueFunction<M::S>
where
    M::S: Hash,
    M::A: Hash,
{
    let mut v = StateValueFunction::new(0.0);
    let gamma = mdp.discount_factor();

    for _ in 0..max_iterations {
        let mut delta: f64 = 0.0;
        for state in mdp.states() {
            if mdp.is_terminal(state) {
                continue;
            }
            let old_v = v.get(&state);
            let mut new_v: f64 = 0.0;

            for action in mdp.actions(state) {
                let action_prob = policy.get_prob(state, action);
                if action_prob == 0.0 {
                    continue;
                }

                let expected_return = expected_return(mdp, &v, state, action, gamma);
                new_v += action_prob * expected_return;
            }

            v.set(state, new_v);
            delta = delta.max((new_v - old_v).abs());
        }

        if delta < theta {
            break;
        }
    }

    v
}

/// Compute the expected return for a state-action pair given current V.
fn expected_return<M: MDP>(
    mdp: &M,
    v: &StateValueFunction<M::S>,
    state: M::S,
    action: M::A,
    gamma: f64,
) -> f64
where
    M::S: Hash,
{
    let mut expected = 0.0;
    for Transition {
        next_state,
        reward,
        probability,
        ..
    } in mdp.transitions(state, action)
    {
        let next_v = if mdp.is_terminal(next_state) {
            0.0
        } else {
            v.get(&next_state)
        };
        expected += probability * (reward + gamma * next_v);
    }
    expected
}

/// Evaluate a deterministic policy.
pub fn policy_evaluation_deterministic<M: MDP>(
    mdp: &M,
    policy: &DeterministicPolicy<M::S, M::A>,
    theta: f64,
    max_iterations: usize,
) -> StateValueFunction<M::S>
where
    M::S: Hash,
{
    let mut stochastic = StochasticPolicy::new();
    for state in mdp.states() {
        if let Some(action) = policy.get(state) {
            stochastic.set_prob(state, action, 1.0);
        }
    }
    policy_evaluation_stochastic(mdp, &stochastic, theta, max_iterations)
}

/// Policy iteration: alternate between policy evaluation and policy improvement.
///
/// Returns the optimal deterministic policy and its value function.
pub fn policy_iteration<M: MDP>(
    mdp: &M,
    theta: f64,
    max_iterations: usize,
) -> (DeterministicPolicy<M::S, M::A>, StateValueFunction<M::S>)
where
    M::S: Hash,
{
    let mut policy = DeterministicPolicy::new();

    // Initialize with first available action for each state
    for state in mdp.states() {
        let state_actions = mdp.actions(state);
        if let Some(&first) = state_actions.first() {
            policy.set(state, first);
        }
    }

    let gamma = mdp.discount_factor();
    let mut v = StateValueFunction::new(0.0);

    for _ in 0..max_iterations {
        // Policy evaluation
        v = policy_evaluation_deterministic(mdp, &policy, theta, max_iterations);

        // Policy improvement
        let mut policy_stable = true;
        for state in mdp.states() {
            if mdp.is_terminal(state) {
                continue;
            }

            let old_action = policy.get(state);
            let mut best_action = None;
            let mut best_value = f64::NEG_INFINITY;

            for action in mdp.actions(state) {
                let val = expected_return(mdp, &v, state, action, gamma);
                if val > best_value {
                    best_value = val;
                    best_action = Some(action);
                }
            }

            if let Some(ba) = best_action {
                if old_action != Some(ba) {
                    policy_stable = false;
                }
                policy.set(state, ba);
            }
        }

        if policy_stable {
            break;
        }
    }

    (policy, v)
}

/// Value iteration: compute the optimal value function directly.
///
/// Returns the optimal value function and the greedy policy derived from it.
pub fn value_iteration<M: MDP>(
    mdp: &M,
    theta: f64,
    max_iterations: usize,
) -> (DeterministicPolicy<M::S, M::A>, StateValueFunction<M::S>)
where
    M::S: Hash,
{
    let mut v = StateValueFunction::new(0.0);
    let gamma = mdp.discount_factor();

    for _ in 0..max_iterations {
        let mut delta: f64 = 0.0;
        for state in mdp.states() {
            if mdp.is_terminal(state) {
                continue;
            }

            let mut max_q = f64::NEG_INFINITY;
            for action in mdp.actions(state) {
                let q = expected_return(mdp, &v, state, action, gamma);
                max_q = max_q.max(q);
            }

            let old_v = v.get(&state);
            let new_v = if max_q == f64::NEG_INFINITY {
                0.0
            } else {
                max_q
            };
            v.set(state, new_v);
            delta = delta.max((new_v - old_v).abs());
        }

        if delta < theta {
            break;
        }
    }

    // Extract greedy policy
    let mut policy = DeterministicPolicy::new();
    for state in mdp.states() {
        if mdp.is_terminal(state) {
            continue;
        }
        let mut best_action = None;
        let mut best_value = f64::NEG_INFINITY;
        for action in mdp.actions(state) {
            let val = expected_return(mdp, &v, state, action, gamma);
            if val > best_value {
                best_value = val;
                best_action = Some(action);
            }
        }
        if let Some(ba) = best_action {
            policy.set(state, ba);
        }
    }

    (policy, v)
}

/// Compute the action-value function Q from the state-value function V.
pub fn v_to_q<M: MDP>(
    mdp: &M,
    v: &StateValueFunction<M::S>,
) -> ActionValueFunction<M::S, M::A>
where
    M::S: Hash,
{
    let gamma = mdp.discount_factor();
    let mut q = ActionValueFunction::new(0.0);

    for state in mdp.states() {
        for action in mdp.actions(state) {
            let val = expected_return(mdp, v, state, action, gamma);
            q.set(state, action, val);
        }
    }

    q
}

//! Temporal difference learning: TD(0), TD(λ).

use crate::eligibility::EligibilityTraces;
use crate::mdp::MDP;
use crate::value::StateValueFunction;
use std::hash::Hash;

/// Experience tuple: (state, action, reward, next_state, done).
#[derive(Debug, Clone)]
pub struct Experience<S, A> {
    pub state: S,
    pub action: A,
    pub reward: f64,
    pub next_state: S,
    pub done: bool,
}

/// TD(0) update: single-step temporal difference.
///
/// V(s) ← V(s) + α [r + γV(s') - V(s)]
pub fn td0_update<S: Clone + Eq + Hash>(
    v: &mut StateValueFunction<S>,
    state: &S,
    reward: f64,
    next_state: &S,
    done: bool,
    alpha: f64,
    gamma: f64,
) -> f64 {
    let next_v = if done { 0.0 } else { v.get(next_state) };
    let current_v = v.get(state);
    let td_error = reward + gamma * next_v - current_v;
    v.set(state.clone(), current_v + alpha * td_error);
    td_error
}

/// TD(λ) with eligibility traces.
///
/// For each step:
/// 1. Update eligibility traces: e(s) += 1 (accumulating) or e(s) = 1 (replacing)
/// 2. Compute TD error: δ = r + γV(s') - V(s)
/// 3. For all states: V(s) += α * δ * e(s)
/// 4. Decay traces: e(s) *= γλ
pub fn td_lambda_update<S: Clone + Eq + Hash>(
    v: &mut StateValueFunction<S>,
    traces: &mut EligibilityTraces<S>,
    state: &S,
    reward: f64,
    next_state: &S,
    done: bool,
    alpha: f64,
    gamma: f64,
    lambda: f64,
    replacing: bool,
) -> f64 {
    let next_v = if done { 0.0 } else { v.get(next_state) };
    let current_v = v.get(state);
    let td_error = reward + gamma * next_v - current_v;

    // Update eligibility for current state
    if replacing {
        traces.set(state, 1.0);
    } else {
        traces.update(state, 1.0);
    }

    // Update all states with traces
    for (s, e) in traces.all_traces() {
        let old_v = v.get(&s);
        v.set(s.clone(), old_v + alpha * td_error * e);
    }

    // Decay traces
    traces.decay(gamma * lambda);

    td_error
}

/// Sample a transition from a list based on probabilities.
pub fn sample_transition<S: crate::mdp::State, A: crate::mdp::Action>(
    transitions: &[crate::mdp::Transition<S, A>],
    rng: &mut impl rand::Rng,
) -> (S, f64) {
    if transitions.is_empty() {
        panic!("No transitions available");
    }

    let r: f64 = rng.gen();
    let mut cumulative = 0.0;
    for t in transitions {
        cumulative += t.probability;
        if r < cumulative {
            return (t.next_state, t.reward);
        }
    }

    let last = transitions.last().unwrap();
    (last.next_state, last.reward)
}

/// Run a full TD(0) episode on an MDP using the given policy.
pub fn td0_episode<M: MDP, R: rand::Rng>(
    mdp: &M,
    v: &mut StateValueFunction<M::S>,
    policy: &dyn Fn(M::S, &mut R) -> M::A,
    start_state: M::S,
    alpha: f64,
    max_steps: usize,
    rng: &mut R,
) -> Vec<f64>
where
    M::S: Hash,
{
    let gamma = mdp.discount_factor();
    let mut state = start_state;
    let mut td_errors = Vec::new();

    for _ in 0..max_steps {
        if mdp.is_terminal(state) {
            break;
        }

        let action = policy(state, rng);
        let transitions = mdp.transitions(state, action);
        let (next_state, reward) = sample_transition(&transitions, rng);

        let done = mdp.is_terminal(next_state);
        let td_error = td0_update(v, &state, reward, &next_state, done, alpha, gamma);
        td_errors.push(td_error);

        state = next_state;
    }

    td_errors
}

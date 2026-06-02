//! Q-learning: tabular Q-learning with ε-greedy exploration.

use crate::mdp::MDP;
use crate::td::sample_transition;
use crate::value::ActionValueFunction;
use std::hash::Hash;

/// Q-learning update rule.
///
/// Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
pub fn q_learning_update<S: Clone + Eq + Hash, A: Clone + Eq + Hash + Copy>(
    q: &mut ActionValueFunction<S, A>,
    state: &S,
    action: A,
    reward: f64,
    next_state: &S,
    done: bool,
    next_actions: &[A],
    alpha: f64,
    gamma: f64,
) -> f64 {
    let max_next_q = if done || next_actions.is_empty() {
        0.0
    } else {
        q.max_q(next_state, next_actions)
    };

    let current_q = q.get(state, &action);
    let td_error = reward + gamma * max_next_q - current_q;
    q.set(state.clone(), action, current_q + alpha * td_error);
    td_error
}

/// ε-greedy action selection.
pub fn epsilon_greedy<S: Clone + Eq + Hash, A: Copy + Eq + Hash>(
    q: &ActionValueFunction<S, A>,
    state: &S,
    actions: &[A],
    epsilon: f64,
    rng: &mut impl rand::Rng,
) -> A {
    if rng.gen::<f64>() < epsilon {
        actions[rng.gen_range(0..actions.len())]
    } else {
        q.greedy_action(state, actions).unwrap_or(actions[0])
    }
}

/// Run a Q-learning episode.
pub fn q_learning_episode<M: MDP, R: rand::Rng>(
    mdp: &M,
    q: &mut ActionValueFunction<M::S, M::A>,
    policy: &dyn Fn(M::S, &mut R) -> M::A,
    start_state: M::S,
    alpha: f64,
    max_steps: usize,
    rng: &mut R,
) -> (f64, usize)
where
    M::S: Hash,
    M::A: Copy + Hash,
{
    let gamma = mdp.discount_factor();
    let mut state = start_state;
    let mut total_reward = 0.0;
    let mut steps = 0;

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
        let done = mdp.is_terminal(next_state);

        let next_actions = mdp.actions(next_state);
        q_learning_update(
            q, &state, action, reward, &next_state, done, &next_actions, alpha, gamma,
        );

        total_reward += reward;
        steps += 1;
        state = next_state;
    }

    (total_reward, steps)
}

/// Extract a greedy policy from a Q-function.
pub fn extract_greedy_policy<M: MDP>(
    mdp: &M,
    q: &ActionValueFunction<M::S, M::A>,
) -> crate::mdp::DeterministicPolicy<M::S, M::A>
where
    M::S: Hash,
    M::A: Copy + Hash,
{
    let mut policy = crate::mdp::DeterministicPolicy::new();
    for state in mdp.states() {
        if mdp.is_terminal(state) {
            continue;
        }
        let actions = mdp.actions(state);
        if let Some(best) = q.greedy_action(&state, &actions) {
            policy.set(state, best);
        }
    }
    policy
}

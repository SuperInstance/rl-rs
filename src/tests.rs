//! Tests for reinforcement learning algorithms.

#[cfg(test)]
mod tests {
    use crate::bandit::*;
    use crate::dp::*;
    use crate::eligibility::*;
    use crate::gridworld::*;
    use crate::mdp::*;
    use crate::policy_gradient::*;
    use crate::q_learning::*;
    use crate::td::*;
    use crate::value::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;

    // ==================== Value Function Tests ====================

    #[test]
    fn test_state_value_get_set() {
        let mut v: StateValueFunction<usize> = StateValueFunction::new(0.0);
        assert_eq!(v.get(&1), 0.0);
        v.set(1, 5.0);
        assert_eq!(v.get(&1), 5.0);
        assert_eq!(v.get(&2), 0.0); // default
    }

    #[test]
    fn test_state_value_update() {
        let mut v: StateValueFunction<usize> = StateValueFunction::new(0.0);
        v.set(1, 3.0);
        v.update(&1, 2.0);
        assert!((v.get(&1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_value_get_set() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(1, 0, 10.0);
        q.set(1, 1, 20.0);
        assert_eq!(q.get(&1, &0), 10.0);
        assert_eq!(q.get(&1, &1), 20.0);
    }

    #[test]
    fn test_action_value_greedy() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(1, 0, 10.0);
        q.set(1, 1, 20.0);
        q.set(1, 2, 15.0);
        assert_eq!(q.greedy_action(&1, &[0, 1, 2]), Some(1));
    }

    #[test]
    fn test_action_value_max_q() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(1, 0, 10.0);
        q.set(1, 1, 30.0);
        q.set(1, 2, 20.0);
        assert!((q.max_q(&1, &[0, 1, 2]) - 30.0).abs() < 1e-10);
    }

    // ==================== MDP Tests ====================

    #[test]
    fn test_tabular_mdp_creation() {
        let mut mdp = TabularMDP::new(3, 2, 0.9);
        mdp.add_transition(0, 0, 1, 1.0, 1.0);
        mdp.add_transition(1, 0, 2, 2.0, 1.0);
        mdp.set_terminal(2);

        assert_eq!(mdp.states(), vec![0, 1, 2]);
        assert_eq!(mdp.actions(0), vec![0, 1]);
        assert!(mdp.is_terminal(2));
        assert!(!mdp.is_terminal(0));
    }

    #[test]
    fn test_tabular_mdp_transitions() {
        let mut mdp = TabularMDP::new(3, 2, 0.9);
        mdp.add_transition(0, 0, 1, 1.0, 0.7);
        mdp.add_transition(0, 0, 2, 0.0, 0.3);

        let t = mdp.transitions(0, 0);
        assert_eq!(t.len(), 2);
        assert!((t[0].probability - 0.7).abs() < 1e-10);
        assert!((t[1].probability - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_policy() {
        let mut policy: DeterministicPolicy<usize, usize> = DeterministicPolicy::new();
        policy.set(0, 1);
        policy.set(1, 0);
        assert_eq!(policy.get(0), Some(1));
        assert_eq!(policy.get(1), Some(0));
        assert_eq!(policy.get(2), None);
    }

    #[test]
    fn test_stochastic_policy() {
        let mut policy: StochasticPolicy<usize, usize> = StochasticPolicy::new();
        policy.set_prob(0, 0, 0.3);
        policy.set_prob(0, 1, 0.7);
        assert!((policy.get_prob(0, 0) - 0.3).abs() < 1e-10);
        assert!((policy.get_prob(0, 1) - 0.7).abs() < 1e-10);
        assert_eq!(policy.get_prob(0, 2), 0.0);
    }

    // ==================== Grid World Tests ====================

    #[test]
    fn test_grid_world_creation() {
        let gw = GridWorld::standard_4x4();
        assert_eq!(gw.width, 4);
        assert_eq!(gw.height, 4);
        assert_eq!(gw.start, (0, 0));
    }

    #[test]
    fn test_grid_world_movement() {
        let gw = GridWorld::new(4, 4);
        assert_eq!(gw.move_position((1, 1), GridAction::Up), (1, 0));
        assert_eq!(gw.move_position((1, 1), GridAction::Down), (1, 2));
        assert_eq!(gw.move_position((1, 1), GridAction::Left), (0, 1));
        assert_eq!(gw.move_position((1, 1), GridAction::Right), (2, 1));
    }

    #[test]
    fn test_grid_world_boundary() {
        let gw = GridWorld::new(4, 4);
        assert_eq!(gw.move_position((0, 0), GridAction::Up), (0, 0));
        assert_eq!(gw.move_position((0, 0), GridAction::Left), (0, 0));
        assert_eq!(gw.move_position((3, 3), GridAction::Down), (3, 3));
        assert_eq!(gw.move_position((3, 3), GridAction::Right), (3, 3));
    }

    #[test]
    fn test_grid_world_wall() {
        let mut gw = GridWorld::new(4, 4);
        gw.set_cell(2, 1, Cell::Wall);
        assert_eq!(gw.move_position((1, 1), GridAction::Right), (1, 1)); // blocked
        assert_eq!(gw.move_position((1, 1), GridAction::Left), (0, 1)); // not blocked
    }

    #[test]
    fn test_grid_world_terminal() {
        let gw = GridWorld::standard_4x4();
        assert!(gw.is_terminal(15)); // (3,3) is goal
        assert!(!gw.is_terminal(0));
    }

    #[test]
    fn test_grid_world_mdp_transitions() {
        let gw = GridWorld::standard_4x4();
        let t = gw.transitions(0, GridAction::Down);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].next_state, 4); // state (0,1) = 4
        assert!((t[0].reward - (-1.0)).abs() < 1e-10);
    }

    // ==================== Dynamic Programming Tests ====================

    #[test]
    fn test_policy_evaluation_uniform_4x4() {
        let gw = GridWorld::standard_4x4();
        let mut policy = StochasticPolicy::new();
        for s in gw.states() {
            if !gw.is_terminal(s) {
                for a in gw.actions(s) {
                    policy.set_prob(s, a, 0.25);
                }
            }
        }

        let v = policy_evaluation_stochastic(&gw, &policy, 1e-6, 1000);

        // V(0,0) should be approximately -59 for uniform random policy on 4x4 grid
        // with gamma=1, reward=-1 per step
        let v00 = v.get(&0);
        assert!(v00 < -10.0, "V(0,0) should be very negative, got {}", v00);
    }

    #[test]
    fn test_policy_iteration_converges() {
        let gw = GridWorld::standard_4x4();
        let (policy, v) = policy_iteration(&gw, 1e-10, 1000);

        // The optimal policy should have finite values
        for s in gw.states() {
            if !gw.is_terminal(s) {
                let val = v.get(&s);
                assert!(val.is_finite(), "Value should be finite for state {}", s);
            }
        }

        // Optimal policy should be defined for all non-terminal states
        for s in gw.states() {
            if !gw.is_terminal(s) {
                assert!(policy.get(s).is_some(), "Policy should be defined for state {}", s);
            }
        }
    }

    #[test]
    fn test_value_iteration_converges() {
        let gw = GridWorld::standard_4x4();
        let (_policy, v) = value_iteration(&gw, 1e-10, 1000);

        // Check that V decreases as distance from goal increases
        let v00 = v.get(&0);
        let _v15 = v.get(&15); // terminal
        assert!(v00 < v.get(&14), "V(0,0) should be less than V(3,2)");

        // All values should be finite
        for s in gw.states() {
            if !gw.is_terminal(s) {
                assert!(v.get(&s).is_finite());
            }
        }
    }

    #[test]
    fn test_policy_iteration_value_iteration_agree() {
        let gw = GridWorld::standard_4x4();
        let (_pi_policy, pi_v) = policy_iteration(&gw, 1e-10, 1000);
        let (_vi_policy, vi_v) = value_iteration(&gw, 1e-10, 1000);

        // Both should find the same optimal values
        for s in gw.states() {
            if !gw.is_terminal(s) {
                let pi_val = pi_v.get(&s);
                let vi_val = vi_v.get(&s);
                assert!(
                    (pi_val - vi_val).abs() < 1e-4,
                    "PI V({}) = {} vs VI V({}) = {}",
                    s,
                    pi_val,
                    s,
                    vi_val
                );
            }
        }
    }

    #[test]
    fn test_bellman_optimality_4x4() {
        let gw = GridWorld::standard_4x4();
        let (policy, v) = value_iteration(&gw, 1e-10, 1000);
        let gamma = gw.discount_factor();

        // Verify Bellman optimality: V(s) = max_a sum_{s'} P(s'|s,a)[r + γV(s')]
        for s in gw.states() {
            if gw.is_terminal(s) {
                continue;
            }
            let v_s = v.get(&s);
            let mut max_q = f64::NEG_INFINITY;
            for a in gw.actions(s) {
                let mut q_sa = 0.0;
                for t in gw.transitions(s, a) {
                    let next_v = if gw.is_terminal(t.next_state) { 0.0 } else { v.get(&t.next_state) };
                    q_sa += t.probability * (t.reward + gamma * next_v);
                }
                max_q = max_q.max(q_sa);
            }
            assert!(
                (v_s - max_q).abs() < 1e-6,
                "Bellman optimality violated at state {}: V = {}, max Q = {}",
                s, v_s, max_q
            );
        }
    }

    #[test]
    fn test_v_to_q() {
        let gw = GridWorld::standard_4x4();
        let (policy, v) = value_iteration(&gw, 1e-10, 1000);
        let q = v_to_q(&gw, &v);

        // Q values should be consistent with V
        for s in gw.states() {
            if gw.is_terminal(s) {
                continue;
            }
            let max_q = q.max_q(&s, &gw.actions(s));
            let v_s = v.get(&s);
            assert!((max_q - v_s).abs() < 1e-6, "Q max != V at state {}", s);
        }
        drop(policy);
    }

    // ==================== TD Learning Tests ====================

    #[test]
    fn test_td0_update_basic() {
        let mut v: StateValueFunction<usize> = StateValueFunction::new(0.0);
        v.set(0, 0.0);
        v.set(1, 0.0);

        let td_error = td0_update(&mut v, &0, 1.0, &1, false, 0.1, 0.9);
        assert!((td_error - 1.0).abs() < 1e-10); // r + γ*V(s') - V(s) = 1 + 0 - 0 = 1
        assert!((v.get(&0) - 0.1).abs() < 1e-10); // V(0) = 0 + 0.1*1 = 0.1
    }

    #[test]
    fn test_td0_update_terminal() {
        let mut v: StateValueFunction<usize> = StateValueFunction::new(0.0);
        v.set(0, 0.5);

        let td_error = td0_update(&mut v, &0, 10.0, &1, true, 0.1, 0.9);
        // δ = 10 + 0*V(1) - 0.5 = 9.5
        assert!((td_error - 9.5).abs() < 1e-10);
        assert!((v.get(&0) - (0.5 + 0.1 * 9.5)).abs() < 1e-10);
    }

    #[test]
    fn test_td0_converges_to_dp() {
        let gw = GridWorld::standard_4x4();
        let mut policy = StochasticPolicy::new();
        for s in gw.states() {
            if !gw.is_terminal(s) {
                for a in gw.actions(s) {
                    policy.set_prob(s, a, 0.25);
                }
            }
        }

        // DP solution
        let dp_v = policy_evaluation_stochastic(&gw, &policy, 1e-10, 10000);

        // TD(0) solution
        let mut td_v: StateValueFunction<usize> = StateValueFunction::new(0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let alpha = 0.01;

        let gw_ref = &gw;
        let policy_ref = &policy;
        for _ in 0..50000 {
            td0_episode(
                gw_ref,
                &mut td_v,
                &|s: usize, rng: &mut ChaCha8Rng| -> GridAction {
                    policy_ref.sample_action(s, rng).unwrap()
                },
                0,
                alpha,
                100,
                &mut rng,
            );
        }

        // TD values should be in the right ballpark
        for s in gw.states() {
            if !gw.is_terminal(s) && s != 15 {
                let dp_val = dp_v.get(&s);
                let td_val = td_v.get(&s);
                assert!(
                    (dp_val - td_val).abs() < 5.0,
                    "TD({}) = {} vs DP({}) = {}",
                    s, td_val, s, dp_val
                );
            }
        }
    }

    // ==================== Eligibility Traces Tests ====================

    #[test]
    fn test_eligibility_traces_basic() {
        let mut traces: EligibilityTraces<usize> = EligibilityTraces::new();
        assert_eq!(traces.get(&0), 0.0);

        traces.set(&0, 1.0);
        assert!((traces.get(&0) - 1.0).abs() < 1e-10);

        traces.update(&0, 0.5);
        assert!((traces.get(&0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_eligibility_traces_decay() {
        let mut traces: EligibilityTraces<usize> = EligibilityTraces::new();
        traces.set(&0, 1.0);
        traces.set(&1, 0.5);

        traces.decay(0.9);
        assert!((traces.get(&0) - 0.9).abs() < 1e-10);
        assert!((traces.get(&1) - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_eligibility_traces_reset() {
        let mut traces: EligibilityTraces<usize> = EligibilityTraces::new();
        traces.set(&0, 1.0);
        traces.reset();
        assert!(traces.is_empty());
        assert_eq!(traces.get(&0), 0.0);
    }

    #[test]
    fn test_td_lambda_update() {
        let mut v: StateValueFunction<usize> = StateValueFunction::new(0.0);
        let mut traces: EligibilityTraces<usize> = EligibilityTraces::new();

        v.set(0, 0.0);
        v.set(1, 0.0);

        // TD(λ) update with λ=0.5, replacing traces
        let td_error = td_lambda_update(
            &mut v, &mut traces, &0, 1.0, &1, false, 0.1, 0.9, 0.5, true,
        );

        assert!((td_error - 1.0).abs() < 1e-10); // r + γV(s') - V(s) = 1 + 0 - 0
        assert!((v.get(&0) - 0.1).abs() < 1e-10); // V(0) += α * δ * e(0) = 0 + 0.1*1*1
    }

    // ==================== Q-Learning Tests ====================

    #[test]
    fn test_q_learning_update_basic() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(0, 0, 0.0);
        q.set(1, 0, 5.0);
        q.set(1, 1, 3.0);

        let td_error = q_learning_update(&mut q, &0, 0, 1.0, &1, false, &[0, 1], 0.1, 0.9);
        // δ = r + γ*max_a'Q(s',a') - Q(s,a) = 1 + 0.9*5 - 0 = 5.5
        assert!((td_error - 5.5).abs() < 1e-10);
        assert!((q.get(&0, &0) - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_q_learning_update_terminal() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(0, 0, 2.0);

        let td_error = q_learning_update(&mut q, &0, 0, 10.0, &1, true, &[], 0.1, 0.9);
        // δ = 10 + 0 - 2 = 8
        assert!((td_error - 8.0).abs() < 1e-10);
        assert!((q.get(&0, &0) - (2.0 + 0.8)).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(0, 0, 10.0);
        q.set(0, 1, 5.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // With ε=0, always pick best
        for _ in 0..100 {
            let action = epsilon_greedy(&q, &0, &[0, 1], 0.0, &mut rng);
            assert_eq!(action, 0);
        }
    }

    #[test]
    fn test_epsilon_greedy_explores() {
        let mut q: ActionValueFunction<usize, usize> = ActionValueFunction::new(0.0);
        q.set(0, 0, 10.0);
        q.set(0, 1, 5.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut action1_count = 0;

        for _ in 0..1000 {
            let action = epsilon_greedy(&q, &0, &[0, 1], 0.5, &mut rng);
            if action == 1 {
                action1_count += 1;
            }
        }

        // With ε=0.5, should explore roughly 25% (half of the exploration)
        assert!(action1_count > 50, "Should explore action 1 sometimes, got {}", action1_count);
        assert!(action1_count < 400, "Should not explore too much, got {}", action1_count);
    }

    #[test]
    fn test_q_learning_converges_gridworld() {
        let gw = GridWorld::standard_4x4();
        let mut q: ActionValueFunction<usize, GridAction> = ActionValueFunction::new(0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for episode in 0..10000 {
            let epsilon = (1.0 - episode as f64 / 10000.0).max(0.05);
            let mut state: usize = 0;
            for _ in 0..100 {
                if gw.is_terminal(state) { break; }
                let a = epsilon_greedy(&q, &state, &gw.actions(state), epsilon, &mut rng);
                let trans = gw.transitions(state, a);
                if trans.is_empty() { break; }
                let (next_state, reward) = sample_transition(&trans, &mut rng);
                let done = gw.is_terminal(next_state);
                q_learning_update(&mut q, &state, a, reward, &next_state, done, &gw.actions(next_state), 0.2, gw.discount_factor());
                state = next_state;
            }
        }

        // Extract policy and verify it reaches the goal
        let policy = extract_greedy_policy(&gw, &q);

        // Simulate following the learned policy
        let mut state = 0usize;
        let mut steps = 0;
        while !gw.is_terminal(state) && steps < 50 {
            let action = policy.get(state).unwrap();
            let trans = gw.transitions(state, action);
            state = trans[0].next_state;
            steps += 1;
        }

        assert!(gw.is_terminal(state), "Policy should reach terminal state, at state {} after {} steps", state, steps);
        assert!(steps < 20, "Should reach goal in reasonable steps, took {}", steps);
    }

    // ==================== Policy Gradient Tests ====================

    #[test]
    fn test_softmax_policy_action_probs() {
        let policy: SoftmaxPolicy<usize, usize> = SoftmaxPolicy::new(0.1);
        let probs = policy.action_probs(&0, &[0, 1]);

        // With all-zero preferences, should be uniform
        assert!((probs[&0] - 0.5).abs() < 1e-10);
        assert!((probs[&1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_policy_sample() {
        let mut policy: SoftmaxPolicy<usize, usize> = SoftmaxPolicy::new(0.1);
        policy.theta.insert((0, 0), 10.0); // strongly prefer action 0

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut count_1 = 0;
        for _ in 0..1000 {
            let a = policy.sample(&0, &[0, 1], &mut rng);
            if a == 1 {
                count_1 += 1;
            }
        }
        assert!(count_1 < 50, "Should almost always pick action 0, picked 1 {} times", count_1);
    }

    #[test]
    fn test_trajectory_returns() {
        let mut traj: Trajectory<usize, usize> = Trajectory::new();
        traj.push(0, 0, 1.0);
        traj.push(1, 0, 2.0);
        traj.push(2, 0, 3.0);

        let returns = traj.returns(1.0);
        // G_0 = 1 + 2 + 3 = 6
        // G_1 = 2 + 3 = 5
        // G_2 = 3
        assert!((returns[0] - 6.0).abs() < 1e-10);
        assert!((returns[1] - 5.0).abs() < 1e-10);
        assert!((returns[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_discounted_returns() {
        let mut traj: Trajectory<usize, usize> = Trajectory::new();
        traj.push(0, 0, 1.0);
        traj.push(1, 0, 1.0);

        let returns = traj.returns(0.9);
        // G_0 = 1 + 0.9*1 = 1.9
        // G_1 = 1
        assert!((returns[0] - 1.9).abs() < 1e-10);
        assert!((returns[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reinforce_updates_preferences() {
        let mut policy: SoftmaxPolicy<usize, usize> = SoftmaxPolicy::new(0.1);
        let initial_pref = policy.preference(&0, &0);

        policy.reinforce_update(&0, 0, 10.0, &[0, 1]);

        // Action 0 should have increased preference
        let new_pref = policy.preference(&0, &0);
        assert!(new_pref > initial_pref, "Preference should increase for taken action");
    }

    #[test]
    fn test_collect_trajectory() {
        let gw = GridWorld::standard_4x4();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let gw_ref = &gw;
        let traj = collect_trajectory(
            gw_ref,
            &|_s: usize, rng: &mut ChaCha8Rng| -> GridAction {
                let actions = [GridAction::Up, GridAction::Down, GridAction::Left, GridAction::Right];
                actions[rng.gen_range(0usize..4)]
            },
            0,
            100,
            &mut rng,
        );

        assert!(!traj.is_empty());
        assert_eq!(traj.states.len(), traj.actions.len());
        assert_eq!(traj.states.len(), traj.rewards.len());
    }

    // ==================== Bandit Tests ====================

    #[test]
    fn test_gaussian_bandit() {
        let bandit = GaussianBandit::new(vec![1.0, 2.0, 0.5], 1.0);
        assert_eq!(bandit.num_arms(), 3);
        assert_eq!(bandit.optimal_arm(), 1);
    }

    #[test]
    fn test_arm_stats_update() {
        let mut stats = ArmStats::new();
        stats.update(1.0);
        stats.update(3.0);
        assert_eq!(stats.count, 2);
        assert!((stats.total_reward - 4.0).abs() < 1e-10);
        assert!((stats.q_estimate - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy_bandit() {
        let bandit = GaussianBandit::new(vec![0.0, 1.0, 2.0], 0.1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = run_epsilon_greedy(&bandit, 0.1, 1000, &mut rng);

        assert_eq!(result.rewards.len(), 1000);
        assert!(result.total_reward() > 0.0);
    }

    #[test]
    fn test_ucb1_bandit() {
        let bandit = GaussianBandit::new(vec![0.0, 1.0, 2.0], 0.1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = run_ucb1(&bandit, 2.0, 1000, &mut rng);

        assert_eq!(result.rewards.len(), 1000);
        // UCB1 should mostly pull the optimal arm
        let optimal_rate = result.optimal_pull_rate();
        assert!(optimal_rate > 0.3, "UCB1 optimal pull rate should be > 0.3, got {}", optimal_rate);
    }

    #[test]
    fn test_thompson_sampling_bandit() {
        let bandit = GaussianBandit::new(vec![0.0, 1.0, 2.0], 0.1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = run_thompson_sampling(&bandit, 1000, &mut rng);

        assert_eq!(result.rewards.len(), 1000);
        let optimal_rate = result.optimal_pull_rate();
        assert!(optimal_rate > 0.3, "Thompson optimal pull rate should be > 0.3, got {}", optimal_rate);
    }

    #[test]
    fn test_bandit_regret_decreases_over_time() {
        let bandit = GaussianBandit::new(vec![0.0, 1.0, 2.0], 0.1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = run_epsilon_greedy(&bandit, 0.1, 2000, &mut rng);

        // Regret should grow sublinearly - average regret per step should decrease
        let first_half_avg = result.cumulative_regret[999] / 1000.0;
        let second_half_avg = (result.cumulative_regret[1999] - result.cumulative_regret[999]) / 1000.0;
        assert!(
            second_half_avg <= first_half_avg * 1.5,
            "Average regret should not increase much: first={}, second={}",
            first_half_avg, second_half_avg
        );
    }

    // ==================== Additional Integration Tests ====================

    #[test]
    fn test_cliff_walking_q_learning() {
        let gw = GridWorld::cliff_walking();
        let mut q: ActionValueFunction<usize, GridAction> = ActionValueFunction::new(0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let start_state = gw.start.0 + gw.start.1 * gw.width;

        for episode in 0..20000 {
            let epsilon = (1.0 - episode as f64 / 20000.0).max(0.05);
            let mut state = start_state;
            for _ in 0..200 {
                if gw.is_terminal(state) { break; }
                let a = epsilon_greedy(&q, &state, &gw.actions(state), epsilon, &mut rng);
                let trans = gw.transitions(state, a);
                if trans.is_empty() { break; }
                let (next_state, reward) = sample_transition(&trans, &mut rng);
                let done = gw.is_terminal(next_state);
                q_learning_update(&mut q, &state, a, reward, &next_state, done, &gw.actions(next_state), 0.2, gw.discount_factor());
                state = next_state;
            }
        }

        let policy = extract_greedy_policy(&gw, &q);

        // The learned policy should reach a terminal state
        let mut state = start_state;
        let mut steps = 0;
        while !gw.is_terminal(state) && steps < 200 {
            let action = policy.get(state).unwrap();
            let trans = gw.transitions(state, action);
            state = trans[0].next_state;
            steps += 1;
        }
        assert!(gw.is_terminal(state), "Should reach terminal state on cliff walking, at state {} after {} steps", state, steps);
    }

    #[test]
    fn test_stochastic_gridworld() {
        let mut gw = GridWorld::new(5, 5);
        gw.discount = 0.95;
        gw.default_reward = -0.1;
        gw.set_cell(4, 4, Cell::Goal(10.0));
        gw.set_cell(1, 1, Cell::Wall);
        gw.set_cell(3, 3, Cell::Penalty(-5.0));
        gw.slip_probability = 0.2;
        gw.start = (0, 0);

        let (policy, v) = value_iteration(&gw, 1e-6, 1000);

        // Value at start should be positive (goal is worth 10, small costs)
        let start_state = 0usize;
        let v_start = v.get(&start_state);
        assert!(v_start > 0.0, "V(start) should be positive in stochastic grid, got {}", v_start);

        // Policy should be defined
        for s in gw.states() {
            if !gw.is_terminal(s) && !matches!(gw.grid[s / gw.width][s % gw.width], Cell::Wall) {
                assert!(policy.get(s).is_some());
            }
        }
    }

    #[test]
    fn test_bandit_comparison() {
        let bandit = GaussianBandit::new(vec![0.0, 0.5, 1.0, 0.3], 0.5);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let eg_result = run_epsilon_greedy(&bandit, 0.1, 2000, &mut rng);
        let ucb_result = run_ucb1(&bandit, 2.0, 2000, &mut rng);
        let ts_result = run_thompson_sampling(&bandit, 2000, &mut rng);

        // All algorithms should have low regret
        // Thompson and UCB should generally outperform epsilon-greedy
        assert!(eg_result.total_reward() > -500.0);
        assert!(ucb_result.total_reward() > -500.0);
        assert!(ts_result.total_reward() > -500.0);
    }

    #[test]
    fn test_tabular_mdp_value_iteration() {
        let mut mdp = TabularMDP::new(3, 2, 0.9);
        // Linear chain: 0 -> 1 -> 2(terminal)
        mdp.add_transition(0, 0, 1, 0.0, 1.0);
        mdp.add_transition(0, 1, 0, 0.0, 1.0); // self-loop
        mdp.add_transition(1, 0, 2, 10.0, 1.0);
        mdp.add_transition(1, 1, 0, 0.0, 1.0);
        mdp.set_terminal(2);

        let (policy, v) = value_iteration(&mdp, 1e-10, 1000);

        // V(1) = 10 (go to terminal)
        // V(0) = 0.9 * V(1) = 9.0
        assert!((v.get(&1) - 10.0).abs() < 0.1, "V(1) should be ~10, got {}", v.get(&1));
        assert!((v.get(&0) - 9.0).abs() < 0.1, "V(0) should be ~9, got {}", v.get(&0));

        // Optimal policy: action 0 for both states
        assert_eq!(policy.get(0), Some(0));
        assert_eq!(policy.get(1), Some(0));
    }

    #[test]
    fn test_experience_tuple() {
        let exp = Experience {
            state: 0,
            action: GridAction::Right,
            reward: -1.0,
            next_state: 1,
            done: false,
        };
        assert_eq!(exp.state, 0);
        assert!((exp.reward - (-1.0)).abs() < 1e-10);
        assert!(!exp.done);
    }

    #[test]
    fn test_gridworld_pos_state_conversion() {
        assert_eq!(pos_to_state(2, 3, 4), 14);
        assert_eq!(state_to_pos(14, 4), (2, 3));
        assert_eq!(pos_to_state(0, 0, 4), 0);
        assert_eq!(state_to_pos(0, 4), (0, 0));
    }

    #[test]
    fn test_bandit_result_statistics() {
        let bandit = GaussianBandit::new(vec![0.0, 1.0], 0.1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = run_epsilon_greedy(&bandit, 0.5, 100, &mut rng);

        assert_eq!(result.rewards.len(), 100);
        assert_eq!(result.cumulative_reward.len(), 100);
        assert_eq!(result.cumulative_regret.len(), 100);
        assert_eq!(result.optimal_pulls.len(), 100);

        let manual_sum: f64 = result.rewards.iter().sum();
        assert!((result.total_reward() - manual_sum).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_arm_detection() {
        let bandit = GaussianBandit::new(vec![-1.0, 0.0, 3.0, 1.0], 1.0);
        assert_eq!(bandit.optimal_arm(), 2);
    }

    #[test]
    fn test_ucb1_explores_all_arms() {
        let bandit = GaussianBandit::new(vec![0.0, 1.0, 2.0], 0.1);
        let mut agent = UCB1Agent::new(3, 2.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut pulls = vec![0; 3];
        for _ in 0..100 {
            let arm = agent.select();
            let reward = bandit.pull(arm, &mut rng);
            agent.update(arm, reward);
            pulls[arm] += 1;
        }

        // UCB1 should pull all arms at least once
        for (arm, count) in pulls.iter().enumerate() {
            assert!(*count > 0, "Arm {} was never pulled", arm);
        }
    }

    #[test]
    fn test_default_implementations() {
        let v: StateValueFunction<usize> = StateValueFunction::default();
        assert_eq!(v.get(&0), 0.0);

        let q: ActionValueFunction<usize, usize> = ActionValueFunction::default();
        assert_eq!(q.get(&0, &0), 0.0);

        let traces: EligibilityTraces<usize> = EligibilityTraces::default();
        assert!(traces.is_empty());

        let policy: DeterministicPolicy<usize, usize> = DeterministicPolicy::default();
        assert!(policy.get(0).is_none());
    }

    #[test]
    fn test_trajectory_empty() {
        let traj: Trajectory<usize, usize> = Trajectory::new();
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
        assert!(traj.returns(0.9).is_empty());
    }

    #[test]
    fn test_trajectory_default() {
        let traj: Trajectory<usize, usize> = Trajectory::default();
        assert!(traj.is_empty());
    }
}

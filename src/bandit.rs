//! Multi-armed bandits: ε-greedy, UCB, Thompson sampling.

use rand::Rng;
use rand_distr::{Gamma, Normal};
use serde::{Deserialize, Serialize};

/// Bandit arm statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStats {
    pub count: usize,
    pub total_reward: f64,
    pub q_estimate: f64,
}

impl ArmStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_reward: 0.0,
            q_estimate: 0.0,
        }
    }

    pub fn update(&mut self, reward: f64) {
        self.count += 1;
        self.total_reward += reward;
        self.q_estimate += (reward - self.q_estimate) / self.count as f64;
    }

    pub fn average_reward(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_reward / self.count as f64
        }
    }
}

impl Default for ArmStats {
    fn default() -> Self {
        Self::new()
    }
}

/// A multi-armed bandit environment.
pub trait Bandit: Send + Sync {
    /// Pull an arm and receive a reward.
    fn pull(&self, arm: usize, rng: &mut impl Rng) -> f64;

    /// Number of arms.
    fn num_arms(&self) -> usize;

    /// True mean reward of each arm (for evaluation).
    fn true_means(&self) -> Vec<f64>;

    /// Optimal arm index.
    fn optimal_arm(&self) -> usize {
        let means = self.true_means();
        means
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// A simple Gaussian bandit: each arm has a fixed mean reward with Gaussian noise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianBandit {
    pub means: Vec<f64>,
    pub std_dev: f64,
}

impl GaussianBandit {
    pub fn new(means: Vec<f64>, std_dev: f64) -> Self {
        Self { means, std_dev }
    }
}

impl Bandit for GaussianBandit {
    fn pull(&self, arm: usize, rng: &mut impl Rng) -> f64 {
        let dist = Normal::new(self.means[arm], self.std_dev).unwrap();
        rng.sample(dist)
    }

    fn num_arms(&self) -> usize {
        self.means.len()
    }

    fn true_means(&self) -> Vec<f64> {
        self.means.clone()
    }
}

/// ε-greedy bandit agent.
#[derive(Debug, Clone)]
pub struct EpsilonGreedyAgent {
    pub epsilon: f64,
    pub stats: Vec<ArmStats>,
}

impl EpsilonGreedyAgent {
    pub fn new(num_arms: usize, epsilon: f64) -> Self {
        Self {
            epsilon,
            stats: vec![ArmStats::new(); num_arms],
        }
    }

    /// Select an arm.
    pub fn select(&self, rng: &mut impl Rng) -> usize {
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.stats.len())
        } else {
            self.stats
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let qa = if a.count == 0 { f64::INFINITY } else { a.q_estimate };
                    let qb = if b.count == 0 { f64::INFINITY } else { b.q_estimate };
                    qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Update after receiving a reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.stats[arm].update(reward);
    }
}

/// UCB1 (Upper Confidence Bound) bandit agent.
#[derive(Debug, Clone)]
pub struct UCB1Agent {
    pub c: f64,
    pub stats: Vec<ArmStats>,
    pub total_pulls: usize,
}

impl UCB1Agent {
    pub fn new(num_arms: usize, c: f64) -> Self {
        Self {
            c,
            stats: vec![ArmStats::new(); num_arms],
            total_pulls: 0,
        }
    }

    /// Select an arm using UCB1 formula.
    pub fn select(&mut self) -> usize {
        for (i, s) in self.stats.iter().enumerate() {
            if s.count == 0 {
                return i;
            }
        }

        let total = self.total_pulls as f64;
        self.stats
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let exploration = self.c * (total.ln() / s.count as f64).sqrt();
                (i, s.q_estimate + exploration)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update after receiving a reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.stats[arm].update(reward);
        self.total_pulls += 1;
    }
}

/// Thompson Sampling agent for Gaussian bandits.
#[derive(Debug, Clone)]
pub struct ThompsonSamplingAgent {
    pub mu: Vec<f64>,
    pub kappa: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl ThompsonSamplingAgent {
    pub fn new(num_arms: usize) -> Self {
        Self {
            mu: vec![0.0; num_arms],
            kappa: vec![1.0; num_arms],
            alpha: vec![1.0; num_arms],
            beta: vec![1.0; num_arms],
        }
    }

    /// Sample from posterior for each arm and select the best.
    pub fn select(&self, rng: &mut impl Rng) -> usize {
        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;

        for i in 0..self.mu.len() {
            let gamma_dist = Gamma::new(self.alpha[i], 1.0 / self.beta[i]).unwrap();
            let precision: f64 = rng.sample(gamma_dist);
            let std_dev = (1.0 / (self.kappa[i] * precision.max(1e-10))).sqrt();

            let normal = Normal::new(self.mu[i], std_dev).unwrap();
            let sample: f64 = rng.sample(normal);

            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }

        best_arm
    }

    /// Update posterior with observed reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        let old_mu = self.mu[arm];
        let old_kappa = self.kappa[arm];

        self.mu[arm] = (old_kappa * old_mu + reward) / (old_kappa + 1.0);
        self.kappa[arm] += 1.0;
        self.alpha[arm] += 0.5;
        self.beta[arm] += 0.5 * old_kappa * (reward - old_mu).powi(2) / (old_kappa + 1.0);
    }
}

/// Run a bandit experiment and collect statistics.
pub struct BanditResult {
    pub rewards: Vec<f64>,
    pub optimal_pulls: Vec<bool>,
    pub cumulative_reward: Vec<f64>,
    pub cumulative_regret: Vec<f64>,
}

impl BanditResult {
    pub fn total_reward(&self) -> f64 {
        self.rewards.iter().sum()
    }

    pub fn total_regret(&self) -> f64 {
        self.cumulative_regret.last().copied().unwrap_or(0.0)
    }

    pub fn optimal_pull_rate(&self) -> f64 {
        if self.optimal_pulls.is_empty() {
            0.0
        } else {
            self.optimal_pulls.iter().filter(|&&x| x).count() as f64
                / self.optimal_pulls.len() as f64
        }
    }
}

/// Run an ε-greedy bandit experiment.
pub fn run_epsilon_greedy<B: Bandit>(
    bandit: &B,
    epsilon: f64,
    num_steps: usize,
    rng: &mut impl Rng,
) -> BanditResult {
    let mut agent = EpsilonGreedyAgent::new(bandit.num_arms(), epsilon);
    let optimal = bandit.optimal_arm();
    let means = bandit.true_means();

    let mut rewards = Vec::with_capacity(num_steps);
    let mut optimal_pulls = Vec::with_capacity(num_steps);
    let mut cumulative_reward = Vec::with_capacity(num_steps);
    let mut cumulative_regret = Vec::with_capacity(num_steps);

    let mut total_r = 0.0;
    let mut total_regret = 0.0;

    for _ in 0..num_steps {
        let arm = agent.select(rng);
        let reward = bandit.pull(arm, rng);
        agent.update(arm, reward);

        total_r += reward;
        total_regret += means[optimal] - means[arm];

        rewards.push(reward);
        optimal_pulls.push(arm == optimal);
        cumulative_reward.push(total_r);
        cumulative_regret.push(total_regret);
    }

    BanditResult {
        rewards,
        optimal_pulls,
        cumulative_reward,
        cumulative_regret,
    }
}

/// Run a UCB1 bandit experiment.
pub fn run_ucb1<B: Bandit>(
    bandit: &B,
    c: f64,
    num_steps: usize,
    rng: &mut impl Rng,
) -> BanditResult {
    let mut agent = UCB1Agent::new(bandit.num_arms(), c);
    let optimal = bandit.optimal_arm();
    let means = bandit.true_means();

    let mut rewards = Vec::with_capacity(num_steps);
    let mut optimal_pulls = Vec::with_capacity(num_steps);
    let mut cumulative_reward = Vec::with_capacity(num_steps);
    let mut cumulative_regret = Vec::with_capacity(num_steps);

    let mut total_r = 0.0;
    let mut total_regret = 0.0;

    for _ in 0..num_steps {
        let arm = agent.select();
        let reward = bandit.pull(arm, rng);
        agent.update(arm, reward);

        total_r += reward;
        total_regret += means[optimal] - means[arm];

        rewards.push(reward);
        optimal_pulls.push(arm == optimal);
        cumulative_reward.push(total_r);
        cumulative_regret.push(total_regret);
    }

    BanditResult {
        rewards,
        optimal_pulls,
        cumulative_reward,
        cumulative_regret,
    }
}

/// Run a Thompson Sampling bandit experiment.
pub fn run_thompson_sampling<B: Bandit>(
    bandit: &B,
    num_steps: usize,
    rng: &mut impl Rng,
) -> BanditResult {
    let mut agent = ThompsonSamplingAgent::new(bandit.num_arms());
    let optimal = bandit.optimal_arm();
    let means = bandit.true_means();

    let mut rewards = Vec::with_capacity(num_steps);
    let mut optimal_pulls = Vec::with_capacity(num_steps);
    let mut cumulative_reward = Vec::with_capacity(num_steps);
    let mut cumulative_regret = Vec::with_capacity(num_steps);

    let mut total_r = 0.0;
    let mut total_regret = 0.0;

    for _ in 0..num_steps {
        let arm = agent.select(rng);
        let reward = bandit.pull(arm, rng);
        agent.update(arm, reward);

        total_r += reward;
        total_regret += means[optimal] - means[arm];

        rewards.push(reward);
        optimal_pulls.push(arm == optimal);
        cumulative_reward.push(total_r);
        cumulative_regret.push(total_regret);
    }

    BanditResult {
        rewards,
        optimal_pulls,
        cumulative_reward,
        cumulative_regret,
    }
}

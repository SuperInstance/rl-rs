//! Grid world environments for testing RL algorithms.

use crate::mdp::{MDP, Transition};
use serde::{Deserialize, Serialize};

/// Actions in a grid world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GridAction {
    Up,
    Down,
    Left,
    Right,
}

/// A cell in the grid world.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Cell {
    Empty,
    Wall,
    Goal(f64),  // Terminal with reward
    Penalty(f64), // Terminal with penalty
}

/// A position in the grid.
pub type Position = (usize, usize);

/// Grid world environment implementing MDP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Vec<Cell>>,
    pub start: Position,
    pub default_reward: f64,
    pub discount: f64,
    pub slip_probability: f64, // Probability of sliding to a random adjacent cell
}

impl GridWorld {
    /// Create a new grid world of given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            grid: vec![vec![Cell::Empty; width]; height],
            start: (0, 0),
            default_reward: -1.0,
            discount: 0.99,
            slip_probability: 0.0,
        }
    }

    /// Create a standard 4x4 grid world.
    pub fn standard_4x4() -> Self {
        let mut gw = Self::new(4, 4);
        gw.set_cell(3, 3, Cell::Goal(0.0));
        gw.start = (0, 0);
        gw.default_reward = -1.0;
        gw.discount = 1.0;
        gw
    }

    /// Create a cliff walking environment.
    pub fn cliff_walking() -> Self {
        let mut gw = Self::new(12, 4);
        // Bottom row is cliff (except start and goal)
        for x in 1..11 {
            gw.set_cell(x, 3, Cell::Penalty(-100.0));
        }
        gw.set_cell(11, 3, Cell::Goal(0.0));
        gw.start = (0, 3);
        gw.default_reward = -1.0;
        gw.discount = 1.0;
        gw
    }

    pub fn set_cell(&mut self, x: usize, y: usize, cell: Cell) {
        self.grid[y][x] = cell;
    }

    pub fn get_cell(&self, x: usize, y: usize) -> Cell {
        self.grid[y][x]
    }

    /// Move in a direction, returning new position (clamped to bounds).
    pub fn move_position(&self, pos: Position, action: GridAction) -> Position {
        let (x, y) = pos;
        let (nx, ny) = match action {
            GridAction::Up => (x, y.saturating_sub(1)),
            GridAction::Down => (x, (y + 1).min(self.height - 1)),
            GridAction::Left => (x.saturating_sub(1), y),
            GridAction::Right => ((x + 1).min(self.width - 1), y),
        };

        // Check for wall
        if matches!(self.grid[ny][nx], Cell::Wall) {
            pos // Stay in place
        } else {
            (nx, ny)
        }
    }

    fn is_terminal_pos(&self, pos: Position) -> bool {
        matches!(
            self.grid[pos.1][pos.0],
            Cell::Goal(_) | Cell::Penalty(_)
        )
    }

    fn reward_at(&self, pos: Position) -> f64 {
        match self.grid[pos.1][pos.0] {
            Cell::Goal(r) => r,
            Cell::Penalty(r) => r,
            _ => self.default_reward,
        }
    }
}

impl MDP for GridWorld {
    type S = usize;
    type A = GridAction;

    fn states(&self) -> Vec<Self::S> {
        (0..self.width * self.height).collect()
    }

    fn actions(&self, _state: Self::S) -> Vec<Self::A> {
        vec![
            GridAction::Up,
            GridAction::Down,
            GridAction::Left,
            GridAction::Right,
        ]
    }

    fn transitions(&self, state: Self::S, action: Self::A) -> Vec<Transition<Self::S, Self::A>> {
        let x = state % self.width;
        let y = state / self.width;
        let pos = (x, y);

        if self.is_terminal_pos(pos) {
            return vec![];
        }

        let mut result = Vec::new();

        if self.slip_probability > 0.0 {
            // Stochastic transitions
            let intended_pos = self.move_position(pos, action);
            let intended_reward = self.reward_at(intended_pos);
            let intended_next = intended_pos.0 + intended_pos.1 * self.width;
            let _intended_terminal = self.is_terminal_pos(intended_pos);

            // Perpendicular directions
            let perp: Vec<GridAction> = match action {
                GridAction::Up | GridAction::Down => {
                    vec![GridAction::Left, GridAction::Right]
                }
                GridAction::Left | GridAction::Right => {
                    vec![GridAction::Up, GridAction::Down]
                }
            };

            let prob_intended = 1.0 - self.slip_probability;
            let prob_slip = self.slip_probability / 2.0;

            // Intended direction
            result.push(Transition::new(
                intended_next,
                intended_reward,
                prob_intended,
            ));

            // Slip directions
            for &slip_action in &perp {
                let slip_pos = self.move_position(pos, slip_action);
                let slip_reward = self.reward_at(slip_pos);
                let slip_next = slip_pos.0 + slip_pos.1 * self.width;
                result.push(Transition::new(slip_next, slip_reward, prob_slip));
            }
        } else {
            // Deterministic
            let next_pos = self.move_position(pos, action);
            let reward = self.reward_at(next_pos);
            let next_state = next_pos.0 + next_pos.1 * self.width;

            result.push(Transition::new(next_state, reward, 1.0));
        }

        result
    }

    fn discount_factor(&self) -> f64 {
        self.discount
    }

    fn is_terminal(&self, state: Self::S) -> bool {
        let x = state % self.width;
        let y = state / self.width;
        self.is_terminal_pos((x, y))
    }

    fn all_actions(&self) -> Vec<Self::A> {
        vec![
            GridAction::Up,
            GridAction::Down,
            GridAction::Left,
            GridAction::Right,
        ]
    }
}

/// Convert a position to a flat state index.
pub fn pos_to_state(x: usize, y: usize, width: usize) -> usize {
    x + y * width
}

/// Convert a flat state index back to a position.
pub fn state_to_pos(state: usize, width: usize) -> Position {
    (state % width, state / width)
}

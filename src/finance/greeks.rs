use super::gbm::{gbm_paths, GbmParams};
use super::heston::{heston_paths_with_scheme, HestonParams, HestonScheme};
use super::jump_diffusion::{merton_paths, MertonParams};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum Payoff {
    Call { strike: f64 },
    Put { strike: f64 },
}

impl Payoff {
    pub fn eval(&self, s_t: f64) -> f64 {
        match self {
            Payoff::Call { strike } => (s_t - strike).max(0.0),
            Payoff::Put { strike } => (strike - s_t).max(0.0),
        }
    }

    pub fn eval_batch(&self, terminals: &[f64]) -> Vec<f64> {
        terminals.iter().map(|&s| self.eval(s)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BumpParam {
    S0,
    Sigma,
    T,
    Mu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Greek {
    Delta,
    Gamma,
    Vega,
    Theta,
    Rho,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BumpDir {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScenarioKey {
    Base,
    Bumped(BumpParam, BumpDir),
}

#[derive(Debug, Clone)]
pub enum ModelSpec {
    Gbm {
        s0: f64,
        mu: f64,
        sigma: f64,
        t: f64,
        steps: usize,
        n_paths: usize,
    },
    Heston {
        s0: f64,
        v0: f64,
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        t: f64,
        steps: usize,
        n_paths: usize,
        scheme: HestonScheme,
    },
    Merton {
        s0: f64,
        mu: f64,
        sigma: f64,
        lambda: f64,
        mu_j: f64,
        sigma_j: f64,
        t: f64,
        steps: usize,
        n_paths: usize,
    },
}

impl ModelSpec {
    pub fn terminal_prices(&self, seed: u128) -> Vec<f64> {
        let paths = match self {
            ModelSpec::Gbm {
                s0,
                mu,
                sigma,
                t,
                steps,
                n_paths,
            } => {
                let params = GbmParams {
                    s0: *s0,
                    mu: *mu,
                    sigma: *sigma,
                    t: *t,
                    steps: *steps,
                    n_paths: *n_paths,
                    antithetic: false,
                };
                gbm_paths(&params, seed)
            }
            ModelSpec::Heston {
                s0,
                v0,
                mu,
                kappa,
                theta,
                xi,
                rho,
                t,
                steps,
                n_paths,
                scheme,
            } => {
                let params = HestonParams {
                    s0: *s0,
                    v0: *v0,
                    mu: *mu,
                    kappa: *kappa,
                    theta: *theta,
                    xi: *xi,
                    rho: *rho,
                    t: *t,
                    steps: *steps,
                    n_paths: *n_paths,
                };
                heston_paths_with_scheme(&params, seed, *scheme)
            }
            ModelSpec::Merton {
                s0,
                mu,
                sigma,
                lambda,
                mu_j,
                sigma_j,
                t,
                steps,
                n_paths,
            } => {
                let params = MertonParams {
                    s0: *s0,
                    mu: *mu,
                    sigma: *sigma,
                    lambda: *lambda,
                    mu_j: *mu_j,
                    sigma_j: *sigma_j,
                    t: *t,
                    steps: *steps,
                    n_paths: *n_paths,
                };
                merton_paths(&params, seed)
            }
        };
        let last_col = paths.ncols() - 1;
        paths.column(last_col).to_vec()
    }

    pub fn with_bumped(&self, param: BumpParam, delta: f64) -> Self {
        let mut cloned = self.clone();
        match (&mut cloned, param) {
            (ModelSpec::Gbm { ref mut s0, .. }, BumpParam::S0) => *s0 += delta,
            (ModelSpec::Gbm { ref mut sigma, .. }, BumpParam::Sigma) => *sigma += delta,
            (ModelSpec::Gbm { ref mut t, .. }, BumpParam::T) => *t += delta,
            (ModelSpec::Gbm { ref mut mu, .. }, BumpParam::Mu) => *mu += delta,

            (ModelSpec::Heston { ref mut s0, .. }, BumpParam::S0) => *s0 += delta,
            (ModelSpec::Heston { ref mut v0, .. }, BumpParam::Sigma) => *v0 += delta,
            (ModelSpec::Heston { ref mut t, .. }, BumpParam::T) => *t += delta,
            (ModelSpec::Heston { ref mut mu, .. }, BumpParam::Mu) => *mu += delta,

            (ModelSpec::Merton { ref mut s0, .. }, BumpParam::S0) => *s0 += delta,
            (ModelSpec::Merton { ref mut sigma, .. }, BumpParam::Sigma) => *sigma += delta,
            (ModelSpec::Merton { ref mut t, .. }, BumpParam::T) => *t += delta,
            (ModelSpec::Merton { ref mut mu, .. }, BumpParam::Mu) => *mu += delta,
        }
        cloned
    }

    pub fn risk_free_rate(&self) -> f64 {
        match self {
            ModelSpec::Gbm { mu, .. }
            | ModelSpec::Heston { mu, .. }
            | ModelSpec::Merton { mu, .. } => *mu,
        }
    }

    pub fn maturity(&self) -> f64 {
        match self {
            ModelSpec::Gbm { t, .. }
            | ModelSpec::Heston { t, .. }
            | ModelSpec::Merton { t, .. } => *t,
        }
    }

    pub fn param_value(&self, param: BumpParam) -> f64 {
        match (self, param) {
            (ModelSpec::Gbm { s0, .. }, BumpParam::S0)
            | (ModelSpec::Heston { s0, .. }, BumpParam::S0)
            | (ModelSpec::Merton { s0, .. }, BumpParam::S0) => *s0,

            (ModelSpec::Gbm { sigma, .. }, BumpParam::Sigma)
            | (ModelSpec::Merton { sigma, .. }, BumpParam::Sigma) => *sigma,
            (ModelSpec::Heston { v0, .. }, BumpParam::Sigma) => *v0,

            (ModelSpec::Gbm { t, .. }, BumpParam::T)
            | (ModelSpec::Heston { t, .. }, BumpParam::T)
            | (ModelSpec::Merton { t, .. }, BumpParam::T) => *t,

            (ModelSpec::Gbm { mu, .. }, BumpParam::Mu)
            | (ModelSpec::Heston { mu, .. }, BumpParam::Mu)
            | (ModelSpec::Merton { mu, .. }, BumpParam::Mu) => *mu,
        }
    }
}

pub fn compute_bump(param_value: f64, bump_size: f64) -> f64 {
    (param_value.abs() * bump_size).max(1e-8)
}

pub fn required_scenarios(greeks: &[Greek]) -> Vec<ScenarioKey> {
    let mut set = Vec::<ScenarioKey>::new();
    let mut add = |key: ScenarioKey| {
        if !set.contains(&key) {
            set.push(key);
        }
    };

    for &g in greeks {
        match g {
            Greek::Delta => {
                add(ScenarioKey::Bumped(BumpParam::S0, BumpDir::Up));
                add(ScenarioKey::Bumped(BumpParam::S0, BumpDir::Down));
            }
            Greek::Gamma => {
                add(ScenarioKey::Bumped(BumpParam::S0, BumpDir::Up));
                add(ScenarioKey::Base);
                add(ScenarioKey::Bumped(BumpParam::S0, BumpDir::Down));
            }
            Greek::Vega => {
                add(ScenarioKey::Bumped(BumpParam::Sigma, BumpDir::Up));
                add(ScenarioKey::Bumped(BumpParam::Sigma, BumpDir::Down));
            }
            Greek::Theta => {
                add(ScenarioKey::Bumped(BumpParam::T, BumpDir::Down));
                add(ScenarioKey::Base);
            }
            Greek::Rho => {
                add(ScenarioKey::Bumped(BumpParam::Mu, BumpDir::Up));
                add(ScenarioKey::Bumped(BumpParam::Mu, BumpDir::Down));
            }
        }
    }
    set
}

pub fn compute_greeks_from_prices(
    greeks: &[Greek],
    prices: &HashMap<ScenarioKey, f64>,
    bumps: &HashMap<BumpParam, f64>,
) -> Vec<(Greek, f64)> {
    greeks
        .iter()
        .map(|&g| {
            let value = match g {
                Greek::Delta => {
                    let h = bumps[&BumpParam::S0];
                    let v_up = prices[&ScenarioKey::Bumped(BumpParam::S0, BumpDir::Up)];
                    let v_down = prices[&ScenarioKey::Bumped(BumpParam::S0, BumpDir::Down)];
                    (v_up - v_down) / (2.0 * h)
                }
                Greek::Gamma => {
                    let h = bumps[&BumpParam::S0];
                    let v_up = prices[&ScenarioKey::Bumped(BumpParam::S0, BumpDir::Up)];
                    let v_base = prices[&ScenarioKey::Base];
                    let v_down = prices[&ScenarioKey::Bumped(BumpParam::S0, BumpDir::Down)];
                    (v_up - 2.0 * v_base + v_down) / (h * h)
                }
                Greek::Vega => {
                    let h = bumps[&BumpParam::Sigma];
                    let v_up = prices[&ScenarioKey::Bumped(BumpParam::Sigma, BumpDir::Up)];
                    let v_down = prices[&ScenarioKey::Bumped(BumpParam::Sigma, BumpDir::Down)];
                    (v_up - v_down) / (2.0 * h)
                }
                Greek::Theta => {
                    let h = bumps[&BumpParam::T];
                    let v_short = prices[&ScenarioKey::Bumped(BumpParam::T, BumpDir::Down)];
                    let v_base = prices[&ScenarioKey::Base];
                    (v_short - v_base) / h
                }
                Greek::Rho => {
                    let h = bumps[&BumpParam::Mu];
                    let v_up = prices[&ScenarioKey::Bumped(BumpParam::Mu, BumpDir::Up)];
                    let v_down = prices[&ScenarioKey::Bumped(BumpParam::Mu, BumpDir::Down)];
                    (v_up - v_down) / (2.0 * h)
                }
            };
            (g, value)
        })
        .collect()
}

fn mc_price(model: &ModelSpec, payoff: &Payoff, seed: u128) -> f64 {
    let terminals = model.terminal_prices(seed);
    let payoffs = payoff.eval_batch(&terminals);
    let r = model.risk_free_rate();
    let t = model.maturity();
    let discount = (-r * t).exp();
    discount * payoffs.iter().sum::<f64>() / payoffs.len() as f64
}

pub fn greeks_fd_core(
    model: &ModelSpec,
    payoff: &Payoff,
    greeks: &[Greek],
    bump_size: f64,
    seed: u128,
) -> Vec<(Greek, f64)> {
    let scenarios = required_scenarios(greeks);

    let mut bumps = HashMap::new();
    for &key in &scenarios {
        if let ScenarioKey::Bumped(param, _) = key {
            bumps
                .entry(param)
                .or_insert_with(|| compute_bump(model.param_value(param), bump_size));
        }
    }

    let mut prices = HashMap::new();
    for &key in &scenarios {
        let price = match key {
            ScenarioKey::Base => mc_price(model, payoff, seed),
            ScenarioKey::Bumped(param, dir) => {
                let h = bumps[&param];
                let delta = match dir {
                    BumpDir::Up => h,
                    BumpDir::Down => -h,
                };
                let bumped = model.with_bumped(param, delta);
                mc_price(&bumped, payoff, seed)
            }
        };
        prices.insert(key, price);
    }

    compute_greeks_from_prices(greeks, &prices, &bumps)
}

pub fn greeks_pathwise_core(
    s0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    strike: f64,
    is_call: bool,
    n_paths: usize,
    steps: usize,
    greeks: &[Greek],
    seed: u128,
) -> Vec<(Greek, f64)> {
    let model = ModelSpec::Gbm {
        s0,
        mu: r,
        sigma,
        t,
        steps,
        n_paths,
    };
    let terminals = model.terminal_prices(seed);
    let discount = (-r * t).exp();
    let n = terminals.len() as f64;

    greeks
        .iter()
        .map(|&g| {
            let value = match g {
                Greek::Delta => {
                    let sum: f64 = terminals
                        .iter()
                        .map(|&s_t| {
                            let in_money = if is_call {
                                s_t > strike
                            } else {
                                s_t < strike
                            };
                            if in_money {
                                let sign = if is_call { 1.0 } else { -1.0 };
                                sign * s_t / s0
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    discount * sum / n
                }
                Greek::Vega => {
                    let sum: f64 = terminals
                        .iter()
                        .map(|&s_t| {
                            let in_money = if is_call {
                                s_t > strike
                            } else {
                                s_t < strike
                            };
                            if in_money {
                                let sign = if is_call { 1.0 } else { -1.0 };
                                let log_return = (s_t / s0).ln();
                                sign * s_t
                                    * (log_return - (r + 0.5 * sigma * sigma) * t)
                                    / sigma
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    discount * sum / n
                }
                _ => f64::NAN,
            };
            (g, value)
        })
        .collect()
}

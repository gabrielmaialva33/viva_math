//// Learning-rate / coefficient schedulers.
////
//// Pure functions `(step, config) -> Float`. No internal state. Useful for:
////
//// - `viva_tensor` learning-rate schedules during training
//// - `viva_emotion` decay of stimulus weights / mood half-lives
//// - any time-varying coefficient
////
//// All schedulers are total — even out-of-range inputs are clamped to a
//// sensible value rather than erroring. This makes them safe inside hot
//// training loops.

import viva_math/constants
import viva_math/scalar

// ============================================================================
// Step / piecewise schedulers
// ============================================================================

/// Step decay: lr · γ^(step / step_size).
///
/// Drops by factor γ every `step_size` steps.
pub fn step_decay(
  base_lr: Float,
  step: Int,
  step_size: Int,
  gamma: Float,
) -> Float {
  case step_size <= 0 {
    True -> base_lr
    False -> {
      let drops = step / step_size
      base_lr *. scalar.pow(gamma, int_to_float(drops))
    }
  }
}

/// Multi-step decay: drops by γ each time step crosses a milestone.
pub fn multi_step_decay(
  base_lr: Float,
  step: Int,
  milestones: List(Int),
  gamma: Float,
) -> Float {
  let crossed = count_le(milestones, step, 0)
  base_lr *. scalar.pow(gamma, int_to_float(crossed))
}

fn count_le(xs: List(Int), step: Int, acc: Int) -> Int {
  case xs {
    [] -> acc
    [m, ..rest] ->
      case step >= m {
        True -> count_le(rest, step, acc + 1)
        False -> count_le(rest, step, acc)
      }
  }
}

// ============================================================================
// Continuous decay
// ============================================================================

/// Exponential decay: lr · γ^step.
pub fn exponential(base_lr: Float, step: Int, gamma: Float) -> Float {
  base_lr *. scalar.pow(gamma, int_to_float(step))
}

/// Inverse decay: lr / (1 + step / τ).
pub fn inverse(base_lr: Float, step: Int, tau: Float) -> Float {
  case tau <=. 0.0 {
    True -> base_lr
    False -> base_lr /. { 1.0 +. int_to_float(step) /. tau }
  }
}

/// Inverse square-root decay: lr / √(1 + step / τ). Common in transformers.
pub fn inverse_sqrt(base_lr: Float, step: Int, tau: Float) -> Float {
  case tau <=. 0.0 {
    True -> base_lr
    False -> base_lr /. scalar.sqrt(1.0 +. int_to_float(step) /. tau)
  }
}

/// Polynomial decay: lr · (1 - step / total)^power, clamped at zero past `total`.
pub fn polynomial(
  base_lr: Float,
  step: Int,
  total_steps: Int,
  power: Float,
) -> Float {
  case total_steps <= 0 {
    True -> base_lr
    False ->
      case step >= total_steps {
        True -> 0.0
        False -> {
          let frac = 1.0 -. int_to_float(step) /. int_to_float(total_steps)
          base_lr *. scalar.pow(frac, power)
        }
      }
  }
}

// ============================================================================
// Warmup + cosine annealing
// ============================================================================

/// Linear warmup: ramps from 0 to base_lr over `warmup_steps`.
pub fn linear_warmup(base_lr: Float, step: Int, warmup_steps: Int) -> Float {
  case warmup_steps <= 0 {
    True -> base_lr
    False ->
      case step >= warmup_steps {
        True -> base_lr
        False -> base_lr *. int_to_float(step) /. int_to_float(warmup_steps)
      }
  }
}

/// Cosine annealing: smooth cosine ride from base_lr to min_lr over T_max steps.
///
/// lr(t) = min_lr + ½(base_lr - min_lr)·(1 + cos(π · t / T_max))
pub fn cosine_annealing(
  base_lr: Float,
  step: Int,
  t_max: Int,
  min_lr: Float,
) -> Float {
  case t_max <= 0 {
    True -> base_lr
    False -> {
      let t = case step > t_max {
        True -> int_to_float(t_max)
        False -> int_to_float(step)
      }
      let progress = t /. int_to_float(t_max)
      let cos_term = cosine(constants.pi *. progress)
      min_lr +. 0.5 *. { base_lr -. min_lr } *. { 1.0 +. cos_term }
    }
  }
}

/// Cosine warmup-restart: cosine annealing that resets to base_lr every
/// `period` steps. The period doubles each restart when `t_mult = 2`.
pub fn cosine_warm_restarts(
  base_lr: Float,
  step: Int,
  period: Int,
  t_mult: Int,
  min_lr: Float,
) -> Float {
  let #(current_period, step_in_period) =
    advance_period(step, period, t_mult, period)
  cosine_annealing(base_lr, step_in_period, current_period, min_lr)
}

fn advance_period(
  step: Int,
  current_period: Int,
  t_mult: Int,
  budget: Int,
) -> #(Int, Int) {
  case step < budget {
    True -> #(current_period, step - { budget - current_period })
    False -> {
      let mult = case t_mult < 1 {
        True -> 1
        False -> t_mult
      }
      let next_period = current_period * mult
      advance_period(step, next_period, t_mult, budget + next_period)
    }
  }
}

// ============================================================================
// One-cycle (Smith 2018)
// ============================================================================

pub type OneCycleConfig {
  OneCycleConfig(
    max_lr: Float,
    initial_lr: Float,
    final_lr: Float,
    total_steps: Int,
    /// Fraction of total spent on increasing LR (default 0.3).
    pct_start: Float,
  )
}

/// Default 1-cycle configuration: 30 % warmup, then anneal.
pub fn one_cycle_defaults(max_lr: Float, total_steps: Int) -> OneCycleConfig {
  OneCycleConfig(
    max_lr: max_lr,
    initial_lr: max_lr /. 25.0,
    final_lr: max_lr /. 1.0e4,
    total_steps: total_steps,
    pct_start: 0.3,
  )
}

/// One-cycle policy: cosine ramp up to `max_lr`, then cosine anneal down.
pub fn one_cycle(config: OneCycleConfig, step: Int) -> Float {
  case config.total_steps <= 0 {
    True -> config.initial_lr
    False -> {
      let warmup_end =
        trunc_to_int(int_to_float(config.total_steps) *. config.pct_start)
      let progress = case step <= warmup_end {
        True -> {
          let p = int_to_float(step) /. int_to_float(int_max(warmup_end, 1))
          cosine_interp(config.initial_lr, config.max_lr, p)
        }
        False -> {
          let span = int_max(config.total_steps - warmup_end, 1)
          let p = int_to_float(step - warmup_end) /. int_to_float(span)
          cosine_interp(config.max_lr, config.final_lr, p)
        }
      }
      progress
    }
  }
}

/// Cosine interpolation between `a` and `b` for `t ∈ [0, 1]`.
fn cosine_interp(a: Float, b: Float, t: Float) -> Float {
  let t_clamped = scalar.clamp_unit(t)
  let cos_term = cosine(constants.pi *. t_clamped)
  a +. 0.5 *. { b -. a } *. { 1.0 -. cos_term }
}

// ============================================================================
// Curve helpers (no LR opinion)
// ============================================================================

/// Triangle waveform: 0 → 1 → 0 over `period` steps.
pub fn triangle(step: Int, period: Int) -> Float {
  case period <= 0 {
    True -> 0.0
    False -> {
      let half = int_to_float(period) /. 2.0
      let pos =
        int_to_float(step)
        -. half
        *. float_floor_div(int_to_float(step), int_to_float(period))
      case pos <=. half {
        True -> pos /. half
        False -> 2.0 -. pos /. half
      }
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

@external(erlang, "erlang", "float")
fn int_to_float_erl(n: Int) -> Float

fn int_to_float(n: Int) -> Float {
  int_to_float_erl(n)
}

@external(erlang, "erlang", "trunc")
fn trunc_to_int(x: Float) -> Int

@external(erlang, "math", "cos")
fn cosine(x: Float) -> Float

fn int_max(a: Int, b: Int) -> Int {
  case a > b {
    True -> a
    False -> b
  }
}

fn float_floor_div(a: Float, b: Float) -> Float {
  trunc_to_int(a /. b) |> int_to_float
}

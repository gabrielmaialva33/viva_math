//// Cusp Catastrophe Theory implementation.
////
//// Based on René Thom's catastrophe theory (1972).
//// The cusp is the simplest catastrophe with two control parameters.
////
//// Potential function: V(x) = x⁴/4 + αx²/2 + βx
//// Gradient: dV/dx = x³ + αx + β
//// Discriminant: Δ = -4α³ - 27β²
////
//// When Δ > 0 and α < 0: bistable region (two stable states)
//// This models emotional "phase transitions" - sudden mood shifts.
////
//// References:
//// - Grasman et al. (2009) "Fitting the Cusp Catastrophe in R"
//// - Van der Maas et al. (2003) "Sudden Transitions in Attitudes"

import gleam/float
import gleam/list
import gleam_community/maths
import viva_math/common

/// Cusp catastrophe parameters.
/// - alpha: normal factor (bifurcation parameter)
/// - beta: asymmetry factor (splitting factor)
pub type CuspParams {
  CuspParams(alpha: Float, beta: Float)
}

/// Result of equilibria calculation.
pub type CuspResult {
  /// Single stable state
  Monostable(equilibrium: Float)
  /// Two stable states with one unstable between them
  Bistable(lower: Float, unstable: Float, upper: Float)
}

/// Create cusp parameters from PAD arousal and dominance.
///
/// Mapping (from VIVA emotional dynamics):
/// - alpha = -arousal (high arousal → negative alpha → bistability)
/// - beta = dominance * 0.5 (dominance biases equilibrium)
///
/// This means high arousal creates emotional volatility.
pub fn from_arousal_dominance(arousal: Float, dominance: Float) -> CuspParams {
  let alpha = 0.0 -. arousal
  let beta = dominance *. 0.5
  CuspParams(alpha: alpha, beta: beta)
}

/// Compute the cusp potential V(x) = x⁴/4 + αx²/2 + βx
///
/// The potential represents emotional "energy landscape".
/// Stable states are at local minima of this function.
pub fn potential(x: Float, params: CuspParams) -> Float {
  let x2 = x *. x
  let x4 = x2 *. x2
  { x4 /. 4.0 } +. { params.alpha *. x2 /. 2.0 } +. { params.beta *. x }
}

/// Compute the gradient dV/dx = x³ + αx + β
///
/// Equilibria are where gradient = 0.
pub fn gradient(x: Float, params: CuspParams) -> Float {
  let x3 = x *. x *. x
  x3 +. { params.alpha *. x } +. params.beta
}

/// Compute the discriminant Δ = -4α³ - 27β²
///
/// Δ > 0 and α < 0: bistable region
/// Δ ≤ 0 or α ≥ 0: monostable region
pub fn discriminant(params: CuspParams) -> Float {
  let alpha3 = params.alpha *. params.alpha *. params.alpha
  let beta2 = params.beta *. params.beta
  { -4.0 *. alpha3 } -. { 27.0 *. beta2 }
}

/// Check if the system is in bistable region.
///
/// Bistability requires:
/// 1. α < 0 (necessary for two minima)
/// 2. Δ > 0 (discriminant positive)
pub fn is_bistable(params: CuspParams) -> Bool {
  params.alpha <. 0.0 && discriminant(params) >. 0.0
}

/// Calculate equilibria (roots of the cubic x³ + αx + β = 0).
///
/// Uses Cardano's formula for the depressed cubic.
/// Returns Monostable or Bistable depending on discriminant.
pub fn equilibria(params: CuspParams) -> CuspResult {
  let _disc = discriminant(params)

  case is_bistable(params) {
    False -> {
      // Monostable: use Cardano's formula for single real root
      let root = cardano_single_root(params.alpha, params.beta)
      Monostable(root)
    }
    True -> {
      // Bistable: three real roots via trigonometric method
      let roots = trigonometric_roots(params.alpha, params.beta)
      case roots {
        [r1, r2, r3] -> {
          // Sort: lower stable, middle unstable, upper stable
          let sorted = sort_three(r1, r2, r3)
          Bistable(sorted.0, sorted.1, sorted.2)
        }
        _ -> {
          // Fallback (shouldn't happen if is_bistable is true)
          Monostable(cardano_single_root(params.alpha, params.beta))
        }
      }
    }
  }
}

/// Find nearest stable equilibrium to current state.
pub fn nearest_equilibrium(x: Float, params: CuspParams) -> Float {
  case equilibria(params) {
    Monostable(eq) -> eq
    Bistable(lower, _unstable, upper) -> {
      let dist_lower = float.absolute_value(x -. lower)
      let dist_upper = float.absolute_value(x -. upper)
      case dist_lower <. dist_upper {
        True -> lower
        False -> upper
      }
    }
  }
}

/// Check if state would "jump" to other attractor.
///
/// In bistable regime, if state crosses the unstable equilibrium,
/// it will rapidly transition to the opposite stable state.
pub fn would_jump(x: Float, params: CuspParams) -> Bool {
  case equilibria(params) {
    Monostable(_) -> False
    Bistable(lower, unstable, upper) -> {
      // Check if state is on opposite side of unstable point from its attractor
      let at_lower =
        float.absolute_value(x -. lower) <. float.absolute_value(x -. upper)
      case at_lower {
        True -> x >. unstable
        // Was attracted to lower but crossed unstable
        False -> x <. unstable
        // Was attracted to upper but crossed unstable
      }
    }
  }
}

/// Compute emotional volatility based on cusp geometry.
///
/// Volatility is high when:
/// 1. In bistable region
/// 2. Close to the unstable equilibrium (catastrophe manifold)
pub fn volatility(x: Float, params: CuspParams) -> Float {
  case is_bistable(params) {
    False -> 0.0
    True -> {
      case equilibria(params) {
        Monostable(_) -> 0.0
        Bistable(_lower, unstable, _upper) -> {
          // Volatility increases near unstable point
          let dist = float.absolute_value(x -. unstable)
          // Gaussian-like decay from unstable point
          let neg_dist_sq = 0.0 -. { dist *. dist }
          maths.exponential(neg_dist_sq)
        }
      }
    }
  }
}

// Internal: Cardano's formula for single real root
fn cardano_single_root(alpha: Float, beta: Float) -> Float {
  // For x³ + px + q = 0, p = alpha, q = beta
  // Discriminant for Cardano: D = (q/2)² + (p/3)³
  let p = alpha
  let q = beta

  let p_over_3 = p /. 3.0
  let q_over_2 = q /. 2.0

  let p3 = p_over_3 *. p_over_3 *. p_over_3
  let q2 = q_over_2 *. q_over_2

  let d = q2 +. p3

  case d >=. 0.0 {
    True -> {
      // One real root
      let sqrt_d = case maths.nth_root(d, 2) {
        Ok(v) -> v
        Error(_) -> 0.0
      }

      let neg_q_over_2 = 0.0 -. q_over_2
      let u = cbrt(neg_q_over_2 +. sqrt_d)
      let v = cbrt(neg_q_over_2 -. sqrt_d)

      u +. v
    }
    False -> {
      // Three real roots - use first one from trigonometric
      case trigonometric_roots(alpha, beta) {
        [r, ..] -> r
        [] -> 0.0
      }
    }
  }
}

// Internal: Trigonometric solution for three real roots
fn trigonometric_roots(alpha: Float, beta: Float) -> List(Float) {
  // For x³ + px + q = 0 with p < 0 and three real roots
  // Using Viète's trigonometric solution
  let p = alpha
  let q = beta

  // Ensure p < 0 for this method
  case p >=. 0.0 {
    True -> []
    False -> {
      // m = 2 * sqrt(-p/3)
      let neg_p = 0.0 -. p
      let neg_p_over_3 = neg_p /. 3.0
      let m = case maths.nth_root(neg_p_over_3, 2) {
        Ok(v) -> 2.0 *. v
        Error(_) -> 0.0
      }

      // cos_arg = (3q/2p) * sqrt(-3/p)
      // Simplified: cos_arg = (3q) / (2p * sqrt((-p/3)^3))
      let neg_p_cubed = neg_p_over_3 *. neg_p_over_3 *. neg_p_over_3
      let cos_arg = case maths.nth_root(neg_p_cubed, 2) {
        Ok(denom) -> {
          case denom == 0.0 {
            True -> 0.0
            False -> {
              let numerator = 3.0 *. q
              let denominator = 2.0 *. p *. denom
              let raw = numerator /. denominator
              0.0 -. raw
            }
          }
        }
        Error(_) -> 0.0
      }

      // Clamp to [-1, 1] for acos
      let cos_arg_clamped = common.clamp(cos_arg, -1.0, 1.0)
      let theta = case maths.acos(cos_arg_clamped) {
        Ok(v) -> v /. 3.0
        Error(_) -> 0.0
      }

      let pi = maths.pi()
      let two_pi_over_3 = 2.0 *. pi /. 3.0
      let four_pi_over_3 = 4.0 *. pi /. 3.0
      let r1 = m *. maths.cos(theta)
      let r2 = m *. maths.cos(theta -. two_pi_over_3)
      let r3 = m *. maths.cos(theta -. four_pi_over_3)

      [r1, r2, r3]
    }
  }
}

// Internal: Cube root (handles negative numbers)
fn cbrt(x: Float) -> Float {
  case x >=. 0.0 {
    True ->
      case maths.nth_root(x, 3) {
        Ok(v) -> v
        Error(_) -> 0.0
      }
    False -> {
      let neg_x = 0.0 -. x
      case maths.nth_root(neg_x, 3) {
        Ok(v) -> 0.0 -. v
        Error(_) -> 0.0
      }
    }
  }
}

// Internal: Sort three floats
fn sort_three(a: Float, b: Float, c: Float) -> #(Float, Float, Float) {
  case a <=. b {
    True ->
      case b <=. c {
        True -> #(a, b, c)
        False ->
          case a <=. c {
            True -> #(a, c, b)
            False -> #(c, a, b)
          }
      }
    False ->
      case a <=. c {
        True -> #(b, a, c)
        False ->
          case b <=. c {
            True -> #(b, c, a)
            False -> #(c, b, a)
          }
      }
  }
}

// ============================================================================
// STOCHASTIC CUSP (DeepSeek R1 proposal)
// ============================================================================

/// Stochastic cusp parameters with noise intensity.
pub type StochasticCuspParams {
  StochasticCuspParams(
    /// Normal factor (bifurcation parameter)
    alpha: Float,
    /// Asymmetry factor (splitting factor)
    beta: Float,
    /// Noise intensity σ
    sigma: Float,
    /// Random seed for reproducibility
    seed: Int,
  )
}

/// Stochastic gradient with Wiener process noise.
///
/// dV/dx = x³ + αx + β + σξ(t)
///
/// Where ξ(t) is white noise (approximated deterministically).
/// Proposed by DeepSeek R1 for modeling emotional uncertainty.
pub fn stochastic_gradient(
  x: Float,
  params: StochasticCuspParams,
  step: Int,
) -> Float {
  let deterministic =
    gradient(x, CuspParams(alpha: params.alpha, beta: params.beta))
  let noise = common.deterministic_noise(step, params.seed)
  deterministic +. params.sigma *. noise
}

/// Stochastic Euler-Maruyama integration step.
///
/// x(t+dt) = x(t) - gradient(x) × dt + σ × √dt × ξ(t)
///
/// Uses DeepSeek R1 recommendation for stochastic dynamics.
pub fn stochastic_step(
  x: Float,
  params: StochasticCuspParams,
  dt: Float,
  step: Int,
) -> Float {
  let grad = gradient(x, CuspParams(alpha: params.alpha, beta: params.beta))
  let noise = common.wiener_increment(step, params.seed, dt)
  x -. grad *. dt +. params.sigma *. noise
}

/// Simulate stochastic cusp trajectory.
///
/// Returns list of states over time.
pub fn simulate_stochastic(
  initial_x: Float,
  params: StochasticCuspParams,
  dt: Float,
  steps: Int,
) -> List(Float) {
  simulate_stochastic_helper(initial_x, params, dt, steps, 0, [initial_x])
}

fn simulate_stochastic_helper(
  x: Float,
  params: StochasticCuspParams,
  dt: Float,
  total_steps: Int,
  current_step: Int,
  acc: List(Float),
) -> List(Float) {
  case current_step >= total_steps {
    True -> list.reverse(acc)
    False -> {
      let new_x = stochastic_step(x, params, dt, current_step)
      simulate_stochastic_helper(
        new_x,
        params,
        dt,
        total_steps,
        current_step + 1,
        [new_x, ..acc],
      )
    }
  }
}

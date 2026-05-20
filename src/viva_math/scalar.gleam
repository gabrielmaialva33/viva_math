//// Scalar mathematical functions.
////
//// This module provides pipeline-friendly scalar primitives that are either
//// missing from `gleam_community_maths` or commonly needed for ML / scientific
//// computing. Where the Erlang stdlib already exposes a fast BIF (such as
//// `:math.erf/1`, `:math.erfc/1` or `:math.fmod/2`), we delegate through FFI
//// rather than reimplementing.
////
//// ## Design rules
////
//// - All functions take and return `Float` directly (no `Result`) unless a
////   genuine domain error is unavoidable.
//// - Numerically stable variants are preferred: `logsumexp`, `softplus` and
////   `safe_log/exp/sqrt` guard against overflow / NaN.
//// - Activations follow the conventions used by `viva_tensor`, `pytorch` and
////   `jax` so that `viva_tensor` can delegate scalar paths here directly.
////
//// ## Erlang BIFs used (OTP 22+, confirmed on OTP 28)
////
//// `:math.erf/1`, `:math.erfc/1`, `:math.fmod/2`, `:math.expm1/1`,
//// `:math.log1p/1`, `:math.tanh/1`, `:math.exp/1`, `:math.log/1`,
//// `:math.sqrt/1`, `:math.pow/2`.

import gleam/float
import viva_math/constants

// ============================================================================
// Erlang FFI - delegate to :math BIFs
// ============================================================================

/// Error function erf(x) = (2/√π) · ∫₀ˣ e^(-t²) dt.
///
/// Delegates to `:math.erf/1` (Erlang stdlib BIF).
@external(erlang, "math", "erf")
pub fn erf(x: Float) -> Float

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// Computed by methods that avoid cancellation for large `x`.
/// Delegates to `:math.erfc/1`.
@external(erlang, "math", "erfc")
pub fn erfc(x: Float) -> Float

/// Floating-point remainder `x mod y` (IEEE 754 fmod).
///
/// Delegates to `:math.fmod/2`.
@external(erlang, "math", "fmod")
pub fn fmod(x: Float, y: Float) -> Float

/// `exp(x) - 1` accurate for small `x`.
///
/// Uses a 4-term Maclaurin series near zero (|x| < 1e-5) to avoid the
/// catastrophic cancellation of `exp(x) - 1.0`. Falls back to the direct
/// expression elsewhere. (Not all OTP builds ship `:math.expm1/1`, so we
/// implement it portably.)
pub fn expm1(x: Float) -> Float {
  case float.absolute_value(x) <. 1.0e-5 {
    True -> x +. x *. x /. 2.0 +. x *. x *. x /. 6.0
    False -> exp(x) -. 1.0
  }
}

/// `log(1 + x)` accurate for small `x`.
///
/// Uses a Maclaurin series near zero to avoid cancellation in `ln(1.0 + x)`.
/// Falls back to direct logarithm elsewhere.
pub fn log1p(x: Float) -> Float {
  case float.absolute_value(x) <. 1.0e-4 {
    True -> {
      let x2 = x *. x
      let x3 = x2 *. x
      let x4 = x3 *. x
      x -. x2 /. 2.0 +. x3 /. 3.0 -. x4 /. 4.0
    }
    False -> ln(1.0 +. x)
  }
}

/// Hyperbolic tangent. Delegates to `:math.tanh/1`.
@external(erlang, "math", "tanh")
pub fn tanh(x: Float) -> Float

/// Natural exponential e^x. Delegates to `:math.exp/1`.
@external(erlang, "math", "exp")
pub fn exp(x: Float) -> Float

/// Natural logarithm ln(x). Domain x > 0.
@external(erlang, "math", "log")
pub fn ln(x: Float) -> Float

/// Square root.
@external(erlang, "math", "sqrt")
pub fn sqrt(x: Float) -> Float

/// Power x^y.
@external(erlang, "math", "pow")
pub fn pow(x: Float, y: Float) -> Float

// ============================================================================
// Safe variants - bounded against NaN / overflow
// ============================================================================

/// Safe natural log: returns `default` for non-positive input.
pub fn safe_log(x: Float, default: Float) -> Float {
  case x <=. 0.0 {
    True -> default
    False -> ln(x)
  }
}

/// Safe exp clamped to avoid overflow (exp(709) ≈ max_float).
pub fn safe_exp(x: Float) -> Float {
  case x {
    x if x >. 700.0 -> exp(700.0)
    x if x <. -700.0 -> 0.0
    _ -> exp(x)
  }
}

/// Safe square root: returns 0 for negative input.
pub fn safe_sqrt(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0
    False -> sqrt(x)
  }
}

/// Safe division.
pub fn safe_div(a: Float, b: Float, default: Float) -> Float {
  case b == 0.0 {
    True -> default
    False -> a /. b
  }
}

// ============================================================================
// Numerically stable utilities
// ============================================================================

/// log(exp(a) + exp(b)) without overflow.
///
/// Uses the identity log(e^a + e^b) = max(a,b) + log(1 + exp(-|a-b|)).
pub fn logaddexp(a: Float, b: Float) -> Float {
  let m = float.max(a, b)
  let d = float.absolute_value(a -. b)
  m +. log1p(exp(0.0 -. d))
}

/// log(Σ exp(xᵢ)) — log-sum-exp with max subtraction for stability.
pub fn logsumexp(xs: List(Float)) -> Float {
  case xs {
    [] -> 0.0 -. constants.max_float
    [x] -> x
    _ -> {
      let m = list_max(xs, 0.0 -. constants.max_float)
      let sum = list_sum_exp_shifted(xs, m, 0.0)
      m +. ln(sum)
    }
  }
}

fn list_max(xs: List(Float), acc: Float) -> Float {
  case xs {
    [] -> acc
    [x, ..rest] -> list_max(rest, float.max(acc, x))
  }
}

fn list_sum_exp_shifted(xs: List(Float), m: Float, acc: Float) -> Float {
  case xs {
    [] -> acc
    [x, ..rest] -> list_sum_exp_shifted(rest, m, acc +. exp(x -. m))
  }
}

/// Hypotenuse √(x² + y²) without intermediate overflow.
pub fn hypot(x: Float, y: Float) -> Float {
  let ax = float.absolute_value(x)
  let ay = float.absolute_value(y)
  case ax, ay {
    0.0, 0.0 -> 0.0
    _, _ -> {
      let m = float.max(ax, ay)
      let n = float.min(ax, ay)
      let r = n /. m
      m *. sqrt(1.0 +. r *. r)
    }
  }
}

// ============================================================================
// Generic logistic family
// ============================================================================

/// Standard sigmoid σ(x) = 1 / (1 + e^(-x)).
pub fn sigmoid(x: Float) -> Float {
  case x >=. 0.0 {
    True -> 1.0 /. { 1.0 +. exp(0.0 -. x) }
    False -> {
      // Numerically stable for very negative x.
      let ex = exp(x)
      ex /. { 1.0 +. ex }
    }
  }
}

/// Logit / inverse sigmoid: ln(p / (1 - p)). Domain p ∈ (0, 1).
pub fn logit(p: Float) -> Float {
  let p_clamped =
    float.max(float.min(p, 1.0 -. constants.epsilon), constants.epsilon)
  ln(p_clamped /. { 1.0 -. p_clamped })
}

/// Generalized sigmoid with steepness `k`: σ(kx).
pub fn sigmoid_k(x: Float, k: Float) -> Float {
  sigmoid(k *. x)
}

// ============================================================================
// Neural network activations (scalar)
// ============================================================================

/// ReLU: max(0, x).
pub fn relu(x: Float) -> Float {
  float.max(0.0, x)
}

/// Leaky ReLU: x if x > 0 else negative_slope · x.
pub fn leaky_relu(x: Float, negative_slope: Float) -> Float {
  case x >. 0.0 {
    True -> x
    False -> negative_slope *. x
  }
}

/// ELU: x if x > 0 else α(e^x - 1).
pub fn elu(x: Float, alpha: Float) -> Float {
  case x >. 0.0 {
    True -> x
    False -> alpha *. expm1(x)
  }
}

/// SELU (self-normalizing). Scale and alpha from Klambauer et al. 2017.
pub fn selu(x: Float) -> Float {
  let scale = 1.0507009873554805
  let alpha = 1.6732632423543772
  scale *. elu(x, alpha)
}

/// GELU exact: x · Φ(x) using erf.
///
/// Φ(x) = ½ · (1 + erf(x / √2)). Used by BERT/GPT.
pub fn gelu(x: Float) -> Float {
  0.5 *. x *. { 1.0 +. erf(x *. constants.inv_sqrt_2) }
}

/// GELU tanh approximation (Hendrycks & Gimpel).
///
/// Faster, used by GPT-2/PaLM. Accurate to ~4 decimals.
pub fn gelu_approx(x: Float) -> Float {
  let c = 0.7978845608028654
  // √(2/π)
  let inner = c *. { x +. 0.044715 *. x *. x *. x }
  0.5 *. x *. { 1.0 +. tanh(inner) }
}

/// SiLU / Swish: x · σ(x).
pub fn silu(x: Float) -> Float {
  x *. sigmoid(x)
}

/// Alias for `silu`. Used by Google's original Swish paper.
pub fn swish(x: Float) -> Float {
  silu(x)
}

/// Mish: x · tanh(softplus(x)).
pub fn mish(x: Float) -> Float {
  x *. tanh(softplus(x))
}

/// Softplus: ln(1 + e^x). Smooth ReLU.
///
/// Uses safe identity for large x to avoid overflow:
/// softplus(x) = max(x, 0) + log1p(exp(-|x|)).
pub fn softplus(x: Float) -> Float {
  float.max(x, 0.0) +. log1p(exp(0.0 -. float.absolute_value(x)))
}

/// Hard sigmoid: piecewise linear approximation of sigmoid.
///
/// Returns 0 for x ≤ -3, 1 for x ≥ 3, linear interpolation between.
pub fn hard_sigmoid(x: Float) -> Float {
  case x {
    x if x <=. -3.0 -> 0.0
    x if x >=. 3.0 -> 1.0
    _ -> x /. 6.0 +. 0.5
  }
}

/// Hard swish: x · hard_sigmoid(x). Used in MobileNetV3.
pub fn hard_swish(x: Float) -> Float {
  x *. hard_sigmoid(x)
}

/// Hard tanh: clamps to [-1, 1].
pub fn hard_tanh(x: Float) -> Float {
  float.max(-1.0, float.min(1.0, x))
}

// ============================================================================
// Interpolation & clamping (shorthand re-exports for pipeline use)
// ============================================================================

/// Clamp to range [min, max].
pub fn clamp(x: Float, min: Float, max: Float) -> Float {
  float.max(min, float.min(max, x))
}

/// Clamp to unit [0, 1].
pub fn clamp_unit(x: Float) -> Float {
  clamp(x, 0.0, 1.0)
}

/// Clamp to bipolar [-1, 1].
pub fn clamp_bipolar(x: Float) -> Float {
  clamp(x, -1.0, 1.0)
}

/// Linear interpolation between a and b.
pub fn lerp(a: Float, b: Float, t: Float) -> Float {
  a +. t *. { b -. a }
}

/// Smoothstep: cubic Hermite interpolation between edge0 and edge1.
pub fn smoothstep(edge0: Float, edge1: Float, x: Float) -> Float {
  let t = clamp_unit({ x -. edge0 } /. { edge1 -. edge0 })
  t *. t *. { 3.0 -. 2.0 *. t }
}

/// Smootherstep: quintic version (zero first and second derivatives at edges).
pub fn smootherstep(edge0: Float, edge1: Float, x: Float) -> Float {
  let t = clamp_unit({ x -. edge0 } /. { edge1 -. edge0 })
  t *. t *. t *. { t *. { t *. 6.0 -. 15.0 } +. 10.0 }
}

// ============================================================================
// Sign & rounding
// ============================================================================

/// Sign function: -1, 0, or 1.
pub fn sign(x: Float) -> Float {
  case x {
    x if x >. 0.0 -> 1.0
    x if x <. 0.0 -> -1.0
    _ -> 0.0
  }
}

/// Step function: 0 if x < threshold, 1 otherwise.
pub fn step(threshold: Float, x: Float) -> Float {
  case x <. threshold {
    True -> 0.0
    False -> 1.0
  }
}

/// Convert degrees to radians.
pub fn deg_to_rad(deg: Float) -> Float {
  deg *. constants.deg_to_rad
}

/// Convert radians to degrees.
pub fn rad_to_deg(rad: Float) -> Float {
  rad *. constants.rad_to_deg
}

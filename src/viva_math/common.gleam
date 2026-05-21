//// Common mathematical utilities for VIVA.
////
//// Self-contained: depends only on stdlib + sibling viva_math modules.

import gleam/float
import gleam/list
import viva_math/constants
import viva_math/scalar

/// Clamp a value to a range [min, max].
///
/// ## Examples
///
/// ```gleam
/// clamp(5.0, 0.0, 10.0)  // -> 5.0
/// clamp(-1.0, 0.0, 10.0) // -> 0.0
/// clamp(15.0, 0.0, 10.0) // -> 10.0
/// ```
pub fn clamp(value: Float, min: Float, max: Float) -> Float {
  value
  |> float.max(min)
  |> float.min(max)
}

/// Clamp a value to the unit interval [0, 1].
///
/// ## Examples
///
/// ```gleam
/// clamp_unit(0.5)  // -> 0.5
/// clamp_unit(-0.5) // -> 0.0
/// clamp_unit(1.5)  // -> 1.0
/// ```
pub fn clamp_unit(value: Float) -> Float {
  clamp(value, 0.0, 1.0)
}

/// Clamp a value to the bipolar interval [-1, 1].
/// Used for PAD dimensions.
///
/// ## Examples
///
/// ```gleam
/// clamp_bipolar(0.5)   // -> 0.5
/// clamp_bipolar(-1.5)  // -> -1.0
/// clamp_bipolar(1.5)   // -> 1.0
/// ```
pub fn clamp_bipolar(value: Float) -> Float {
  clamp(value, -1.0, 1.0)
}

/// Linear interpolation between two values.
///
/// lerp(a, b, 0.0) = a
/// lerp(a, b, 1.0) = b
/// lerp(a, b, 0.5) = (a + b) / 2
///
/// ## Examples
///
/// ```gleam
/// lerp(0.0, 10.0, 0.5)  // -> 5.0
/// lerp(0.0, 10.0, 0.25) // -> 2.5
/// ```
pub fn lerp(a: Float, b: Float, t: Float) -> Float {
  a +. { t *. { b -. a } }
}

/// Inverse linear interpolation - find t given value in range [a, b].
///
/// ## Examples
///
/// ```gleam
/// inverse_lerp(0.0, 10.0, 5.0)  // -> Ok(0.5)
/// inverse_lerp(0.0, 10.0, 2.5)  // -> Ok(0.25)
/// inverse_lerp(0.0, 0.0, 5.0)   // -> Error(Nil) (division by zero)
/// ```
pub fn inverse_lerp(a: Float, b: Float, value: Float) -> Result(Float, Nil) {
  let range = b -. a
  case range == 0.0 {
    True -> Error(Nil)
    False -> Ok({ value -. a } /. range)
  }
}

/// Sigmoid function: 1 / (1 + exp(-k * x))
///
/// Maps any real number to (0, 1).
/// k controls steepness (k=1 is standard sigmoid).
///
/// ## Examples
///
/// ```gleam
/// sigmoid(0.0, 1.0)   // -> 0.5
/// sigmoid(100.0, 1.0) // -> ~1.0
/// sigmoid(-100.0, 1.0) // -> ~0.0
/// ```
pub fn sigmoid(x: Float, k: Float) -> Float {
  let neg_kx = 0.0 -. { k *. x }
  1.0 /. { 1.0 +. scalar.exp(neg_kx) }
}

/// Standard sigmoid with k=1.
pub fn sigmoid_standard(x: Float) -> Float {
  sigmoid(x, 1.0)
}

/// Softmax function: converts a list of values to probabilities.
///
/// softmax([x1, x2, ...]) = [exp(x1)/sum, exp(x2)/sum, ...]
/// where sum = exp(x1) + exp(x2) + ...
///
/// ## Examples
///
/// ```gleam
/// softmax([1.0, 2.0, 3.0])  // -> [0.09, 0.24, 0.67] (approx)
/// softmax([0.0, 0.0])        // -> [0.5, 0.5]
/// ```
pub fn softmax(values: List(Float)) -> Result(List(Float), Nil) {
  case values {
    [] -> Error(Nil)
    _ -> {
      // Find max for numerical stability
      let max_val = list.fold(values, 0.0, float.max)

      // Compute exp(x - max) for stability
      let exps = list.map(values, fn(x) { scalar.exp(x -. max_val) })

      // Sum of exponentials
      let sum = list.fold(exps, 0.0, fn(acc, x) { acc +. x })

      case sum == 0.0 {
        True -> Error(Nil)
        False -> Ok(list.map(exps, fn(x) { x /. sum }))
      }
    }
  }
}

/// Safe division with default value on division by zero.
///
/// ## Examples
///
/// ```gleam
/// safe_div(10.0, 2.0, 0.0)  // -> 5.0
/// safe_div(10.0, 0.0, -1.0) // -> -1.0
/// ```
pub fn safe_div(a: Float, b: Float, default: Float) -> Float {
  case b == 0.0 {
    True -> default
    False -> a /. b
  }
}

/// Smooth step function (Hermite interpolation).
/// Returns 0 if x < edge0, 1 if x > edge1, smooth transition otherwise.
///
/// ## Examples
///
/// ```gleam
/// smoothstep(0.0, 1.0, 0.5)  // -> 0.5 (roughly)
/// smoothstep(0.0, 1.0, 0.0)  // -> 0.0
/// smoothstep(0.0, 1.0, 1.0)  // -> 1.0
/// ```
pub fn smoothstep(edge0: Float, edge1: Float, x: Float) -> Float {
  let t = clamp_unit({ x -. edge0 } /. { edge1 -. edge0 })
  t *. t *. { 3.0 -. 2.0 *. t }
}

/// Exponential decay: value * exp(-rate * time)
///
/// ## Examples
///
/// ```gleam
/// exponential_decay(1.0, 0.5, 1.0)  // -> ~0.606
/// exponential_decay(1.0, 0.0, 10.0) // -> 1.0 (no decay)
/// ```
pub fn exponential_decay(value: Float, rate: Float, time: Float) -> Float {
  let neg_rt = 0.0 -. { rate *. time }
  value *. scalar.exp(neg_rt)
}

/// Re-export useful constants from viva_math/constants
pub const pi = constants.pi

pub const e = constants.e

pub const tau = constants.tau

// ============================================================================
// STOCHASTIC UTILITIES (Pattern from viva_glyph/codebook.gleam)
// ============================================================================

/// Deterministic pseudo-random noise generator.
/// Uses hash-based seeding for reproducibility (no external RNG needed).
///
/// Returns value in [-1, 1] range.
///
/// ## Examples
///
/// ```gleam
/// deterministic_noise(0, 42)  // -> some value in [-1, 1]
/// deterministic_noise(0, 42)  // -> same value (deterministic)
/// ```
pub fn deterministic_noise(step: Int, seed: Int) -> Float {
  // Hash-based pseudo-random (pattern from viva_glyph)
  let hash = { step * 31 + seed * 17 + 7919 } % 1000
  { int_to_float(hash) -. 500.0 } /. 500.0
}

/// Generate Wiener process increment (discrete approximation).
/// dW ≈ √dt × N(0,1) where N is approximated by deterministic noise.
///
/// Used for stochastic differential equations.
///
/// ## Parameters
/// - step: Current time step (for determinism)
/// - seed: Random seed
/// - dt: Time step size
///
/// ## Examples
///
/// ```gleam
/// wiener_increment(0, 42, 0.01)  // -> small noise scaled by √dt
/// ```
pub fn wiener_increment(step: Int, seed: Int, dt: Float) -> Float {
  let noise = deterministic_noise(step, seed)
  case float.square_root(dt) {
    Ok(sqrt_dt) -> noise *. sqrt_dt
    Error(_) -> 0.0
  }
}

/// Inverse decay rate (pattern from viva_glyph/association.gleam).
/// η(t) = η₀ / (1 + t/τ)
///
/// Used for learning rate decay, consolidation, etc.
pub fn inverse_decay(base_rate: Float, t: Float, tau: Float) -> Float {
  case tau <=. 0.0 {
    True -> base_rate
    False -> base_rate /. { 1.0 +. t /. tau }
  }
}

/// Inverse square root decay.
/// η(t) = η₀ / √(1 + t/τ)
pub fn inverse_sqrt_decay(base_rate: Float, t: Float, tau: Float) -> Float {
  case tau <=. 0.0 {
    True -> base_rate
    False -> {
      case float.square_root(1.0 +. t /. tau) {
        Ok(sqrt_val) -> base_rate /. sqrt_val
        Error(_) -> base_rate
      }
    }
  }
}

// Helper: convert int to float
fn int_to_float(n: Int) -> Float {
  case n {
    0 -> 0.0
    1 -> 1.0
    _ -> {
      case n < 0 {
        True -> 0.0 -. int_to_float(0 - n)
        False -> {
          let half = n / 2
          let remainder = n - half * 2
          int_to_float(half) *. 2.0 +. int_to_float(remainder)
        }
      }
    }
  }
}

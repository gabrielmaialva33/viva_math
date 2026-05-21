//// Numerical calculus: differentiation and integration.
////
//// Closed-form symbolic calculus is out of scope. This module provides
//// finite-difference approximations for derivatives and quadrature rules
//// for integrals.
////
//// ## Order of accuracy
////
//// | Function                     | Order   | Notes                       |
//// | ---------------------------- | ------- | --------------------------- |
//// | `forward_diff` / `backward`  | O(h)    | One-sided                   |
//// | `central_diff`               | O(h²)   | Recommended default         |
//// | `five_point_diff`            | O(h⁴)   | Five evaluations, very accurate |
//// | `second_derivative`          | O(h²)   | Central stencil             |
//// | `trapezoid` / `simpson`      | O(h²/h⁴) | Composite rules             |
//// | `romberg`                    | adaptive | Richardson extrapolation   |

import gleam/list
import viva_math/scalar

// ============================================================================
// Numerical differentiation
// ============================================================================

/// Forward difference (f(x+h) - f(x)) / h. O(h) accurate.
pub fn forward_diff(f: fn(Float) -> Float, x: Float, h: Float) -> Float {
  { f(x +. h) -. f(x) } /. h
}

/// Backward difference (f(x) - f(x-h)) / h. O(h) accurate.
pub fn backward_diff(f: fn(Float) -> Float, x: Float, h: Float) -> Float {
  { f(x) -. f(x -. h) } /. h
}

/// Central difference (f(x+h) - f(x-h)) / (2h). O(h²) accurate.
pub fn central_diff(f: fn(Float) -> Float, x: Float, h: Float) -> Float {
  { f(x +. h) -. f(x -. h) } /. { 2.0 *. h }
}

/// Five-point stencil O(h⁴): the gold standard for smooth functions.
///
/// f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
pub fn five_point_diff(f: fn(Float) -> Float, x: Float, h: Float) -> Float {
  let f_p2 = f(x +. 2.0 *. h)
  let f_p1 = f(x +. h)
  let f_m1 = f(x -. h)
  let f_m2 = f(x -. 2.0 *. h)
  { 0.0 -. f_p2 +. 8.0 *. f_p1 -. 8.0 *. f_m1 +. f_m2 } /. { 12.0 *. h }
}

/// Second derivative via central stencil. O(h²).
///
/// f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
pub fn second_derivative(f: fn(Float) -> Float, x: Float, h: Float) -> Float {
  { f(x +. h) -. 2.0 *. f(x) +. f(x -. h) } /. { h *. h }
}

/// Gradient of a multivariate function `f: List(Float) -> Float`.
///
/// Uses central differences along each axis with step `h`.
pub fn gradient(
  f: fn(List(Float)) -> Float,
  point: List(Float),
  h: Float,
) -> List(Float) {
  gradient_loop(f, point, h, 0, list.length(point), [])
}

fn gradient_loop(
  f: fn(List(Float)) -> Float,
  point: List(Float),
  h: Float,
  i: Int,
  n: Int,
  acc: List(Float),
) -> List(Float) {
  case i >= n {
    True -> list.reverse(acc)
    False -> {
      let plus = perturb(point, i, h)
      let minus = perturb(point, i, 0.0 -. h)
      let partial = { f(plus) -. f(minus) } /. { 2.0 *. h }
      gradient_loop(f, point, h, i + 1, n, [partial, ..acc])
    }
  }
}

fn perturb(point: List(Float), index: Int, delta: Float) -> List(Float) {
  perturb_loop(point, index, delta, 0, [])
}

fn perturb_loop(
  point: List(Float),
  index: Int,
  delta: Float,
  i: Int,
  acc: List(Float),
) -> List(Float) {
  case point {
    [] -> list.reverse(acc)
    [x, ..rest] -> {
      let val = case i == index {
        True -> x +. delta
        False -> x
      }
      perturb_loop(rest, index, delta, i + 1, [val, ..acc])
    }
  }
}

// ============================================================================
// Numerical integration
// ============================================================================

/// Composite trapezoid rule over n equal subintervals.
///
/// ∫f ≈ h · ( ½f(a) + Σ f(a + i·h) + ½f(b) ),  h = (b - a) / n
pub fn trapezoid(f: fn(Float) -> Float, a: Float, b: Float, n: Int) -> Float {
  case n < 1 {
    True -> 0.0
    False -> {
      let h = { b -. a } /. int_to_float(n)
      let inner = trapezoid_inner_sum(f, a, h, 1, n)
      h *. { 0.5 *. f(a) +. inner +. 0.5 *. f(b) }
    }
  }
}

fn trapezoid_inner_sum(
  f: fn(Float) -> Float,
  a: Float,
  h: Float,
  i: Int,
  n: Int,
) -> Float {
  case i >= n {
    True -> 0.0
    False ->
      f(a +. int_to_float(i) *. h) +. trapezoid_inner_sum(f, a, h, i + 1, n)
  }
}

/// Composite Simpson's rule. n must be even and ≥ 2.
///
/// ∫f ≈ (h/3) · (f₀ + 4f₁ + 2f₂ + 4f₃ + ... + 4fₙ₋₁ + fₙ)
pub fn simpson(
  f: fn(Float) -> Float,
  a: Float,
  b: Float,
  n: Int,
) -> Result(Float, Nil) {
  case n < 2, n % 2 != 0 {
    True, _ -> Error(Nil)
    _, True -> Error(Nil)
    _, _ -> {
      let h = { b -. a } /. int_to_float(n)
      let total = simpson_loop(f, a, h, 1, n, 0.0)
      Ok(h /. 3.0 *. { f(a) +. f(b) +. total })
    }
  }
}

fn simpson_loop(
  f: fn(Float) -> Float,
  a: Float,
  h: Float,
  i: Int,
  n: Int,
  acc: Float,
) -> Float {
  case i >= n {
    True -> acc
    False -> {
      let x = a +. int_to_float(i) *. h
      let weight = case i % 2 == 0 {
        True -> 2.0
        False -> 4.0
      }
      simpson_loop(f, a, h, i + 1, n, acc +. weight *. f(x))
    }
  }
}

/// Romberg integration via Richardson extrapolation.
///
/// Repeatedly halves `h` and combines trapezoid estimates to cancel error
/// terms. `levels` controls the depth (typically 5-8 suffices).
pub fn romberg(
  f: fn(Float) -> Float,
  a: Float,
  b: Float,
  levels: Int,
) -> Float {
  case levels < 1 {
    True -> trapezoid(f, a, b, 1)
    False -> {
      let first_row = romberg_row(f, a, b, levels)
      romberg_extrapolate(first_row, 1)
    }
  }
}

fn romberg_row(
  f: fn(Float) -> Float,
  a: Float,
  b: Float,
  levels: Int,
) -> List(Float) {
  build_row(f, a, b, 0, levels, [])
}

fn build_row(
  f: fn(Float) -> Float,
  a: Float,
  b: Float,
  i: Int,
  levels: Int,
  acc: List(Float),
) -> List(Float) {
  case i >= levels {
    True -> list.reverse(acc)
    False -> {
      let n = int_pow(2, i)
      let estimate = trapezoid(f, a, b, n)
      build_row(f, a, b, i + 1, levels, [estimate, ..acc])
    }
  }
}

fn romberg_extrapolate(row: List(Float), k: Int) -> Float {
  case row {
    [] -> 0.0
    [single] -> single
    _ -> {
      let next_row = combine_pairs(row, k)
      romberg_extrapolate(next_row, k + 1)
    }
  }
}

fn combine_pairs(row: List(Float), k: Int) -> List(Float) {
  case row {
    [a, b, ..rest] -> {
      let factor = int_pow(4, k) |> int_to_float
      let improved = { factor *. b -. a } /. { factor -. 1.0 }
      [improved, ..combine_pairs([b, ..rest], k)]
    }
    _ -> []
  }
}

// ============================================================================
// Helpers
// ============================================================================

@external(erlang, "erlang", "float")
@external(javascript, "../viva_math_random_ffi.mjs", "int_to_float")
fn int_to_float_erl(n: Int) -> Float

fn int_to_float(n: Int) -> Float {
  int_to_float_erl(n)
}

fn int_pow(base: Int, exp: Int) -> Int {
  case exp {
    0 -> 1
    _ -> base * int_pow(base, exp - 1)
  }
}

// Re-export sqrt for callers wanting calculus-flavoured helpers.
pub fn norm_l2(values: List(Float)) -> Float {
  scalar.sqrt(list.fold(values, 0.0, fn(acc, x) { acc +. x *. x }))
}

//// Shared test helpers — float tolerance, vector approximate equality,
//// and complex/list comparators. Consolidates `is_close`, `close`,
//// `close_complex`, `close_list` that were previously duplicated across
//// `viva_math_test.gleam`, `qcheck_test.gleam`, `fft_test.gleam`, etc.
////
//// Convention (mirroring `gleam-lang/stdlib`): test files import this
//// module and use the helper functions directly — `is_close(a, b, tol)`
//// for scalars, `is_close_vec3(...)` for PAD vectors, etc.

import gleam/float
import gleam/list
import viva_math/complex.{type Complex}
import viva_math/vector.{type Vec3}

/// `|a − b| ≤ tol`. Use this instead of `should.equal` for any `Float`
/// comparison — direct equality on floats is a flake source.
pub fn is_close(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

/// Componentwise `is_close` on `Vec3`.
pub fn is_close_vec3(a: Vec3, b: Vec3, tol: Float) -> Bool {
  is_close(a.x, b.x, tol) && is_close(a.y, b.y, tol) && is_close(a.z, b.z, tol)
}

/// Componentwise `is_close` on `Complex`.
pub fn is_close_complex(a: Complex, b: Complex, tol: Float) -> Bool {
  is_close(a.re, b.re, tol) && is_close(a.im, b.im, tol)
}

/// Pairwise `is_close` on two `List(Float)` of equal length. Returns `False`
/// if the lists have different lengths.
pub fn is_close_list(a: List(Float), b: List(Float), tol: Float) -> Bool {
  case list.length(a) == list.length(b) {
    False -> False
    True ->
      list.zip(a, b)
      |> list.all(fn(pair) {
        let #(x, y) = pair
        is_close(x, y, tol)
      })
  }
}

/// Default tight tolerance for closed-form algebraic identities.
/// `1e-12` ≈ ~10 ulps of an IEEE-754 double near `1.0`.
pub const tight: Float = 1.0e-12

/// "Machine-precision" tolerance — used when the only error is the final
/// rounding of a single IEEE-754 operation. `1e-15` is roughly 4-5 ulps.
pub const machine: Float = 1.0e-15

/// Default tolerance for transcendental round-trips (`exp(ln(x)) = x`,
/// `sqrt(x²) = x`) where one operation accumulates ~10 ulps.
pub const transcendental: Float = 1.0e-13

/// Default loose tolerance for stochastic / Monte-Carlo / iterative checks.
pub const loose: Float = 1.0e-6

/// Relative-error comparator: `|a − b| ≤ tol·max(|a|, |b|)`.
/// Use for values whose magnitude is far from 1 (where `is_close` with an
/// absolute tolerance would be either too loose or too strict).
pub fn is_close_rel(a: Float, b: Float, rel_tol: Float) -> Bool {
  let scale = float.max(float.absolute_value(a), float.absolute_value(b))
  case scale {
    0.0 -> is_close(a, b, rel_tol)
    _ -> float.absolute_value(a -. b) <=. rel_tol *. scale
  }
}

/// Hybrid comparator: passes if either absolute or relative tolerance holds.
/// Matches CPython's `result_check` convention for math library tests.
pub fn is_close_hybrid(
  a: Float,
  b: Float,
  abs_tol: Float,
  rel_tol: Float,
) -> Bool {
  is_close(a, b, abs_tol) || is_close_rel(a, b, rel_tol)
}

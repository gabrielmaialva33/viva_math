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
pub const tight: Float = 1.0e-12

/// Default loose tolerance for stochastic / Monte-Carlo / iterative checks.
pub const loose: Float = 1.0e-6

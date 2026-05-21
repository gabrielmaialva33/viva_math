//// N-dimensional vectors as `List(Float)`.
////
//// Pure functional, no NIF, no broadcasting beyond scalar/vector. For heavy
//// linear algebra, defer to `viva_tensor`.
////
//// Most functions return `Result` when shapes mismatch.

import gleam/float
import gleam/list
import viva_math/scalar

pub type VecN =
  List(Float)

pub fn zeros(n: Int) -> VecN {
  list.repeat(0.0, n)
}

pub fn ones(n: Int) -> VecN {
  list.repeat(1.0, n)
}

pub fn splat(value: Float, n: Int) -> VecN {
  list.repeat(value, n)
}

/// Element-wise add. Errors on length mismatch.
pub fn add(a: VecN, b: VecN) -> Result(VecN, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True -> Ok(list.map(list.zip(a, b), fn(p) { p.0 +. p.1 }))
  }
}

pub fn sub(a: VecN, b: VecN) -> Result(VecN, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True -> Ok(list.map(list.zip(a, b), fn(p) { p.0 -. p.1 }))
  }
}

pub fn multiply(a: VecN, b: VecN) -> Result(VecN, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True -> Ok(list.map(list.zip(a, b), fn(p) { p.0 *. p.1 }))
  }
}

/// Scalar multiplication.
pub fn scale(v: VecN, s: Float) -> VecN {
  list.map(v, fn(x) { x *. s })
}

pub fn add_scalar(v: VecN, s: Float) -> VecN {
  list.map(v, fn(x) { x +. s })
}

pub fn dot(a: VecN, b: VecN) -> Result(Float, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True -> Ok(list.fold(list.zip(a, b), 0.0, fn(acc, p) { acc +. p.0 *. p.1 }))
  }
}

pub fn length_squared(v: VecN) -> Float {
  list.fold(v, 0.0, fn(acc, x) { acc +. x *. x })
}

/// Length using progressive `hypot` reduction to avoid overflow.
pub fn length(v: VecN) -> Float {
  case v {
    [] -> 0.0
    [x] -> scalar.safe_sqrt(x *. x)
    [a, b, ..rest] -> length_hypot(rest, scalar.hypot(a, b))
  }
}

fn length_hypot(rest: List(Float), acc: Float) -> Float {
  case rest {
    [] -> acc
    [x, ..tail] -> length_hypot(tail, scalar.hypot(acc, x))
  }
}

pub fn distance(a: VecN, b: VecN) -> Result(Float, Nil) {
  case sub(a, b) {
    Ok(diff) -> Ok(length(diff))
    Error(_) -> Error(Nil)
  }
}

pub fn normalize(v: VecN) -> VecN {
  let l = length(v)
  case l == 0.0 {
    True -> v
    False -> scale(v, 1.0 /. l)
  }
}

pub fn negate(v: VecN) -> VecN {
  list.map(v, fn(x) { 0.0 -. x })
}

pub fn clamp(v: VecN, lo: Float, hi: Float) -> VecN {
  list.map(v, fn(x) { scalar.clamp(x, lo, hi) })
}

pub fn lerp(a: VecN, b: VecN, t: Float) -> Result(VecN, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True -> Ok(list.map(list.zip(a, b), fn(p) { scalar.lerp(p.0, p.1, t) }))
  }
}

pub fn sum(v: VecN) -> Float {
  list.fold(v, 0.0, fn(acc, x) { acc +. x })
}

pub fn mean(v: VecN) -> Result(Float, Nil) {
  case v {
    [] -> Error(Nil)
    _ -> Ok(sum(v) /. int_to_float(list.length(v)))
  }
}

/// Component-wise zip with a custom binary function.
pub fn zip_with(
  a: VecN,
  b: VecN,
  f: fn(Float, Float) -> Float,
) -> Result(VecN, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True -> Ok(list.map(list.zip(a, b), fn(p) { f(p.0, p.1) }))
  }
}

/// Euclidean (L₂) distance — alias for `distance/2` for API parity with
/// `gleam_community_maths` and downstream packages migrated off it.
pub fn euclidean_distance(a: VecN, b: VecN) -> Result(Float, Nil) {
  distance(a, b)
}

/// Manhattan (L₁) distance: `Σ |aᵢ − bᵢ|`.
pub fn manhattan_distance(a: VecN, b: VecN) -> Result(Float, Nil) {
  case list.length(a) == list.length(b) {
    False -> Error(Nil)
    True ->
      Ok(
        list.fold(list.zip(a, b), 0.0, fn(acc, p) {
          acc +. float.absolute_value(p.0 -. p.1)
        }),
      )
  }
}

/// Cosine similarity: `(a · b) / (‖a‖ · ‖b‖)`.
///
/// Returns `Error(Nil)` if either vector has zero length or lengths differ.
pub fn cosine_similarity(a: VecN, b: VecN) -> Result(Float, Nil) {
  case dot(a, b) {
    Error(_) -> Error(Nil)
    Ok(d) -> {
      let na = length(a)
      let nb = length(b)
      case na == 0.0 || nb == 0.0 {
        True -> Error(Nil)
        False -> Ok(d /. { na *. nb })
      }
    }
  }
}

/// General Lₚ norm: `(Σ |xᵢ|ᵖ)^(1/p)`.
///
/// Defined for `p ≥ 1` (where it satisfies the triangle inequality and is a
/// true norm). For `0 < p < 1` the formula still computes but the result is
/// only a pseudo-norm (triangle inequality fails). The function does not
/// reject `0 < p < 1` so callers can opt into the pseudo-norm regime.
///
/// **Domain**: `p > 0` strictly. `p ≤ 0` returns `Error(Nil)` because
/// `1/p` is undefined or infinite, and the underlying `pow` raises
/// `badarith` on Erlang while returning `Infinity` on JavaScript — i.e. the
/// behaviour would diverge across targets, which is unacceptable for a
/// dual-target library.
///
/// Special cases: `p = 2.0` collapses to Euclidean (`length`), `p = 1.0` to
/// Manhattan. Returns `Ok(0.0)` for the empty vector.
pub fn lp_norm(v: VecN, p: Float) -> Result(Float, Nil) {
  case p <=. 0.0 {
    True -> Error(Nil)
    False ->
      case v {
        [] -> Ok(0.0)
        _ -> {
          let powed =
            list.fold(v, 0.0, fn(acc, x) {
              acc +. scalar.pow(float.absolute_value(x), p)
            })
          Ok(scalar.pow(powed, 1.0 /. p))
        }
      }
  }
}

@external(erlang, "erlang", "float")
@external(javascript, "../viva_math_random_ffi.mjs", "int_to_float")
fn int_to_float(n: Int) -> Float

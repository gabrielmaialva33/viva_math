//// N-dimensional vectors as `List(Float)`.
////
//// Pure functional, no NIF, no broadcasting beyond scalar/vector. For heavy
//// linear algebra, defer to `viva_tensor`.
////
//// Most functions return `Result` when shapes mismatch.

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

@external(erlang, "erlang", "float")
@external(javascript, "../viva_math_random_ffi.mjs", "int_to_float")
fn int_to_float(n: Int) -> Float

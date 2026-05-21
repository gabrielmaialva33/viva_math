//// 2-D vector operations.
////
//// Companion to `viva_math/vector` (which is dedicated to PAD/Vec3).
//// Useful for 2-D dynamics, planar geometry, polar coordinates, etc.

import gleam/float
import viva_math/scalar

pub type Vec2 {
  Vec2(x: Float, y: Float)
}

pub fn zero() -> Vec2 {
  Vec2(0.0, 0.0)
}

pub fn splat(value: Float) -> Vec2 {
  Vec2(value, value)
}

pub fn from_polar(r: Float, theta: Float) -> Vec2 {
  Vec2(r *. cosine(theta), r *. sine(theta))
}

pub fn add(a: Vec2, b: Vec2) -> Vec2 {
  Vec2(a.x +. b.x, a.y +. b.y)
}

pub fn sub(a: Vec2, b: Vec2) -> Vec2 {
  Vec2(a.x -. b.x, a.y -. b.y)
}

pub fn scale(v: Vec2, s: Float) -> Vec2 {
  Vec2(v.x *. s, v.y *. s)
}

pub fn negate(v: Vec2) -> Vec2 {
  Vec2(0.0 -. v.x, 0.0 -. v.y)
}

pub fn dot(a: Vec2, b: Vec2) -> Float {
  a.x *. b.x +. a.y *. b.y
}

/// 2-D "cross" returns the z-component of the 3-D cross product.
pub fn cross(a: Vec2, b: Vec2) -> Float {
  a.x *. b.y -. a.y *. b.x
}

pub fn length_squared(v: Vec2) -> Float {
  dot(v, v)
}

/// Length √(x² + y²) — uses `hypot` to avoid overflow/underflow at extreme scales.
pub fn length(v: Vec2) -> Float {
  scalar.hypot(v.x, v.y)
}

pub fn distance(a: Vec2, b: Vec2) -> Float {
  length(sub(a, b))
}

pub fn normalize(v: Vec2) -> Vec2 {
  let l = length(v)
  case l == 0.0 {
    True -> zero()
    False -> Vec2(v.x /. l, v.y /. l)
  }
}

pub fn lerp(a: Vec2, b: Vec2, t: Float) -> Vec2 {
  Vec2(scalar.lerp(a.x, b.x, t), scalar.lerp(a.y, b.y, t))
}

pub fn clamp(v: Vec2, lo: Float, hi: Float) -> Vec2 {
  Vec2(scalar.clamp(v.x, lo, hi), scalar.clamp(v.y, lo, hi))
}

/// Rotate `v` by `theta` radians around the origin.
pub fn rotate(v: Vec2, theta: Float) -> Vec2 {
  let c = cosine(theta)
  let s = sine(theta)
  Vec2(v.x *. c -. v.y *. s, v.x *. s +. v.y *. c)
}

/// Perpendicular vector (90° counter-clockwise rotation).
pub fn perpendicular(v: Vec2) -> Vec2 {
  Vec2(0.0 -. v.y, v.x)
}

/// Angle in radians of `v` relative to the positive x-axis. Range (-π, π].
pub fn angle(v: Vec2) -> Float {
  atan2(v.y, v.x)
}

/// Component-wise approximate equality.
pub fn is_close(a: Vec2, b: Vec2, tol: Float) -> Bool {
  float.absolute_value(a.x -. b.x) <=. tol
  && float.absolute_value(a.y -. b.y) <=. tol
}

@external(erlang, "math", "cos")
@external(javascript, "../viva_math_random_ffi.mjs", "cos")
fn cosine(x: Float) -> Float

@external(erlang, "math", "sin")
@external(javascript, "../viva_math_random_ffi.mjs", "sin")
fn sine(x: Float) -> Float

@external(erlang, "math", "atan2")
@external(javascript, "../viva_math_random_ffi.mjs", "atan2")
fn atan2(y: Float, x: Float) -> Float

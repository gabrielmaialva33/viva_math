//// 4-D vector operations.
////
//// Used for RGBA colors, homogeneous coordinates and quaternions (when the
//// fourth component is treated as w/real).

import gleam/float
import viva_math/scalar

pub type Vec4 {
  Vec4(x: Float, y: Float, z: Float, w: Float)
}

pub fn zero() -> Vec4 {
  Vec4(0.0, 0.0, 0.0, 0.0)
}

pub fn splat(value: Float) -> Vec4 {
  Vec4(value, value, value, value)
}

pub fn add(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(a.x +. b.x, a.y +. b.y, a.z +. b.z, a.w +. b.w)
}

pub fn sub(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(a.x -. b.x, a.y -. b.y, a.z -. b.z, a.w -. b.w)
}

pub fn scale(v: Vec4, s: Float) -> Vec4 {
  Vec4(v.x *. s, v.y *. s, v.z *. s, v.w *. s)
}

pub fn negate(v: Vec4) -> Vec4 {
  Vec4(0.0 -. v.x, 0.0 -. v.y, 0.0 -. v.z, 0.0 -. v.w)
}

pub fn dot(a: Vec4, b: Vec4) -> Float {
  a.x *. b.x +. a.y *. b.y +. a.z *. b.z +. a.w *. b.w
}

pub fn length_squared(v: Vec4) -> Float {
  dot(v, v)
}

/// Length using two-level hypot to avoid overflow at extreme scales.
pub fn length(v: Vec4) -> Float {
  let xy = scalar.hypot(v.x, v.y)
  let zw = scalar.hypot(v.z, v.w)
  scalar.hypot(xy, zw)
}

pub fn distance(a: Vec4, b: Vec4) -> Float {
  length(sub(a, b))
}

pub fn normalize(v: Vec4) -> Vec4 {
  let l = length(v)
  case l == 0.0 {
    True -> zero()
    False -> Vec4(v.x /. l, v.y /. l, v.z /. l, v.w /. l)
  }
}

pub fn lerp(a: Vec4, b: Vec4, t: Float) -> Vec4 {
  Vec4(
    scalar.lerp(a.x, b.x, t),
    scalar.lerp(a.y, b.y, t),
    scalar.lerp(a.z, b.z, t),
    scalar.lerp(a.w, b.w, t),
  )
}

/// Hadamard (component-wise) product.
pub fn multiply(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(a.x *. b.x, a.y *. b.y, a.z *. b.z, a.w *. b.w)
}

pub fn clamp(v: Vec4, lo: Float, hi: Float) -> Vec4 {
  Vec4(
    scalar.clamp(v.x, lo, hi),
    scalar.clamp(v.y, lo, hi),
    scalar.clamp(v.z, lo, hi),
    scalar.clamp(v.w, lo, hi),
  )
}

pub fn is_close(a: Vec4, b: Vec4, tol: Float) -> Bool {
  float.absolute_value(a.x -. b.x) <=. tol
  && float.absolute_value(a.y -. b.y) <=. tol
  && float.absolute_value(a.z -. b.z) <=. tol
  && float.absolute_value(a.w -. b.w) <=. tol
}

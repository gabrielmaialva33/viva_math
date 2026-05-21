//// Unit quaternions for 3-D rotation.
////
//// A quaternion `w + xi + yj + zk` extends complex numbers to four
//// dimensions. Unit quaternions provide a numerically stable,
//// gimbal-lock-free representation of 3-D rotation that interpolates
//// smoothly via SLERP.
////
//// ## Conventions
////
//// - `q.w` is the scalar (real) part.
//// - `(q.x, q.y, q.z)` is the vector (imaginary) part.
//// - Rotation by angle `θ` around unit axis `(ax, ay, az)`:
////   `q = cos(θ/2) + sin(θ/2)·(ax·i + ay·j + az·k)`.
//// - Right-handed coordinates.
////
//// ## When to use
////
//// - Smooth interpolation between two orientations (`slerp`).
//// - Composing rotations without matrix multiplication overhead.
//// - Avoiding gimbal lock present in Euler-angle representations.

import gleam/float
import viva_math/scalar
import viva_math/vector.{type Vec3, Vec3}

pub type Quaternion {
  Quaternion(w: Float, x: Float, y: Float, z: Float)
}

// ============================================================================
// Construction
// ============================================================================

/// The identity quaternion (no rotation).
pub fn identity() -> Quaternion {
  Quaternion(w: 1.0, x: 0.0, y: 0.0, z: 0.0)
}

/// Build a unit quaternion from an axis-angle representation.
///
/// `axis` need not be normalised — this function normalises it. `theta` is
/// the rotation angle in radians.
pub fn from_axis_angle(axis: Vec3, theta: Float) -> Quaternion {
  let n = vector.normalize(axis)
  let half = theta /. 2.0
  let s = sine(half)
  Quaternion(w: cosine(half), x: n.x *. s, y: n.y *. s, z: n.z *. s)
}

/// Build a quaternion from raw components without normalisation.
pub fn raw(w: Float, x: Float, y: Float, z: Float) -> Quaternion {
  Quaternion(w: w, x: x, y: y, z: z)
}

// ============================================================================
// Algebra
// ============================================================================

/// Hamilton product `a · b`. Non-commutative.
pub fn mul(a: Quaternion, b: Quaternion) -> Quaternion {
  Quaternion(
    w: a.w *. b.w -. a.x *. b.x -. a.y *. b.y -. a.z *. b.z,
    x: a.w *. b.x +. a.x *. b.w +. a.y *. b.z -. a.z *. b.y,
    y: a.w *. b.y -. a.x *. b.z +. a.y *. b.w +. a.z *. b.x,
    z: a.w *. b.z +. a.x *. b.y -. a.y *. b.x +. a.z *. b.w,
  )
}

/// Quaternion conjugate `(w, -x, -y, -z)`. For unit quaternions this is the inverse.
pub fn conjugate(q: Quaternion) -> Quaternion {
  Quaternion(w: q.w, x: 0.0 -. q.x, y: 0.0 -. q.y, z: 0.0 -. q.z)
}

/// Magnitude / Euclidean norm of a quaternion.
pub fn magnitude(q: Quaternion) -> Float {
  scalar.sqrt(q.w *. q.w +. q.x *. q.x +. q.y *. q.y +. q.z *. q.z)
}

/// Squared magnitude (cheaper than `magnitude` when only comparison is needed).
pub fn magnitude_squared(q: Quaternion) -> Float {
  q.w *. q.w +. q.x *. q.x +. q.y *. q.y +. q.z *. q.z
}

/// Normalise to unit length. Returns identity if the input is zero.
pub fn normalize(q: Quaternion) -> Quaternion {
  let m = magnitude(q)
  case m == 0.0 {
    True -> identity()
    False -> Quaternion(w: q.w /. m, x: q.x /. m, y: q.y /. m, z: q.z /. m)
  }
}

/// Inverse `conj(q) / |q|²`. For unit quaternions equals the conjugate.
pub fn inverse(q: Quaternion) -> Quaternion {
  let m2 = magnitude_squared(q)
  case m2 == 0.0 {
    True -> identity()
    False -> {
      let c = conjugate(q)
      Quaternion(w: c.w /. m2, x: c.x /. m2, y: c.y /. m2, z: c.z /. m2)
    }
  }
}

/// Quaternion-quaternion dot product (treating each as a 4-vector).
pub fn dot(a: Quaternion, b: Quaternion) -> Float {
  a.w *. b.w +. a.x *. b.x +. a.y *. b.y +. a.z *. b.z
}

// ============================================================================
// Rotation application
// ============================================================================

/// Rotate a 3-D vector `v` by unit quaternion `q`.
///
/// Computes `q · v · q⁻¹` using the optimised Rodrigues-style formula that
/// skips two of the quaternion products.
pub fn rotate(q: Quaternion, v: Vec3) -> Vec3 {
  // u = vector part of q, s = scalar part
  let s = q.w
  let u = Vec3(q.x, q.y, q.z)
  // v' = 2·(u·v)·u + (s² - u·u)·v + 2·s·(u × v)
  let u_dot_v = vector.dot(u, v)
  let u_dot_u = vector.dot(u, u)
  let cross_uv = vector.cross(u, v)
  let term1 = vector.scale(u, 2.0 *. u_dot_v)
  let term2 = vector.scale(v, s *. s -. u_dot_u)
  let term3 = vector.scale(cross_uv, 2.0 *. s)
  vector.add(vector.add(term1, term2), term3)
}

// ============================================================================
// Interpolation
// ============================================================================

/// Linear interpolation (LERP) between two quaternions, then normalise.
///
/// Cheaper than SLERP but yields non-uniform angular speed. Acceptable for
/// small angular distances (< ~30°). For large or critical interpolations
/// use `slerp`.
pub fn nlerp(a: Quaternion, b: Quaternion, t: Float) -> Quaternion {
  let t_clamped = scalar.clamp_unit(t)
  // Pick shorter arc.
  let b_signed = case dot(a, b) <. 0.0 {
    True ->
      Quaternion(w: 0.0 -. b.w, x: 0.0 -. b.x, y: 0.0 -. b.y, z: 0.0 -. b.z)
    False -> b
  }
  let one_minus_t = 1.0 -. t_clamped
  normalize(Quaternion(
    w: one_minus_t *. a.w +. t_clamped *. b_signed.w,
    x: one_minus_t *. a.x +. t_clamped *. b_signed.x,
    y: one_minus_t *. a.y +. t_clamped *. b_signed.y,
    z: one_minus_t *. a.z +. t_clamped *. b_signed.z,
  ))
}

/// Spherical linear interpolation (SLERP) — constant angular velocity.
///
/// Falls back to `nlerp` when the angle between the quaternions is very
/// small (`sin(θ) < 1e-6`), where SLERP becomes numerically unstable.
pub fn slerp(a: Quaternion, b: Quaternion, t: Float) -> Quaternion {
  let t_clamped = scalar.clamp_unit(t)
  let cos_theta = dot(a, b)
  // Pick shorter arc.
  let #(b_signed, cos_theta_signed) = case cos_theta <. 0.0 {
    True -> #(
      Quaternion(w: 0.0 -. b.w, x: 0.0 -. b.x, y: 0.0 -. b.y, z: 0.0 -. b.z),
      0.0 -. cos_theta,
    )
    False -> #(b, cos_theta)
  }
  case cos_theta_signed >. 0.9995 {
    True -> nlerp(a, b_signed, t_clamped)
    False -> {
      let theta = acos_safe(cos_theta_signed)
      let sin_theta = sine(theta)
      let ratio_a = sine({ 1.0 -. t_clamped } *. theta) /. sin_theta
      let ratio_b = sine(t_clamped *. theta) /. sin_theta
      Quaternion(
        w: ratio_a *. a.w +. ratio_b *. b_signed.w,
        x: ratio_a *. a.x +. ratio_b *. b_signed.x,
        y: ratio_a *. a.y +. ratio_b *. b_signed.y,
        z: ratio_a *. a.z +. ratio_b *. b_signed.z,
      )
    }
  }
}

// ============================================================================
// Conversion
// ============================================================================

/// Quaternion → axis-angle. Returns `(axis, angle)` with `axis` unit-length.
/// When the rotation is identity returns `(z-axis, 0)` by convention.
pub fn to_axis_angle(q: Quaternion) -> #(Vec3, Float) {
  let qn = normalize(q)
  let w_clamped = case qn.w {
    w if w >. 1.0 -> 1.0
    w if w <. -1.0 -> -1.0
    w -> w
  }
  let angle = 2.0 *. acos_safe(w_clamped)
  let s = scalar.sqrt(1.0 -. w_clamped *. w_clamped)
  case s <. 1.0e-9 {
    True -> #(Vec3(0.0, 0.0, 1.0), 0.0)
    False -> #(Vec3(qn.x /. s, qn.y /. s, qn.z /. s), angle)
  }
}

/// Approximate equality up to a tolerance.
pub fn is_close(a: Quaternion, b: Quaternion, tol: Float) -> Bool {
  float.absolute_value(a.w -. b.w) <=. tol
  && float.absolute_value(a.x -. b.x) <=. tol
  && float.absolute_value(a.y -. b.y) <=. tol
  && float.absolute_value(a.z -. b.z) <=. tol
}

@external(erlang, "math", "sin")
@external(javascript, "../viva_math_random_ffi.mjs", "sin")
fn sine(x: Float) -> Float

@external(erlang, "math", "cos")
@external(javascript, "../viva_math_random_ffi.mjs", "cos")
fn cosine(x: Float) -> Float

@external(erlang, "math", "acos")
@external(javascript, "../viva_math_random_ffi.mjs", "acos")
fn acos_raw(x: Float) -> Float

fn acos_safe(x: Float) -> Float {
  acos_raw(x)
}

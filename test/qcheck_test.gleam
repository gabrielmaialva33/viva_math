//// Property-based tests with random input generation via `qcheck`.
////
//// Each property runs 100 randomised cases; failures shrink to a minimal
//// counterexample. Verifies invariants that must hold for *all* inputs in
//// the relevant domain.

import gleam/float
import gleam/list
import gleeunit/should
import qcheck
import viva_math/complex
import viva_math/precision
import viva_math/quaternion
import viva_math/scalar
import viva_math/vector

fn close(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

// ============================================================================
// scalar: sigmoid symmetry σ(x) + σ(-x) = 1 across all floats
// ============================================================================

pub fn qcheck_sigmoid_symmetry_test() {
  use x <- qcheck.given(qcheck.bounded_float(-50.0, 50.0))
  let sum = scalar.sigmoid(x) +. scalar.sigmoid(0.0 -. x)
  should.be_true(close(sum, 1.0, 1.0e-12))
}

pub fn qcheck_relu_idempotent_test() {
  use x <- qcheck.given(qcheck.bounded_float(-100.0, 100.0))
  let twice = scalar.relu(scalar.relu(x))
  let once = scalar.relu(x)
  should.be_true(twice == once)
}

pub fn qcheck_softplus_positive_test() {
  // softplus(x) > 0 for all finite x
  use x <- qcheck.given(qcheck.bounded_float(-100.0, 100.0))
  should.be_true(scalar.softplus(x) >. 0.0)
}

pub fn qcheck_softplus_monotonic_test() {
  // x < y → softplus(x) < softplus(y)
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(-50.0, 50.0),
    qcheck.bounded_float(0.001, 5.0),
  ))
  let #(x, delta) = pair
  let y = x +. delta
  should.be_true(scalar.softplus(x) <. scalar.softplus(y))
}

pub fn qcheck_tanh_bounded_test() {
  use x <- qcheck.given(qcheck.bounded_float(-1000.0, 1000.0))
  let t = scalar.tanh(x)
  should.be_true(t >=. -1.0 && t <=. 1.0)
}

pub fn qcheck_hypot_pythagorean_test() {
  // hypot is symmetric and non-negative
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(-1.0e6, 1.0e6),
    qcheck.bounded_float(-1.0e6, 1.0e6),
  ))
  let #(a, b) = pair
  let h = scalar.hypot(a, b)
  let h_swapped = scalar.hypot(b, a)
  should.be_true(h >=. 0.0)
  should.be_true(close(h, h_swapped, 1.0e-9))
}

// ============================================================================
// precision: Neumaier sum invariants
// ============================================================================

pub fn qcheck_neumaier_matches_pairwise_test() {
  // On well-conditioned inputs the two methods must agree.
  use xs <- qcheck.given(qcheck.list_from(qcheck.bounded_float(-100.0, 100.0)))
  let neumaier = precision.neumaier_sum(xs)
  let pairwise = precision.pairwise_sum(xs)
  should.be_true(close(neumaier, pairwise, 1.0e-9))
}

pub fn qcheck_two_sum_exactness_test() {
  // two_sum: a + b = hi + lo exactly (free of rounding error).
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(-1.0e6, 1.0e6),
    qcheck.bounded_float(-1.0e6, 1.0e6),
  ))
  let #(a, b) = pair
  let #(hi, lo) = precision.two_sum(a, b)
  // hi + lo must reproduce (a + b) up to a single rounding.
  should.be_true(close(hi +. lo, a +. b, 1.0e-9))
}

// ============================================================================
// vector: dot product is commutative
// ============================================================================

pub fn qcheck_vec3_dot_commutative_test() {
  use t <- qcheck.given(qcheck.tuple6(
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
  ))
  let #(ax, ay, az, bx, by, bz) = t
  let a = vector.Vec3(ax, ay, az)
  let b = vector.Vec3(bx, by, bz)
  should.be_true(close(vector.dot(a, b), vector.dot(b, a), 1.0e-9))
}

pub fn qcheck_vec3_cross_perpendicular_test() {
  // (a × b) · a = 0 always
  use t <- qcheck.given(qcheck.tuple6(
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
    qcheck.bounded_float(-10.0, 10.0),
  ))
  let #(ax, ay, az, bx, by, bz) = t
  let a = vector.Vec3(ax, ay, az)
  let b = vector.Vec3(bx, by, bz)
  let c = vector.cross(a, b)
  should.be_true(close(vector.dot(c, a), 0.0, 1.0e-6))
  should.be_true(close(vector.dot(c, b), 0.0, 1.0e-6))
}

// ============================================================================
// quaternion: rotation preserves length
// ============================================================================

pub fn qcheck_quaternion_rotation_preserves_length_test() {
  use t <- qcheck.given(qcheck.tuple4(
    qcheck.bounded_float(-5.0, 5.0),
    qcheck.bounded_float(-5.0, 5.0),
    qcheck.bounded_float(-5.0, 5.0),
    qcheck.bounded_float(-3.14, 3.14),
  ))
  let #(vx, vy, vz, theta) = t
  let v = vector.Vec3(vx, vy, vz)
  let axis = vector.Vec3(1.0, 0.0, 0.0)
  let q = quaternion.from_axis_angle(axis, theta)
  let rotated = quaternion.rotate(q, v)
  should.be_true(close(vector.length(rotated), vector.length(v), 1.0e-9))
}

pub fn qcheck_quaternion_inverse_test() {
  // q · q⁻¹ = identity for unit quaternions.
  use t <- qcheck.given(qcheck.tuple4(
    qcheck.bounded_float(0.1, 5.0),
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(-2.0, 2.0),
  ))
  let #(theta, ax, ay, az) = t
  let q = quaternion.from_axis_angle(vector.Vec3(ax, ay, az), theta)
  let qi = quaternion.inverse(q)
  let prod = quaternion.mul(q, qi)
  let identity = quaternion.identity()
  should.be_true(quaternion.is_close(prod, identity, 1.0e-9))
}

// ============================================================================
// complex: |z|² = z · conj(z)
// ============================================================================

pub fn qcheck_complex_magnitude_squared_test() {
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(-100.0, 100.0),
    qcheck.bounded_float(-100.0, 100.0),
  ))
  let #(re, im) = pair
  let z = complex.Complex(re: re, im: im)
  let m2 = complex.magnitude_squared(z)
  let prod = complex.mul(z, complex.conjugate(z))
  // z · conj(z) has zero imaginary part by definition.
  should.be_true(close(prod.re, m2, 1.0e-6))
  should.be_true(close(prod.im, 0.0, 1.0e-6))
}

pub fn qcheck_complex_exp_log_inverse_test() {
  // log(exp(z)) ≈ z (mod 2πi, but for small im this holds)
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(-3.0, 3.0),
    qcheck.bounded_float(-3.0, 3.0),
  ))
  let #(re, im) = pair
  let z = complex.Complex(re: re, im: im)
  let round_trip = complex.log(complex.exp(z))
  should.be_true(close(round_trip.re, z.re, 1.0e-9))
  should.be_true(close(round_trip.im, z.im, 1.0e-9))
}

// ============================================================================
// precision moments: variance is non-negative
// ============================================================================

pub fn qcheck_moments_variance_non_negative_test() {
  use xs <- qcheck.given(qcheck.list_from(qcheck.bounded_float(-100.0, 100.0)))
  // Skip empty list — variance is genuinely undefined there.
  case xs {
    [] -> should.be_true(True)
    _ -> {
      let m = precision.moments_from_list(xs)
      let assert Ok(v) = precision.moments_variance(m)
      should.be_true(v >=. 0.0)
    }
  }
}

pub fn qcheck_moments_combine_associativity_test() {
  // Combining two lists is the same as combining their accumulators.
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.list_from(qcheck.bounded_float(-50.0, 50.0)),
    qcheck.list_from(qcheck.bounded_float(-50.0, 50.0)),
  ))
  let #(left, right) = pair
  // Skip when both halves are empty (variance undefined).
  case left, right {
    [], [] -> should.be_true(True)
    _, _ -> {
      let merged_list = list.append(left, right)
      let m_combined =
        precision.moments_combine(
          precision.moments_from_list(left),
          precision.moments_from_list(right),
        )
      let m_direct = precision.moments_from_list(merged_list)
      let assert Ok(v_combined) = precision.moments_variance(m_combined)
      let assert Ok(v_direct) = precision.moments_variance(m_direct)
      should.be_true(close(v_combined, v_direct, 1.0e-6))
    }
  }
}

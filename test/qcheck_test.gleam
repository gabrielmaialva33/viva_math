//// Property-based tests with random input generation via `qcheck`.
////
//// Each property runs 100 randomised cases; failures shrink to a minimal
//// counterexample. Verifies invariants that must hold for *all* inputs in
//// the relevant domain.

import gleam/float
import gleam/list
import gleeunit/should
import qcheck
import viva_math/common
import viva_math/complex
import viva_math/entropy
import viva_math/free_energy
import viva_math/ou
import viva_math/precision
import viva_math/quaternion
import viva_math/scalar
import viva_math/transport
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

// ============================================================================
// closed forms: additional invariants
// ============================================================================

pub fn property_clamp_idempotent_test() {
  use t <- qcheck.given(qcheck.tuple4(
    qcheck.bounded_float(-100.0, 100.0),
    qcheck.bounded_float(-100.0, 100.0),
    qcheck.bounded_float(0.0, 100.0),
    qcheck.bounded_float(0.0, 1.0),
  ))
  let #(x, lo, width, _) = t
  let hi = lo +. width
  let once = common.clamp(x, lo, hi)
  let twice = common.clamp(once, lo, hi)
  should.be_true(twice == once)
}

pub fn property_sigmoid_in_unit_interval_test() {
  use x <- qcheck.given(qcheck.bounded_float(-100.0, 100.0))
  let y = scalar.sigmoid(x)
  should.be_true(y >=. 0.0 && y <=. 1.0)
}

pub fn property_sin_cos_identity_test() {
  use x <- qcheck.given(qcheck.bounded_float(-100.0, 100.0))
  let s = scalar.sin(x)
  let c = scalar.cos(x)
  should.be_true(close(s *. s +. c *. c, 1.0, 1.0e-9))
}

pub fn property_softmax_sums_to_one_test() {
  use t <- qcheck.given(qcheck.tuple4(
    qcheck.bounded_float(-100.0, 100.0),
    qcheck.bounded_float(-100.0, 100.0),
    qcheck.bounded_float(-100.0, 100.0),
    qcheck.bounded_float(-100.0, 100.0),
  ))
  let #(a, b, c, d) = t
  case common.softmax([a, b, c, d]) {
    Ok(probs) -> {
      let sum = list.fold(probs, 0.0, fn(acc, x) { acc +. x })
      should.be_true(close(sum, 1.0, 1.0e-12))
    }
    Error(_) -> should.be_true(False)
  }
}

pub fn property_js_symmetric_test() {
  use t <- qcheck.given(qcheck.tuple4(
    qcheck.bounded_float(0.01, 100.0),
    qcheck.bounded_float(0.01, 100.0),
    qcheck.bounded_float(0.01, 100.0),
    qcheck.bounded_float(0.01, 100.0),
  ))
  let #(a, b, c, d) = t
  let p_total = a +. b
  let q_total = c +. d
  let p = [a /. p_total, b /. p_total]
  let q = [c /. q_total, d /. q_total]

  case entropy.jensen_shannon(p, q), entropy.jensen_shannon(q, p) {
    Ok(left), Ok(right) -> should.be_true(close(left, right, 1.0e-12))
    _, _ -> should.be_true(False)
  }
}

pub fn property_kl_self_zero_test() {
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(0.01, 100.0),
    qcheck.bounded_float(0.01, 100.0),
  ))
  let #(a, b) = pair
  let total = a +. b
  let p = [a /. total, b /. total]

  case entropy.kl_divergence(p, p) {
    Ok(kl) -> should.be_true(close(kl, 0.0, 1.0e-12))
    Error(_) -> should.be_true(False)
  }
}

pub fn property_shannon_nonneg_test() {
  use pair <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(0.01, 100.0),
    qcheck.bounded_float(0.01, 100.0),
  ))
  let #(a, b) = pair
  let total = a +. b
  let probs = [a /. total, b /. total]
  should.be_true(entropy.shannon(probs) >=. 0.0)
}

pub fn property_erf_odd_test() {
  use x <- qcheck.given(qcheck.bounded_float(-10.0, 10.0))
  should.be_true(close(scalar.erf(0.0 -. x), 0.0 -. scalar.erf(x), 1.0e-12))
}

pub fn property_activations_zero_at_zero_test() {
  should.be_true(close(scalar.gelu(0.0), 0.0, 1.0e-12))
  should.be_true(close(scalar.silu(0.0), 0.0, 1.0e-12))
}

// ============================================================================
// OU dynamics — closed-form invariants
// ============================================================================

// mean_at(t=0) = x_0 for any valid params.
pub fn property_ou_mean_at_zero_is_x0_test() {
  use t <- qcheck.given(qcheck.tuple3(
    qcheck.bounded_float(0.01, 5.0),
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(-2.0, 2.0),
  ))
  let #(theta, mu, x0) = t
  let params = ou.OUParams1D(theta: theta, mu: mu, sigma: 0.5)
  should.be_true(close(ou.mean_at(params, x0, 0.0), x0, 1.0e-12))
}

// stationary_variance ≥ 0.
pub fn property_ou_stationary_variance_nonneg_test() {
  use t <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(0.01, 5.0),
    qcheck.bounded_float(0.0, 2.0),
  ))
  let #(theta, sigma) = t
  let params = ou.OUParams1D(theta: theta, mu: 0.0, sigma: sigma)
  should.be_true(ou.stationary_variance(params) >=. 0.0)
}

// autocovariance(0) = stationary_variance.
pub fn property_ou_autocov_zero_equals_stationary_test() {
  use t <- qcheck.given(qcheck.tuple2(
    qcheck.bounded_float(0.01, 5.0),
    qcheck.bounded_float(0.01, 2.0),
  ))
  let #(theta, sigma) = t
  let params = ou.OUParams1D(theta: theta, mu: 0.0, sigma: sigma)
  let v = ou.stationary_variance(params)
  let c = ou.autocovariance(params, 0.0)
  should.be_true(close(v, c, 1.0e-12))
}

// |mean_at(t) − μ| ≤ |x0 − μ| (monotonic mean reversion).
pub fn property_ou_mean_reverts_monotonically_test() {
  use t <- qcheck.given(qcheck.tuple3(
    qcheck.bounded_float(0.01, 5.0),
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(0.0, 10.0),
  ))
  let #(theta, x0, t_val) = t
  let mu = 0.5
  let params = ou.OUParams1D(theta: theta, mu: mu, sigma: 0.1)
  let m = ou.mean_at(params, x0, t_val)
  let initial_gap = float.absolute_value(x0 -. mu)
  let current_gap = float.absolute_value(m -. mu)
  should.be_true(current_gap <=. initial_gap +. 1.0e-9)
}

// ============================================================================
// Variational FE — ELBO bound + KL non-negativity
// ============================================================================

// scalar_gaussian_kl ≥ 0.
pub fn property_scalar_gaussian_kl_nonneg_test() {
  use t <- qcheck.given(qcheck.tuple3(
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(0.1, 5.0),
  ))
  let #(mu_q, mu_p, var_common) = t
  let kl = free_energy.scalar_gaussian_kl(mu_q, var_common, mu_p, var_common)
  should.be_true(kl >=. 0.0 -. 1.0e-12)
}

// ELBO ≤ log p(x) (variational bound — Jensen's inequality).
pub fn property_elbo_bound_test() {
  use t <- qcheck.given(qcheck.tuple3(
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(-2.0, 2.0),
    qcheck.bounded_float(0.2, 3.0),
  ))
  let #(obs, q_mean, q_var) = t
  let prior_mean = 0.0
  let prior_var = 1.0
  let lik_var = 1.0
  let e = free_energy.elbo(obs, q_mean, q_var, prior_mean, prior_var, lik_var)
  let logp =
    free_energy.log_evidence_gaussian(obs, prior_mean, prior_var, lik_var)
  // Floating point slack: bound deve valer com tolerância pequena.
  should.be_true(e.total <=. logp +. 1.0e-9)
}

// ============================================================================
// Wasserstein — symmetry + self-zero
// ============================================================================

pub fn property_wasserstein_self_zero_test() {
  use t <- qcheck.given(qcheck.tuple3(
    qcheck.bounded_float(-5.0, 5.0),
    qcheck.bounded_float(-5.0, 5.0),
    qcheck.bounded_float(-5.0, 5.0),
  ))
  let #(a, b, c) = t
  let xs = [a, b, c]
  let assert Ok(d) = transport.wasserstein_1_empirical(xs, xs)
  should.be_true(close(d, 0.0, 1.0e-9))
}

pub fn property_wasserstein_symmetric_test() {
  use t <- qcheck.given(qcheck.tuple3(
    qcheck.bounded_float(-3.0, 3.0),
    qcheck.bounded_float(-3.0, 3.0),
    qcheck.bounded_float(-3.0, 3.0),
  ))
  let #(a, b, c) = t
  let p = [a, b]
  let q = [b, c]
  let assert Ok(d_pq) = transport.wasserstein_1_empirical(p, q)
  let assert Ok(d_qp) = transport.wasserstein_1_empirical(q, p)
  should.be_true(close(d_pq, d_qp, 1.0e-12))
}

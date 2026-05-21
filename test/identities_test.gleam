//// Algebraic identities — universal laws that must hold for all valid inputs.
////
//// These tests defend against subtle regressions that golden-value tests
//// miss: round-trips (`exp(ln(x)) = x`), recurrences (`Γ(x+1) = x·Γ(x)`),
//// translation invariance (`softmax(x+c) = softmax(x)`), and limit
//// behaviour (`Var_OU(t→∞) = σ²/(2θ)`).
////
//// Anchored on the Codex GPT-5.5 god-audit (1.2.102) which flagged each
//// of these identities as unexercised.

import gleam/list
import gleeunit/should
import test_support.{is_close, is_close_rel}
import viva_math/common
import viva_math/cusp
import viva_math/matrix
import viva_math/ou
import viva_math/random
import viva_math/scalar
import viva_math/special

// ============================================================================
// Scalar — transcendental round-trips
// ============================================================================

// exp(ln(x)) = x  for x > 0.
pub fn id_exp_ln_round_trip_test() {
  let values = [0.5, 1.0, 2.0, 10.0, 100.0]
  list.each(values, fn(x) {
    let assert Ok(l) = scalar.logarithm(x)
    let r = scalar.exp(l)
    should.be_true(is_close_rel(r, x, 1.0e-13))
  })
}

// ln(exp(x)) = x  for moderate x (away from overflow).
pub fn id_ln_exp_round_trip_test() {
  let values = [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
  list.each(values, fn(x) {
    let assert Ok(l) = scalar.logarithm(scalar.exp(x))
    should.be_true(is_close(l, x, 1.0e-12))
  })
}

// sqrt(x)² = x  for x ≥ 0.
pub fn id_sqrt_squared_test() {
  let values = [0.0, 0.25, 1.0, 2.0, 9.0, 100.0, 1.0e6]
  list.each(values, fn(x) {
    let assert Ok(s) = scalar.square_root(x)
    should.be_true(is_close_rel(s *. s, x, 1.0e-13))
  })
}

// (cbrt(x))³ = x  for any real x (including negatives).
pub fn id_cbrt_cubed_test() {
  let values = [-8.0, -1.0, 0.0, 1.0, 8.0, 27.0]
  list.each(values, fn(x) {
    let assert Ok(c) = scalar.cube_root(x)
    let cubed = c *. c *. c
    should.be_true(is_close(cubed, x, 1.0e-12))
  })
}

// sin²(x) + cos²(x) = 1 across a representative sweep.
pub fn id_sin_cos_pythagorean_test() {
  let samples = [-3.14, -1.0, -0.5, 0.0, 0.5, 1.0, 3.14, 6.28]
  list.each(samples, fn(x) {
    let s = scalar.sin(x)
    let c = scalar.cos(x)
    should.be_true(is_close(s *. s +. c *. c, 1.0, 1.0e-13))
  })
}

// ============================================================================
// Common — softmax translation invariance
// ============================================================================

// `softmax(x + c) = softmax(x)` (the stable-softmax invariant) — adding a
// constant to every logit must not change the resulting distribution.
pub fn id_softmax_translation_invariance_test() {
  let xs = [-2.0, 0.5, 1.5]
  let shifted = list.map(xs, fn(x) { x +. 7.0 })
  let assert Ok(a) = common.softmax(xs)
  let assert Ok(b) = common.softmax(shifted)
  list.each(list.zip(a, b), fn(pair) {
    let #(ai, bi) = pair
    should.be_true(is_close(ai, bi, 1.0e-13))
  })
}

// ============================================================================
// Special — gamma + digamma recurrences
// ============================================================================

// Γ(x+1) = x·Γ(x) — fundamental recurrence (anchors all integer factorials).
pub fn id_gamma_recurrence_test() {
  let xs = [0.7, 1.3, 2.5, 4.2, 10.5]
  list.each(xs, fn(x) {
    let lhs = special.gamma(x +. 1.0)
    let rhs = x *. special.gamma(x)
    should.be_true(is_close_rel(lhs, rhs, 1.0e-10))
  })
}

// ψ(x+1) = ψ(x) + 1/x — digamma recurrence.
pub fn id_digamma_recurrence_test() {
  let xs = [0.5, 1.0, 2.0, 3.7, 10.0]
  list.each(xs, fn(x) {
    let lhs = special.digamma(x +. 1.0)
    let rhs = special.digamma(x) +. 1.0 /. x
    should.be_true(is_close(lhs, rhs, 1.0e-7))
  })
}

// lbeta(x, y) = lgamma(x) + lgamma(y) − lgamma(x+y).
pub fn id_lbeta_decomposition_test() {
  let pairs = [#(1.0, 1.0), #(2.0, 3.0), #(0.5, 2.5), #(4.0, 4.0)]
  list.each(pairs, fn(p) {
    let #(x, y) = p
    let lhs = special.lbeta(x, y)
    let rhs = special.lgamma(x) +. special.lgamma(y) -. special.lgamma(x +. y)
    should.be_true(is_close(lhs, rhs, 1.0e-12))
  })
}

// ============================================================================
// OU — limit behaviour + zero-time identity
// ============================================================================

// At t → ∞ the variance approaches the stationary variance.
pub fn id_ou_variance_converges_to_stationary_test() {
  let params = ou.OUParams1D(theta: 1.0, mu: 0.0, sigma: 1.0)
  let v_inf = ou.variance_at(params, 0.0, 50.0)
  let v_stationary = ou.stationary_variance(params)
  should.be_true(is_close(v_inf, v_stationary, 1.0e-12))
}

// step(dt=0) must be a no-op on the state (modulo no noise consumed of the seed).
pub fn id_ou_step_dt_zero_is_identity_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 1.0, sigma: 0.7)
  let seed = random.from_int(42)
  let #(x_next, _) = ou.step(params, 0.3, 0.0, seed)
  should.be_true(is_close(x_next, 0.3, 1.0e-15))
}

// mean_at(t=0) = x0 regardless of params.
pub fn id_ou_mean_at_t_zero_test() {
  let params = ou.OUParams1D(theta: 2.5, mu: -0.7, sigma: 1.0)
  should.be_true(is_close(ou.mean_at(params, 0.42, 0.0), 0.42, 1.0e-15))
}

// ============================================================================
// Cusp — gradient is the derivative of the potential
// ============================================================================

// `cusp.gradient(params, x) = dV/dx` computed via central finite differences
// on `cusp.potential`. The closed form must match FD to ~h².
pub fn id_cusp_gradient_matches_potential_derivative_test() {
  let params = cusp.CuspParams(alpha: -1.0, beta: 0.3)
  let points = [-0.8, -0.3, 0.1, 0.5, 1.2]
  let h = 1.0e-5
  list.each(points, fn(x) {
    let v_plus = cusp.potential(x +. h, params)
    let v_minus = cusp.potential(x -. h, params)
    let fd = { v_plus -. v_minus } /. { 2.0 *. h }
    let analytical = cusp.gradient(x, params)
    should.be_true(is_close(fd, analytical, 1.0e-7))
  })
}

// ============================================================================
// Matrix — transpose preserves determinant
// ============================================================================

// det(Aᵀ) = det(A) for any square matrix. Uses Mat2 since `matn_determinant`
// is not exposed (matrix.gleam only ships `mat2_determinant` /
// `mat3_determinant` direct closed forms).
pub fn id_matrix_det_transpose_invariance_test() {
  let a = matrix.Mat2(1.0, 2.0, 3.0, 4.0)
  let at = matrix.mat2_transpose(a)
  let det_a = matrix.mat2_determinant(a)
  let det_at = matrix.mat2_determinant(at)
  should.be_true(is_close(det_a, det_at, 1.0e-15))
}

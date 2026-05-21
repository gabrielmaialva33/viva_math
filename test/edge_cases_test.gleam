//// Edge cases — `x = 0`, `x = -0.0`, large `x`, subnormals, domain boundaries.
////
//// Anchored on N1630 (WG14) and CPython's `test_math.py`. These tests
//// exercise the boundaries where a math library is most likely to silently
//// degrade — pole errors, cancellation, overflow, underflow.

import gleeunit/should
import test_support.{is_close, is_close_rel}
import viva_math/ou
import viva_math/scalar
import viva_math/vector.{Vec3}

// ============================================================================
// `square_root` / `cube_root` / `logarithm` domain boundaries
// ============================================================================

pub fn edge_square_root_zero_test() {
  let assert Ok(s) = scalar.square_root(0.0)
  should.be_true(is_close(s, 0.0, 1.0e-15))
}

pub fn edge_square_root_negative_rejects_test() {
  scalar.square_root(-1.0e-300) |> should.equal(Error(Nil))
  scalar.square_root(-1.0) |> should.equal(Error(Nil))
}

pub fn edge_cube_root_negative_test() {
  // Cube root is total over the reals — `∛(-x) = -∛x`.
  let assert Ok(c) = scalar.cube_root(-8.0)
  should.be_true(is_close(c, -2.0, 1.0e-12))
}

pub fn edge_cube_root_zero_test() {
  let assert Ok(c) = scalar.cube_root(0.0)
  should.be_true(is_close(c, 0.0, 1.0e-15))
}

pub fn edge_logarithm_at_one_test() {
  // ln(1) = 0 exactly.
  let assert Ok(l) = scalar.logarithm(1.0)
  should.be_true(is_close(l, 0.0, 1.0e-15))
}

pub fn edge_logarithm_zero_rejects_test() {
  scalar.logarithm(0.0) |> should.equal(Error(Nil))
}

pub fn edge_logarithm_negative_rejects_test() {
  scalar.logarithm(-0.5) |> should.equal(Error(Nil))
  scalar.logarithm(-1.0e-300) |> should.equal(Error(Nil))
}

pub fn edge_logarithm_2_powers_test() {
  // log₂(2ⁿ) = n exactly for small n.
  let assert Ok(l1) = scalar.logarithm_2(2.0)
  let assert Ok(l2) = scalar.logarithm_2(4.0)
  let assert Ok(l3) = scalar.logarithm_2(8.0)
  should.be_true(is_close(l1, 1.0, 1.0e-15))
  should.be_true(is_close(l2, 2.0, 1.0e-15))
  should.be_true(is_close(l3, 3.0, 1.0e-15))
}

pub fn edge_logarithm_10_powers_test() {
  // log₁₀(10ⁿ) = n.
  let assert Ok(l1) = scalar.logarithm_10(10.0)
  let assert Ok(l2) = scalar.logarithm_10(100.0)
  should.be_true(is_close(l1, 1.0, 1.0e-15))
  should.be_true(is_close(l2, 2.0, 1.0e-14))
}

pub fn edge_nth_root_n_one_test() {
  // x^(1/1) = x — identity for n=1.
  let assert Ok(r) = scalar.nth_root(7.5, 1)
  should.be_true(is_close(r, 7.5, 1.0e-15))
}

pub fn edge_nth_root_n_zero_rejects_test() {
  scalar.nth_root(2.0, 0) |> should.equal(Error(Nil))
  scalar.nth_root(2.0, -1) |> should.equal(Error(Nil))
}

pub fn edge_nth_root_even_negative_rejects_test() {
  scalar.nth_root(-1.0, 4) |> should.equal(Error(Nil))
}

pub fn edge_nth_root_odd_negative_test() {
  // 5th root of -32 is -2.
  let assert Ok(r) = scalar.nth_root(-32.0, 5)
  should.be_true(is_close(r, -2.0, 1.0e-12))
}

// ============================================================================
// `expm1` cancellation — `e^x − 1` for tiny x
// ============================================================================

// `expm1(x)/x → 1` as x → 0. With naive `exp(x) - 1` this is catastrophic.
pub fn edge_expm1_tiny_relative_accuracy_test() {
  let x = 1.0e-10
  let ratio = scalar.expm1(x) /. x
  should.be_true(is_close(ratio, 1.0, 1.0e-9))
}

pub fn edge_expm1_at_zero_test() {
  // expm1(0) = 0 exactly.
  should.be_true(is_close(scalar.expm1(0.0), 0.0, 1.0e-15))
}

// ============================================================================
// OU — vec3 path coverage + extreme regimes
// ============================================================================

pub fn edge_ou_is_valid_vec3_test() {
  let good =
    ou.OUParamsVec3(
      theta: Vec3(1.0, 2.0, 3.0),
      mu: Vec3(0.0, 0.0, 0.0),
      sigma: Vec3(0.1, 0.2, 0.3),
    )
  ou.is_valid_vec3(good) |> should.be_true

  // y-axis θ = 0 violates "every theta strictly positive".
  let bad_theta =
    ou.OUParamsVec3(
      theta: Vec3(1.0, 0.0, 3.0),
      mu: Vec3(0.0, 0.0, 0.0),
      sigma: Vec3(0.1, 0.2, 0.3),
    )
  ou.is_valid_vec3(bad_theta) |> should.be_false

  // Negative σ on y violates "every sigma non-negative".
  let bad_sigma =
    ou.OUParamsVec3(
      theta: Vec3(1.0, 1.0, 1.0),
      mu: Vec3(0.0, 0.0, 0.0),
      sigma: Vec3(0.1, -0.1, 0.3),
    )
  ou.is_valid_vec3(bad_sigma) |> should.be_false
}

pub fn edge_ou_stationary_std_test() {
  // σ²/(2θ) = 1, so std = 1.
  let p = ou.OUParams1D(theta: 0.5, mu: 0.0, sigma: 1.0)
  should.be_true(is_close(ou.stationary_std(p), 1.0, 1.0e-12))
}

pub fn edge_ou_variance_at_huge_t_test() {
  // For very large t the variance saturates at σ²/(2θ).
  let p = ou.OUParams1D(theta: 2.0, mu: 0.0, sigma: 3.0)
  let v = ou.variance_at(p, 0.0, 1.0e6)
  let stat = ou.stationary_variance(p)
  should.be_true(is_close_rel(v, stat, 1.0e-12))
}

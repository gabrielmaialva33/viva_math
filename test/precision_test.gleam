//// High-precision tests with golden values from external references.
////
//// Tolerances:
////   ULTRA  = 1.0e-12  ← exact identity / closed form
////   TIGHT  = 1.0e-9   ← well-conditioned numerical computations
////   LOOSE  = 1.0e-6   ← high-order numerical methods (RK4, Simpson)
////
//// Golden values come from:
////   - Erlang :math (which delegates to the platform libm)
////   - Closed-form analytical expressions
////   - Wolfram Alpha / scipy.special for reference

import gleam/float
import gleam/list
import gleeunit/should
import viva_math/calculus
import viva_math/constants
import viva_math/distributions
import viva_math/entropy
import viva_math/matrix
import viva_math/ode
import viva_math/random
import viva_math/scalar
import viva_math/scheduler
import viva_math/statistics

const ultra: Float = 1.0e-12

const tight: Float = 1.0e-9

const loose: Float = 1.0e-6

fn approx(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

// ============================================================================
// scalar.erf - golden values from scipy.special.erf
// ============================================================================

pub fn precision_erf_zero_test() {
  should.be_true(approx(scalar.erf(0.0), 0.0, ultra))
}

pub fn precision_erf_half_test() {
  // erf(0.5) = 0.5204998778130465
  should.be_true(approx(scalar.erf(0.5), 0.5204998778130465, ultra))
}

pub fn precision_erf_one_test() {
  should.be_true(approx(scalar.erf(1.0), 0.8427007929497149, ultra))
}

pub fn precision_erf_two_test() {
  should.be_true(approx(scalar.erf(2.0), 0.9953222650189527, ultra))
}

pub fn precision_erf_is_odd_test() {
  // erf is odd: erf(-x) = -erf(x)
  should.be_true(approx(scalar.erf(-1.0), 0.0 -. scalar.erf(1.0), ultra))
  should.be_true(approx(scalar.erf(-2.5), 0.0 -. scalar.erf(2.5), ultra))
}

pub fn precision_erfc_complement_test() {
  // erf(x) + erfc(x) = 1 exactly
  let xs = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.erf(x) +. scalar.erfc(x), 1.0, ultra))
  })
}

pub fn precision_erfc_tail_test() {
  // erfc(5) ≈ 1.537459794428e-12 — should not underflow.
  let result = scalar.erfc(5.0)
  should.be_true(result >. 0.0)
  should.be_true(approx(result, 1.5374597944280348e-12, 1.0e-20))
}

// ============================================================================
// scalar activations - golden values
// ============================================================================

pub fn precision_gelu_at_one_test() {
  // gelu(1) exact = 0.5 * (1 + erf(1/√2)) ≈ 0.8413447460685429
  should.be_true(approx(scalar.gelu(1.0), 0.8413447460685429, ultra))
}

pub fn precision_gelu_at_two_test() {
  // gelu(2) ≈ 1.9544997361036416
  should.be_true(approx(scalar.gelu(2.0), 1.9544997361036416, ultra))
}

pub fn precision_gelu_negative_test() {
  // gelu(-3) ≈ -0.00404969...
  should.be_true(approx(scalar.gelu(-3.0), -0.00404969409489031, ultra))
}

pub fn precision_silu_one_test() {
  should.be_true(approx(scalar.silu(1.0), 0.7310585786300049, ultra))
}

pub fn precision_silu_is_self_gating_test() {
  // silu(x) = x · sigmoid(x)
  let xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.silu(x), x *. scalar.sigmoid(x), ultra))
  })
}

pub fn precision_mish_at_one_test() {
  // mish(1) = 1 · tanh(softplus(1)) ≈ 0.8650983882673103
  should.be_true(approx(scalar.mish(1.0), 0.8650983882673103, ultra))
}

pub fn precision_softplus_at_zero_test() {
  // softplus(0) = ln(2) exactly
  should.be_true(approx(scalar.softplus(0.0), constants.ln_2, ultra))
}

pub fn precision_softplus_at_one_test() {
  // softplus(1) = ln(1 + e) ≈ 1.3132616875182228
  should.be_true(approx(scalar.softplus(1.0), 1.3132616875182228, ultra))
}

pub fn precision_softplus_extreme_negative_test() {
  // softplus(-10) ≈ 4.5398899e-5 (does not underflow to 0)
  let r = scalar.softplus(-10.0)
  should.be_true(r >. 0.0)
  should.be_true(approx(r, 4.5398899216870535e-5, 1.0e-15))
}

pub fn precision_softplus_extreme_positive_test() {
  // softplus(700) ≈ 700 (no overflow)
  let r = scalar.softplus(700.0)
  should.be_true(approx(r, 700.0, 1.0e-9))
}

pub fn precision_sigmoid_symmetry_test() {
  // σ(x) + σ(-x) = 1 for all x
  let xs = [-100.0, -1.0, -0.1, 0.0, 0.1, 1.0, 100.0]
  list.each(xs, fn(x) {
    should.be_true(approx(
      scalar.sigmoid(x) +. scalar.sigmoid(0.0 -. x),
      1.0,
      ultra,
    ))
  })
}

pub fn precision_tanh_consistency_test() {
  // tanh(x) = (e^x - e^-x)/(e^x + e^-x); compare with our tanh on safe range.
  let xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
  list.each(xs, fn(x) {
    let manual =
      { scalar.exp(x) -. scalar.exp(0.0 -. x) }
      /. { scalar.exp(x) +. scalar.exp(0.0 -. x) }
    should.be_true(approx(scalar.tanh(x), manual, tight))
  })
}

// ============================================================================
// logsumexp / logaddexp - stability under large values
// ============================================================================

pub fn precision_logsumexp_pair_test() {
  // logsumexp([0, 0]) = ln(2)
  should.be_true(approx(scalar.logsumexp([0.0, 0.0]), constants.ln_2, ultra))
}

pub fn precision_logsumexp_triple_test() {
  // logsumexp([1,2,3]) ≈ 3.4076059644443806
  should.be_true(approx(
    scalar.logsumexp([1.0, 2.0, 3.0]),
    3.4076059644443806,
    ultra,
  ))
}

pub fn precision_logsumexp_no_overflow_test() {
  // logsumexp([1000, 1000]) = 1000 + ln(2) — must not overflow.
  let r = scalar.logsumexp([1000.0, 1000.0])
  should.be_true(approx(r, 1000.0 +. constants.ln_2, ultra))
}

pub fn precision_logsumexp_no_underflow_test() {
  // logsumexp([-1000, -1000]) = -1000 + ln(2) — must not underflow.
  let r = scalar.logsumexp([-1000.0, -1000.0])
  should.be_true(approx(r, 0.0 -. 1000.0 +. constants.ln_2, ultra))
}

pub fn precision_logaddexp_no_nan_at_infinity_test() {
  // logaddexp(a, a) = a + ln(2) — no NaN from ∞-∞.
  let r = scalar.logaddexp(1.0e300, 1.0e300)
  should.be_true(approx(r, 1.0e300 +. constants.ln_2, 1.0e290))
}

pub fn precision_logaddexp_equal_args_test() {
  // logaddexp(x, x) = x + ln(2)
  let xs = [-500.0, 0.0, 500.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.logaddexp(x, x), x +. constants.ln_2, ultra))
  })
}

// ============================================================================
// hypot - no overflow at extreme scales
// ============================================================================

pub fn precision_hypot_pythagorean_test() {
  should.be_true(approx(scalar.hypot(3.0, 4.0), 5.0, ultra))
}

pub fn precision_hypot_no_underflow_test() {
  // hypot(1e-300, 1e-300) = √2 · 1e-300; direct sqrt(x²+y²) would underflow.
  let r = scalar.hypot(1.0e-300, 1.0e-300)
  should.be_true(r >. 0.0)
  should.be_true(approx(r, constants.sqrt_2 *. 1.0e-300, 1.0e-310))
}

pub fn precision_hypot_no_overflow_test() {
  // hypot(1e200, 1e200) = √2 · 1e200; direct sqrt(x²+y²) would overflow.
  let r = scalar.hypot(1.0e200, 1.0e200)
  let expected = constants.sqrt_2 *. 1.0e200
  should.be_true(float.absolute_value(r -. expected) /. expected <=. tight)
}

// ============================================================================
// log1p / expm1 - small argument precision
// ============================================================================

pub fn precision_log1p_small_test() {
  // log1p(1e-15) ≈ 1e-15 (direct ln(1 + 1e-15) loses precision)
  let r = scalar.log1p(1.0e-15)
  should.be_true(approx(r, 1.0e-15, 1.0e-25))
}

pub fn precision_log1p_one_test() {
  should.be_true(approx(scalar.log1p(1.0), constants.ln_2, ultra))
}

pub fn precision_expm1_small_test() {
  // expm1(1e-15) ≈ 1e-15 (direct exp(x)-1 loses precision)
  let r = scalar.expm1(1.0e-15)
  should.be_true(approx(r, 1.0e-15, 1.0e-25))
}

pub fn precision_expm1_one_test() {
  // expm1(1) = e - 1
  should.be_true(approx(scalar.expm1(1.0), constants.e -. 1.0, tight))
}

// ============================================================================
// Constants invariants
// ============================================================================

pub fn precision_sqrt_2_squared_test() {
  should.be_true(approx(constants.sqrt_2 *. constants.sqrt_2, 2.0, ultra))
}

pub fn precision_tau_test() {
  should.be_true(approx(constants.tau, 2.0 *. constants.pi, ultra))
}

pub fn precision_inv_sqrt_2pi_test() {
  // 1/√(2π) · √(2π) = 1
  should.be_true(approx(
    constants.inv_sqrt_2pi *. constants.sqrt_2pi,
    1.0,
    ultra,
  ))
}

pub fn precision_log_e_test() {
  // ln(e) = 1
  should.be_true(approx(scalar.ln(constants.e), 1.0, ultra))
}

// ============================================================================
// statistics - golden values
// ============================================================================

pub fn precision_mean_simple_test() {
  let assert Ok(m) = statistics.mean([1.0, 2.0, 3.0, 4.0, 5.0])
  should.be_true(approx(m, 3.0, ultra))
}

pub fn precision_variance_test() {
  // Population variance of [2,4,4,4,5,5,7,9] = 4
  let assert Ok(v) =
    statistics.variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  should.be_true(approx(v, 4.0, ultra))
}

pub fn precision_sample_variance_test() {
  // Sample variance of [1,2,3,4,5] = 2.5
  let assert Ok(v) = statistics.sample_variance([1.0, 2.0, 3.0, 4.0, 5.0])
  should.be_true(approx(v, 2.5, ultra))
}

pub fn precision_stddev_test() {
  // stddev of [2,4,4,4,5,5,7,9] = 2
  let assert Ok(s) = statistics.stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  should.be_true(approx(s, 2.0, ultra))
}

pub fn precision_median_even_test() {
  let assert Ok(m) = statistics.median([1.0, 2.0, 3.0, 4.0])
  should.be_true(approx(m, 2.5, ultra))
}

pub fn precision_median_odd_test() {
  let assert Ok(m) = statistics.median([7.0, 1.0, 4.0, 3.0, 5.0])
  should.be_true(approx(m, 4.0, ultra))
}

pub fn precision_percentile_test() {
  // numpy.percentile([1..5], 25, method='linear') = 2.0
  let assert Ok(q) = statistics.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25)
  should.be_true(approx(q, 2.0, ultra))
}

pub fn precision_pearson_perfect_test() {
  let xs = [1.0, 2.0, 3.0, 4.0, 5.0]
  let ys = [2.0, 4.0, 6.0, 8.0, 10.0]
  // Perfect linear correlation = 1
  let assert Ok(r) = statistics.pearson(xs, ys)
  should.be_true(approx(r, 1.0, tight))
}

pub fn precision_skewness_symmetric_test() {
  // Symmetric distribution has skewness = 0
  let assert Ok(s) = statistics.skewness([-2.0, -1.0, 0.0, 1.0, 2.0])
  should.be_true(approx(s, 0.0, ultra))
}

pub fn precision_geometric_mean_test() {
  // GM([1, 2, 4, 8, 16]) = 4
  let assert Ok(g) = statistics.geometric_mean([1.0, 2.0, 4.0, 8.0, 16.0])
  should.be_true(approx(g, 4.0, tight))
}

pub fn precision_harmonic_mean_test() {
  // HM([1, 2, 4]) = 3 / (1 + 1/2 + 1/4) = 3 / 1.75 ≈ 1.7142857
  let assert Ok(h) = statistics.harmonic_mean([1.0, 2.0, 4.0])
  should.be_true(approx(h, 12.0 /. 7.0, tight))
}

// ============================================================================
// distributions - golden values
// ============================================================================

pub fn precision_gaussian_pdf_at_mean_test() {
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  // f(0; 0, 1) = 1/√(2π)
  should.be_true(approx(
    distributions.gaussian_pdf(g, 0.0),
    constants.inv_sqrt_2pi,
    ultra,
  ))
}

pub fn precision_gaussian_pdf_at_one_sigma_test() {
  // f(1; 0, 1) = e^(-0.5)/√(2π) ≈ 0.24197072451914337
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  should.be_true(approx(
    distributions.gaussian_pdf(g, 1.0),
    0.24197072451914337,
    ultra,
  ))
}

pub fn precision_gaussian_cdf_1_96_test() {
  // F(1.96; 0, 1) ≈ 0.9750021048517796 (the famous Z value)
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  should.be_true(approx(
    distributions.gaussian_cdf(g, 1.96),
    0.9750021048517796,
    ultra,
  ))
}

pub fn precision_gaussian_cdf_symmetry_test() {
  // F(-x) = 1 - F(x)
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  let xs = [0.5, 1.0, 1.5, 2.0]
  list.each(xs, fn(x) {
    should.be_true(approx(
      distributions.gaussian_cdf(g, 0.0 -. x),
      1.0 -. distributions.gaussian_cdf(g, x),
      ultra,
    ))
  })
}

pub fn precision_gaussian_log_pdf_consistency_test() {
  // log_pdf(x) should equal ln(pdf(x))
  let g = distributions.Gaussian(mean: 0.5, stddev: 1.5)
  let xs = [-1.0, 0.0, 0.5, 1.0]
  list.each(xs, fn(x) {
    let pdf = distributions.gaussian_pdf(g, x)
    let log_pdf = distributions.gaussian_log_pdf(g, x)
    should.be_true(approx(log_pdf, scalar.ln(pdf), tight))
  })
}

pub fn precision_uniform_cdf_test() {
  let u = distributions.Uniform(low: 0.0, high: 10.0)
  should.be_true(approx(distributions.uniform_cdf(u, 5.0), 0.5, ultra))
  should.be_true(approx(distributions.uniform_cdf(u, -1.0), 0.0, ultra))
  should.be_true(approx(distributions.uniform_cdf(u, 100.0), 1.0, ultra))
}

pub fn precision_exponential_cdf_test() {
  let e = distributions.Exponential(rate: 1.0)
  // F(1; λ=1) = 1 - e^-1 ≈ 0.6321205588285577
  should.be_true(approx(
    distributions.exponential_cdf(e, 1.0),
    1.0 -. scalar.exp(-1.0),
    ultra,
  ))
}

// ============================================================================
// scheduler - golden values
// ============================================================================

pub fn precision_cosine_anneal_half_test() {
  // At step T_max/2, cosine_annealing = (base + min) / 2
  // base=1, min=0, t_max=100, step=50 → 0.5
  should.be_true(approx(
    scheduler.cosine_annealing(1.0, 50, 100, 0.0),
    0.5,
    ultra,
  ))
}

pub fn precision_cosine_anneal_with_min_test() {
  // base=1, min=0.1, t_max=100, step=100 → 0.1
  should.be_true(approx(
    scheduler.cosine_annealing(1.0, 100, 100, 0.1),
    0.1,
    ultra,
  ))
}

pub fn precision_linear_warmup_quarter_test() {
  should.be_true(approx(scheduler.linear_warmup(4.0, 25, 100), 1.0, ultra))
}

pub fn precision_exponential_decay_test() {
  // exp(1, 10, 0.5) = 1 · 0.5^10 = 0.0009765625
  should.be_true(approx(
    scheduler.exponential(1.0, 10, 0.5),
    0.0009765625,
    ultra,
  ))
}

pub fn precision_triangle_period_test() {
  // Triangle should be 0 at step=0 and step=period
  should.be_true(approx(scheduler.triangle(0, 10), 0.0, ultra))
  should.be_true(approx(scheduler.triangle(10, 10), 0.0, ultra))
  // 1.0 at half period
  should.be_true(approx(scheduler.triangle(5, 10), 1.0, ultra))
}

// ============================================================================
// ODE - high-order accuracy verification
// ============================================================================

pub fn precision_rk4_exp_test() {
  // dx/dt = x, x(0)=1 → x(t) = e^t. Test x(1.0) using 10 steps of dt=0.1.
  // RK4 truncation error per step is O(dt⁵); cumulative over 10 steps with
  // dt=0.1 lands around 2·10⁻⁶ in practice on IEEE-754 doubles.
  let f = fn(_t: Float, x: Float) { x }
  let traj = ode.integrate(ode.rk4, f, 0.0, 1.0, 0.1, 10)
  let assert Ok(final_pair) = list.last(traj)
  let #(_, x_final) = final_pair
  should.be_true(approx(x_final, constants.e, 1.0e-5))
}

pub fn precision_rk4_better_than_euler_test() {
  // Same ODE: RK4 should be much more accurate than Euler.
  let f = fn(_t: Float, x: Float) { x }
  let traj_rk4 = ode.integrate(ode.rk4, f, 0.0, 1.0, 0.1, 10)
  let traj_euler = ode.integrate(ode.euler, f, 0.0, 1.0, 0.1, 10)
  let assert Ok(rk4_final) = list.last(traj_rk4)
  let assert Ok(euler_final) = list.last(traj_euler)
  let err_rk4 = float.absolute_value(rk4_final.1 -. constants.e)
  let err_euler = float.absolute_value(euler_final.1 -. constants.e)
  should.be_true(err_rk4 <. err_euler)
}

pub fn precision_rk4_decay_test() {
  // dx/dt = -x, x(0) = 1 → x(t) = e^-t. Test x(1).
  let g = fn(_t: Float, x: Float) { 0.0 -. x }
  let traj = ode.integrate(ode.rk4, g, 0.0, 1.0, 0.01, 100)
  let assert Ok(final_pair) = list.last(traj)
  should.be_true(approx(final_pair.1, scalar.exp(-1.0), loose))
}

// ============================================================================
// calculus - high-order quadrature precision
// ============================================================================

pub fn precision_simpson_polynomial_test() {
  // ∫₀² x³ dx = 4
  let f = fn(x: Float) { x *. x *. x }
  let assert Ok(r) = calculus.simpson(f, 0.0, 2.0, 100)
  // Simpson is exact for cubics in absence of round-off.
  should.be_true(approx(r, 4.0, tight))
}

pub fn precision_simpson_sin_test() {
  // ∫₀^π sin(x) dx = 2
  let assert Ok(r) = calculus.simpson(scheduler_sin, 0.0, constants.pi, 100)
  should.be_true(approx(r, 2.0, loose))
}

pub fn precision_romberg_test() {
  // Romberg should achieve very high precision on smooth integrands.
  // ∫₀¹ e^x dx = e - 1
  let f = fn(x: Float) { scalar.exp(x) }
  let r = calculus.romberg(f, 0.0, 1.0, 8)
  should.be_true(approx(r, constants.e -. 1.0, 1.0e-10))
}

pub fn precision_central_diff_test() {
  // d/dx sin(x) at x=π/4 = cos(π/4) = √2/2
  let f = fn(x: Float) { scheduler_sin(x) }
  let r = calculus.central_diff(f, constants.quarter_pi, 1.0e-5)
  should.be_true(approx(r, constants.sqrt_2 /. 2.0, 1.0e-7))
}

pub fn precision_five_point_diff_test() {
  // Higher order: d/dx (x^4) at x=1 = 4
  let f = fn(x: Float) { x *. x *. x *. x }
  let r = calculus.five_point_diff(f, 1.0, 1.0e-3)
  should.be_true(approx(r, 4.0, 1.0e-9))
}

@external(erlang, "math", "sin")
@external(javascript, "./viva_math_random_ffi.mjs", "sin")
fn scheduler_sin(x: Float) -> Float

// ============================================================================
// matrix - identity and inverse precision
// ============================================================================

pub fn precision_mat3_inverse_test() {
  // Specific invertible 3x3 matrix; check M · M⁻¹ ≈ I
  let m =
    matrix.Mat3(
      m11: 6.0,
      m12: 1.0,
      m13: 2.0,
      m21: 0.0,
      m22: 5.0,
      m23: 4.0,
      m31: 8.0,
      m32: 7.0,
      m33: 1.0,
    )
  let assert Ok(inv) = matrix.mat3_inverse(m)
  let prod = matrix.mat3_mul(m, inv)
  let id = matrix.mat3_identity()
  should.be_true(approx(prod.m11, id.m11, tight))
  should.be_true(approx(prod.m22, id.m22, tight))
  should.be_true(approx(prod.m33, id.m33, tight))
  should.be_true(approx(prod.m12, 0.0, tight))
  should.be_true(approx(prod.m23, 0.0, tight))
}

pub fn precision_mat3_determinant_test() {
  // Determinant computed by hand for this matrix = -186
  let m =
    matrix.Mat3(
      m11: 6.0,
      m12: 1.0,
      m13: 2.0,
      m21: 0.0,
      m22: 5.0,
      m23: 4.0,
      m31: 8.0,
      m32: 7.0,
      m33: 1.0,
    )
  should.be_true(approx(matrix.mat3_determinant(m), -186.0, ultra))
}

pub fn precision_mat3_rotation_orthogonal_test() {
  // R(θ) is orthogonal: R · R^T = I
  let r = matrix.mat3_rot_z(constants.half_pi)
  let prod = matrix.mat3_mul(r, matrix.mat3_transpose(r))
  should.be_true(approx(prod.m11, 1.0, tight))
  should.be_true(approx(prod.m22, 1.0, tight))
  should.be_true(approx(prod.m33, 1.0, tight))
  should.be_true(approx(prod.m12, 0.0, tight))
}

pub fn precision_matn_validates_empty_rows_test() {
  // matn_from_rows([[], []]) must error.
  let result = matrix.matn_from_rows([[], []])
  case result {
    Error(_) -> should.be_true(True)
    Ok(_) -> should.fail()
  }
}

// ============================================================================
// entropy - Tsallis continuity at q ≈ 1
// ============================================================================

pub fn precision_tsallis_q_one_continuous_test() {
  // S_q → Shannon as q → 1; fuzzy check at q very close to 1.
  let probs = [0.1, 0.3, 0.6]
  let assert Ok(t) = entropy.tsallis(probs, 1.0 +. 1.0e-12)
  should.be_true(approx(t, entropy.shannon(probs), tight))
}

pub fn precision_tsallis_uniform_test() {
  // S_q for uniform [1/n, ..., 1/n] = (1 - n^(1-q)) / (q-1)
  // For q=2, n=4: (1 - 4^-1) / 1 = 0.75
  let probs = [0.25, 0.25, 0.25, 0.25]
  let assert Ok(t) = entropy.tsallis(probs, 2.0)
  should.be_true(approx(t, 0.75, tight))
}

pub fn precision_fisher_gaussian_test() {
  // I(σ) = 1/σ²
  let assert Ok(i) = entropy.fisher_information_gaussian(0.5)
  should.be_true(approx(i, 4.0, ultra))
}

// ============================================================================
// random - reproducibility / statistical sanity
// ============================================================================

pub fn precision_random_normal_stddev_test() {
  // Sample n times from N(0, 1) and verify empirical sigma ≈ 1.
  // Tolerance loose because of finite-sample variance.
  let seed = random.from_int(2026)
  let #(samples, _) = random.standard_normals(seed, 5000)
  let assert Ok(sigma) = statistics.stddev(samples)
  should.be_true(float.absolute_value(sigma -. 1.0) <. 0.05)
}

pub fn precision_random_normal_mean_test() {
  // Empirical mean of large N(0,1) sample ≈ 0.
  let seed = random.from_int(2026)
  let #(samples, _) = random.standard_normals(seed, 5000)
  let assert Ok(m) = statistics.mean(samples)
  should.be_true(float.absolute_value(m) <. 0.05)
}

pub fn precision_random_uniform_mean_test() {
  // Uniform [0,1) mean ≈ 0.5
  let seed = random.from_int(2026)
  let #(samples, _) = random.uniforms(seed, 5000)
  let assert Ok(m) = statistics.mean(samples)
  should.be_true(float.absolute_value(m -. 0.5) <. 0.02)
}

pub fn precision_random_normal_with_sigma_test() {
  // After codex fix: random.normal(seed, mu, sigma) honours sigma (not variance).
  // Empirical stddev should approach the requested sigma, not √sigma.
  let seed = random.from_int(99)
  let #(samples, _) = draw_normals_with(seed, 0.0, 2.0, 5000, [])
  let assert Ok(sigma_est) = statistics.stddev(samples)
  should.be_true(float.absolute_value(sigma_est -. 2.0) <. 0.1)
}

fn draw_normals_with(
  seed: random.Seed,
  mu: Float,
  sigma: Float,
  n: Int,
  acc: List(Float),
) -> #(List(Float), random.Seed) {
  case n <= 0 {
    True -> #(acc, seed)
    False -> {
      let #(x, s) = random.normal(seed, mu, sigma)
      draw_normals_with(s, mu, sigma, n - 1, [x, ..acc])
    }
  }
}

pub fn precision_random_categorical_rejects_negative_test() {
  // After codex fix: categorical must reject negative probs.
  let seed = random.from_int(1)
  case random.categorical(seed, [-0.1, 0.5, 0.6]) {
    Error(_) -> should.be_true(True)
    Ok(_) -> should.fail()
  }
}

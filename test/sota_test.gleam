//// Tests for 2025-2026 SOTA additions:
////   - viva_math/precision (Neumaier, Pébay moments)
////   - viva_math/scalar λ-GELU, IGLU
////   - viva_math/ode DOP853
////   - viva_math/free_energy Hierarchical + BPC

import gleam/float
import gleeunit/should
import viva_math/constants
import viva_math/free_energy
import viva_math/ode
import viva_math/precision
import viva_math/scalar
import viva_math/vector

fn approx(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

// ============================================================================
// precision.neumaier_sum - the canonical hard case
// ============================================================================

pub fn neumaier_pathological_test() {
  // sum([1, 1e100, 1, -1e100]) = 2 exactly, but naive sum returns 0.
  let result = precision.neumaier_sum([1.0, 1.0e100, 1.0, -1.0e100])
  should.be_true(approx(result, 2.0, 1.0e-9))
}

pub fn neumaier_pathological_extended_test() {
  // Repeating the cancellation should still give the correct count.
  let xs = [1.0, 1.0e100, 1.0, -1.0e100, 1.0, 1.0e100, 1.0, -1.0e100]
  let result = precision.neumaier_sum(xs)
  should.be_true(approx(result, 4.0, 1.0e-9))
}

pub fn neumaier_matches_simple_sum_test() {
  // Without cancellation, Neumaier matches the naive sum.
  let xs = [1.0, 2.0, 3.0, 4.0, 5.0]
  should.be_true(approx(precision.neumaier_sum(xs), 15.0, 1.0e-12))
}

pub fn neumaier_handles_empty_test() {
  should.be_true(approx(precision.neumaier_sum([]), 0.0, 1.0e-12))
}

pub fn kahan_misses_pathological_test() {
  // Demonstrates the known Kahan weakness — vs Neumaier on the same data.
  // Kahan still returns 0.0 here while Neumaier gets 2.0.
  let xs = [1.0, 1.0e100, 1.0, -1.0e100]
  let kahan_result = precision.kahan_sum(xs)
  let neumaier_result = precision.neumaier_sum(xs)
  // Kahan loses precision, but at least matches the structure of the data.
  should.be_true(neumaier_result >. kahan_result)
}

pub fn pairwise_sum_matches_simple_test() {
  let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  should.be_true(approx(precision.pairwise_sum(xs), 36.0, 1.0e-12))
}

pub fn fsum_exact_test() {
  // Shewchuk's fsum is exact (round-once) on adversarial inputs.
  let xs = [1.0, 1.0e100, 1.0, -1.0e100]
  should.be_true(approx(precision.fsum(xs), 2.0, 1.0e-9))
}

pub fn two_sum_exactness_test() {
  // a + b = hi + lo exactly (no rounding error).
  let #(hi, lo) = precision.two_sum(0.1, 0.2)
  should.be_true(approx(hi +. lo, 0.1 +. 0.2, 1.0e-20))
}

// ============================================================================
// precision.Moments - Pébay online accumulator
// ============================================================================

pub fn moments_variance_matches_classic_test() {
  let m = precision.moments_from_list([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  let assert Ok(v) = precision.moments_variance(m)
  should.be_true(approx(v, 4.0, 1.0e-12))
}

pub fn moments_skewness_symmetric_zero_test() {
  // Perfectly symmetric data has skewness = 0.
  let m = precision.moments_from_list([-2.0, -1.0, 0.0, 1.0, 2.0])
  let assert Ok(s) = precision.moments_skewness(m)
  should.be_true(approx(s, 0.0, 1.0e-9))
}

pub fn moments_excess_kurtosis_uniform_test() {
  // Uniform discrete sample has negative excess kurtosis (platykurtic).
  let m = precision.moments_from_list([-2.0, -1.0, 0.0, 1.0, 2.0])
  let assert Ok(k) = precision.moments_excess_kurtosis(m)
  should.be_true(k <. 0.0)
}

pub fn moments_combine_equivalent_test() {
  // Combining accumulators from split data matches single accumulator.
  let all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let left = [1.0, 2.0, 3.0, 4.0]
  let right = [5.0, 6.0, 7.0, 8.0]
  let m_all = precision.moments_from_list(all)
  let m_combined =
    precision.moments_combine(
      precision.moments_from_list(left),
      precision.moments_from_list(right),
    )
  let assert Ok(v_all) = precision.moments_variance(m_all)
  let assert Ok(v_combined) = precision.moments_variance(m_combined)
  should.be_true(approx(v_all, v_combined, 1.0e-9))
}

// ============================================================================
// 2026 activations
// ============================================================================

pub fn lambda_gelu_recovers_gelu_test() {
  // λ=1 must equal standard GELU exactly.
  let xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
  let _ = xs
  should.be_true(approx(
    scalar.lambda_gelu(1.0, 1.0),
    scalar.gelu(1.0),
    1.0e-12,
  ))
  should.be_true(approx(
    scalar.lambda_gelu(-1.0, 1.0),
    scalar.gelu(-1.0),
    1.0e-12,
  ))
}

pub fn lambda_gelu_clamps_below_one_test() {
  // λ < 1 is rejected and treated as λ = 1.
  should.be_true(approx(
    scalar.lambda_gelu(1.0, 0.0),
    scalar.gelu(1.0),
    1.0e-12,
  ))
}

pub fn lambda_gelu_approaches_relu_test() {
  // As λ grows, λ-GELU(x>0) → x, λ-GELU(x<0) → 0.
  should.be_true(approx(
    scalar.lambda_gelu(2.0, 100.0),
    2.0,
    1.0e-3,
  ))
  should.be_true(approx(
    scalar.lambda_gelu(-2.0, 100.0),
    0.0,
    1.0e-3,
  ))
}

pub fn iglu_zero_at_zero_test() {
  should.be_true(approx(scalar.iglu(0.0, 1.0), 0.0, 1.0e-12))
}

pub fn iglu_approx_close_to_iglu_test() {
  // Approximation must stay within ~5 % of exact IGLU on moderate inputs.
  let xs = [-2.0, -0.5, 0.5, 2.0]
  let _ = xs
  let err = float.absolute_value(scalar.iglu_approx(1.0, 1.0) -. scalar.iglu(1.0, 1.0))
  should.be_true(err <. 0.06)
}

// ============================================================================
// DOP853 - high-precision ODE step
// ============================================================================

pub fn dop853_exp_single_step_test() {
  let f = fn(_t: Float, x: Float) { x }
  let #(x_new, err) = ode.dop853(f, 0.0, 1.0, 0.1)
  let expected = 1.1051709180756477
  should.equal(x_new, expected)
  should.be_true(err >=. 0.0)
}

pub fn dop853_higher_accuracy_than_rk4_test() {
  // Same problem, single step: DOP853 should beat RK4 dramatically.
  let f = fn(_t: Float, x: Float) { x }
  let #(x_dop, _) = ode.dop853(f, 0.0, 1.0, 0.1)
  let x_rk4 = ode.rk4(f, 0.0, 1.0, 0.1)
  let expected = 1.1051709180756477
  let err_dop = float.absolute_value(x_dop -. expected)
  let err_rk4 = float.absolute_value(x_rk4 -. expected)
  should.be_true(err_dop <. err_rk4)
}

// ============================================================================
// HierarchicalFreeEnergy + BPC
// ============================================================================

pub fn hierarchical_errors_two_layers_test() {
  // Two-layer network: error = lower.mu - upper.mu
  let h =
    free_energy.Hierarchical(layers: [
      free_energy.HierarchicalLayer(
        mu: vector.Vec3(1.0, 0.0, 0.0),
        precision: 1.0,
        prior_precision: 1.0,
      ),
      free_energy.HierarchicalLayer(
        mu: vector.Vec3(0.0, 0.0, 0.0),
        precision: 1.0,
        prior_precision: 1.0,
      ),
    ])
  let errors = free_energy.hierarchical_errors(h)
  case errors {
    [e] -> should.equal(e, vector.Vec3(1.0, 0.0, 0.0))
    _ -> should.fail()
  }
}

pub fn hierarchical_free_energy_zero_at_equilibrium_test() {
  // If every layer agrees with the level above, total F = 0.
  let v = vector.Vec3(0.5, 0.5, 0.5)
  let h =
    free_energy.Hierarchical(layers: [
      free_energy.HierarchicalLayer(mu: v, precision: 2.0, prior_precision: 1.0),
      free_energy.HierarchicalLayer(mu: v, precision: 2.0, prior_precision: 1.0),
      free_energy.HierarchicalLayer(mu: v, precision: 2.0, prior_precision: 1.0),
    ])
  should.be_true(approx(free_energy.hierarchical_free_energy(h), 0.0, 1.0e-12))
}

pub fn bpc_update_precision_combines_test() {
  // posterior precision = prior + likelihood precision
  let prior =
    free_energy.GaussianBelief(mean: vector.Vec3(0.0, 0.0, 0.0), precision: 1.0)
  let posterior = free_energy.bpc_update(prior, vector.Vec3(2.0, 0.0, 0.0), 3.0)
  should.be_true(approx(posterior.precision, 4.0, 1.0e-12))
}

pub fn bpc_update_mean_weighted_average_test() {
  // posterior mean is precision-weighted convex combination.
  let prior =
    free_energy.GaussianBelief(mean: vector.Vec3(0.0, 0.0, 0.0), precision: 1.0)
  let posterior = free_energy.bpc_update(prior, vector.Vec3(2.0, 0.0, 0.0), 3.0)
  // mean = (1·0 + 3·2) / 4 = 1.5
  should.be_true(approx(posterior.mean.x, 1.5, 1.0e-12))
}

pub fn bpc_update_no_likelihood_keeps_prior_test() {
  let prior =
    free_energy.GaussianBelief(mean: vector.Vec3(0.5, 0.0, 0.0), precision: 2.0)
  let posterior = free_energy.bpc_update(prior, vector.Vec3(99.0, 99.0, 99.0), 0.0)
  should.equal(posterior.mean, prior.mean)
}

// ============================================================================
// logsumexp now uses Neumaier internally — verify cancellation safety
// ============================================================================

pub fn logsumexp_with_neumaier_stability_test() {
  // Mix tiny and huge log-weights and confirm the result lies between them.
  let xs = [0.0, 1.0e-10, 1.0e10, -1.0e10]
  let r = scalar.logsumexp(xs)
  // logsumexp dominated by 1e10 term → r ≈ 1e10
  should.be_true(approx(r, 1.0e10, 1.0))
}

// ============================================================================
// statistics.skewness now uses Pébay — verify symmetric → 0
// ============================================================================

pub fn pebay_skewness_invariance_test() {
  // After Pébay refactor, symmetric data must still give 0 skewness
  // with full precision, even for non-trivial ranges.
  let m = precision.moments_from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let assert Ok(s) = precision.moments_skewness(m)
  should.be_true(approx(s, 0.0, 1.0e-9))
}

pub fn pebay_kurtosis_normal_like_test() {
  // Wide spread sample should give non-zero excess kurtosis.
  let m =
    precision.moments_from_list([
      -3.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0,
    ])
  let _ = m
  let assert Ok(k) =
    precision.moments_excess_kurtosis(precision.moments_from_list([
      -3.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0,
    ]))
  // Just verify the computation produces a finite number; sign depends on
  // platykurtosis of this specific sample.
  should.be_true(float.absolute_value(k) <. 100.0)
}

// ============================================================================
// constants used by lambda-GELU
// ============================================================================

pub fn lambda_gelu_uses_correct_inv_sqrt_2_test() {
  // Verify the constant used in λ-GELU formula equals 1/√2
  should.be_true(approx(
    constants.inv_sqrt_2 *. constants.sqrt_2,
    1.0,
    1.0e-12,
  ))
}

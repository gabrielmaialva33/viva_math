import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_math/attractor
import viva_math/calculus
import viva_math/common
import viva_math/constants
import viva_math/cusp
import viva_math/distributions
import viva_math/entropy
import viva_math/free_energy
import viva_math/matrix
import viva_math/ode
import viva_math/ou
import viva_math/random
import viva_math/scalar
import viva_math/scheduler
import viva_math/statistics
import viva_math/transport
import viva_math/vec2
import viva_math/vec4
import viva_math/vecn
import viva_math/vector.{Vec3}

pub fn main() {
  gleeunit.main()
}

// ============================================================================
// common.gleam tests
// ============================================================================

pub fn clamp_test() {
  common.clamp(5.0, 0.0, 10.0)
  |> should.equal(5.0)

  common.clamp(-1.0, 0.0, 10.0)
  |> should.equal(0.0)

  common.clamp(15.0, 0.0, 10.0)
  |> should.equal(10.0)
}

pub fn clamp_unit_test() {
  common.clamp_unit(0.5)
  |> should.equal(0.5)

  common.clamp_unit(-0.5)
  |> should.equal(0.0)

  common.clamp_unit(1.5)
  |> should.equal(1.0)
}

pub fn clamp_bipolar_test() {
  common.clamp_bipolar(0.5)
  |> should.equal(0.5)

  common.clamp_bipolar(-1.5)
  |> should.equal(-1.0)

  common.clamp_bipolar(1.5)
  |> should.equal(1.0)
}

pub fn lerp_test() {
  common.lerp(0.0, 10.0, 0.5)
  |> should.equal(5.0)

  common.lerp(0.0, 10.0, 0.0)
  |> should.equal(0.0)

  common.lerp(0.0, 10.0, 1.0)
  |> should.equal(10.0)
}

pub fn sigmoid_center_test() {
  // sigmoid(0) should be 0.5
  let result = common.sigmoid(0.0, 1.0)
  should.be_true(is_close(result, 0.5, 0.001))
}

pub fn sigmoid_extremes_test() {
  // sigmoid(-100) should be ~0
  let low = common.sigmoid(-100.0, 1.0)
  should.be_true(low <. 0.001)

  // sigmoid(100) should be ~1
  let high = common.sigmoid(100.0, 1.0)
  should.be_true(high >. 0.999)
}

pub fn softmax_test() {
  // Equal inputs should give equal outputs
  let assert Ok(result) = common.softmax([1.0, 1.0])
  case result {
    [a, b] -> {
      should.be_true(is_close(a, 0.5, 0.001))
      should.be_true(is_close(b, 0.5, 0.001))
    }
    _ -> should.fail()
  }
}

pub fn softmax_sum_to_one_test() {
  let assert Ok(result) = common.softmax([1.0, 2.0, 3.0])
  let sum = list.fold(result, 0.0, fn(acc, x) { acc +. x })
  should.be_true(is_close(sum, 1.0, 0.001))
}

pub fn safe_div_test() {
  common.safe_div(10.0, 2.0, 0.0)
  |> should.equal(5.0)

  common.safe_div(10.0, 0.0, -1.0)
  |> should.equal(-1.0)
}

// ============================================================================
// vector.gleam tests
// ============================================================================

pub fn vec3_zero_test() {
  vector.zero()
  |> should.equal(Vec3(0.0, 0.0, 0.0))
}

pub fn vec3_add_test() {
  let a = Vec3(1.0, 2.0, 3.0)
  let b = Vec3(4.0, 5.0, 6.0)
  vector.add(a, b)
  |> should.equal(Vec3(5.0, 7.0, 9.0))
}

pub fn vec3_sub_test() {
  let a = Vec3(5.0, 7.0, 9.0)
  let b = Vec3(1.0, 2.0, 3.0)
  vector.sub(a, b)
  |> should.equal(Vec3(4.0, 5.0, 6.0))
}

pub fn vec3_scale_test() {
  let v = Vec3(1.0, 2.0, 3.0)
  vector.scale(v, 2.0)
  |> should.equal(Vec3(2.0, 4.0, 6.0))
}

pub fn vec3_dot_test() {
  let a = Vec3(1.0, 2.0, 3.0)
  let b = Vec3(4.0, 5.0, 6.0)
  vector.dot(a, b)
  |> should.equal(32.0)
  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

pub fn vec3_length_test() {
  let v = Vec3(3.0, 4.0, 0.0)
  let len = vector.length(v)
  should.be_true(is_close(len, 5.0, 0.001))
}

pub fn vec3_distance_test() {
  let a = Vec3(0.0, 0.0, 0.0)
  let b = Vec3(1.0, 1.0, 1.0)
  let dist = vector.distance(a, b)
  // sqrt(3) ≈ 1.732
  should.be_true(is_close(dist, 1.732, 0.01))
}

pub fn vec3_normalize_test() {
  let v = Vec3(3.0, 0.0, 0.0)
  vector.normalize(v)
  |> should.equal(Vec3(1.0, 0.0, 0.0))
}

pub fn vec3_clamp_pad_test() {
  let v = Vec3(2.0, -2.0, 0.5)
  vector.clamp_pad(v)
  |> should.equal(Vec3(1.0, -1.0, 0.5))
}

pub fn vec3_pad_test() {
  let v = vector.pad(0.5, -0.3, 0.8)
  should.equal(v.x, 0.5)
  should.equal(v.y, -0.3)
  should.equal(v.z, 0.8)
}

// ============================================================================
// cusp.gleam tests
// ============================================================================

pub fn cusp_potential_at_zero_test() {
  // V(0) = 0^4/4 + α*0^2/2 + β*0 = 0
  let params = cusp.CuspParams(-1.0, 0.0)
  cusp.potential(0.0, params)
  |> should.equal(0.0)
}

pub fn cusp_gradient_at_zero_test() {
  // dV/dx(0) = 0^3 + α*0 + β = β
  let params = cusp.CuspParams(-1.0, 0.5)
  cusp.gradient(0.0, params)
  |> should.equal(0.5)
}

pub fn cusp_is_bistable_test() {
  // α = -1 < 0, Δ = -4*(-1)^3 - 27*0^2 = 4 > 0 → bistable
  let params = cusp.CuspParams(-1.0, 0.0)
  cusp.is_bistable(params)
  |> should.be_true()
}

pub fn cusp_is_monostable_test() {
  // α = 1 ≥ 0 → monostable regardless of β
  let params = cusp.CuspParams(1.0, 0.0)
  cusp.is_bistable(params)
  |> should.be_false()
}

pub fn cusp_from_arousal_test() {
  // High arousal (0.8) should give negative alpha → potential bistability
  let params = cusp.from_arousal_dominance(0.8, 0.0)
  should.be_true(params.alpha <. 0.0)
}

pub fn cusp_equilibria_monostable_test() {
  // With α > 0, should be monostable
  let params = cusp.CuspParams(1.0, 0.0)
  case cusp.equilibria(params) {
    cusp.Monostable(_) -> should.be_true(True)
    cusp.Bistable(_, _, _) -> should.fail()
  }
}

pub fn cusp_equilibria_bistable_test() {
  // With α = -1, β = 0, should be bistable
  let params = cusp.CuspParams(-1.0, 0.0)
  case cusp.equilibria(params) {
    cusp.Bistable(_, _, _) -> should.be_true(True)
    cusp.Monostable(_) -> should.fail()
  }
}

// ============================================================================
// free_energy.gleam tests
// ============================================================================

pub fn prediction_error_zero_test() {
  // Same state → zero error
  let state = Vec3(0.5, 0.3, -0.2)
  free_energy.prediction_error(state, state)
  |> should.equal(0.0)
}

pub fn prediction_error_nonzero_test() {
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 0.0, 0.0)
  // Squared distance = 1.0
  free_energy.prediction_error(expected, actual)
  |> should.equal(1.0)
}

pub fn free_energy_homeostatic_test() {
  // Low free energy should be homeostatic
  let state = Vec3(0.0, 0.0, 0.0)
  let result = free_energy.compute_state_simple(state, state, state, 0.1)
  should.equal(result.feeling, free_energy.Homeostatic)
}

pub fn free_energy_alarmed_test() {
  // High prediction error should NOT be homeostatic
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 1.0, 0.0)
  let baseline = Vec3(0.0, 0.0, 0.0)
  let result = free_energy.compute_state_simple(expected, actual, baseline, 0.1)
  // Distance squared = 2.0 + complexity ~= 2.2, should be Alarmed or Overwhelmed
  should.be_true(
    result.feeling == free_energy.Alarmed
    || result.feeling == free_energy.Overwhelmed,
  )
}

pub fn free_energy_precision_weighted_test() {
  // Higher precision should amplify prediction error
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 0.0, 0.0)

  let low_precision =
    free_energy.precision_weighted_prediction_error(expected, actual, 0.5)
  let high_precision =
    free_energy.precision_weighted_prediction_error(expected, actual, 2.0)

  should.be_true(high_precision >. low_precision)
  should.be_true(is_close(low_precision, 0.5, 0.001))
  should.be_true(is_close(high_precision, 2.0, 0.001))
}

pub fn free_energy_gaussian_kl_test() {
  // KL divergence between same distributions is 0
  let state = Vec3(0.5, 0.3, -0.2)
  let kl = free_energy.gaussian_kl_divergence(state, state, 1.0)
  should.be_true(is_close(kl, 0.0, 0.001))
}

pub fn free_energy_normalized_thresholds_test() {
  // Test normalized threshold classification
  let thresholds = free_energy.FeelingThresholds(mean: 1.0, std_dev: 0.5)

  // F < μ - σ = 0.5 → Homeostatic
  should.equal(
    free_energy.classify_feeling_normalized(0.3, thresholds),
    free_energy.Homeostatic,
  )

  // μ - σ ≤ F < μ → Surprised
  should.equal(
    free_energy.classify_feeling_normalized(0.7, thresholds),
    free_energy.Surprised,
  )

  // μ ≤ F < μ + σ → Alarmed
  should.equal(
    free_energy.classify_feeling_normalized(1.2, thresholds),
    free_energy.Alarmed,
  )

  // F ≥ μ + σ → Overwhelmed
  should.equal(
    free_energy.classify_feeling_normalized(2.0, thresholds),
    free_energy.Overwhelmed,
  )
}

// ============================================================================
// attractor.gleam tests
// ============================================================================

pub fn attractor_classify_joy_test() {
  // Point near joy attractor
  let state = Vec3(0.7, 0.5, 0.3)
  attractor.classify_emotion(state)
  |> should.equal("joy")
}

pub fn attractor_classify_sadness_test() {
  // Point near sadness attractor
  let state = Vec3(-0.6, -0.3, -0.3)
  attractor.classify_emotion(state)
  |> should.equal("sadness")
}

pub fn attractor_classify_fear_test() {
  // Point near fear attractor
  let state = Vec3(-0.6, 0.6, -0.4)
  attractor.classify_emotion(state)
  |> should.equal("fear")
}

pub fn attractor_nearest_test() {
  let attractors = attractor.emotional_attractors()
  let point = Vec3(0.76, 0.48, 0.35)
  // Exactly at joy
  let assert Ok(nearest) = attractor.nearest(point, attractors)
  should.equal(nearest.name, "joy")
}

pub fn attractor_basin_weights_sum_test() {
  let attractors = attractor.emotional_attractors()
  let point = Vec3(0.0, 0.0, 0.0)
  let weights = attractor.basin_weights(point, attractors, 1.0)
  let sum =
    list.fold(weights, 0.0, fn(acc, pair) {
      let #(_, w) = pair
      acc +. w
    })
  // Weights should sum to ~1.0
  should.be_true(is_close(sum, 1.0, 0.01))
}

// ============================================================================
// entropy.gleam tests
// ============================================================================

pub fn entropy_uniform_test() {
  // Uniform distribution [0.5, 0.5] has entropy = 1 bit
  let h = entropy.shannon([0.5, 0.5])
  should.be_true(is_close(h, 1.0, 0.001))
}

pub fn entropy_certain_test() {
  // Certain outcome [1.0, 0.0] has entropy = 0
  let h = entropy.shannon([1.0, 0.0])
  should.be_true(is_close(h, 0.0, 0.001))
}

pub fn entropy_four_uniform_test() {
  // Uniform [0.25, 0.25, 0.25, 0.25] has entropy = 2 bits
  let h = entropy.shannon([0.25, 0.25, 0.25, 0.25])
  should.be_true(is_close(h, 2.0, 0.001))
}

pub fn kl_divergence_same_test() {
  // KL divergence of identical distributions is 0
  let p = [0.5, 0.5]
  let assert Ok(kl) = entropy.kl_divergence(p, p)
  should.be_true(is_close(kl, 0.0, 0.001))
}

pub fn kl_divergence_different_test() {
  // KL divergence of different distributions is positive
  let p = [0.9, 0.1]
  let q = [0.5, 0.5]
  let assert Ok(kl) = entropy.kl_divergence(p, q)
  should.be_true(kl >. 0.0)
}

pub fn jensen_shannon_symmetric_test() {
  // JS divergence should be symmetric
  let p = [0.9, 0.1]
  let q = [0.5, 0.5]
  let assert Ok(js_pq) = entropy.jensen_shannon(p, q)
  let assert Ok(js_qp) = entropy.jensen_shannon(q, p)
  should.be_true(is_close(js_pq, js_qp, 0.001))
}

// ============================================================================
// NEW: Stochastic cusp tests (DeepSeek R1 proposals)
// ============================================================================

pub fn stochastic_cusp_deterministic_test() {
  // Same seed should produce same noise
  let noise1 = common.deterministic_noise(0, 42)
  let noise2 = common.deterministic_noise(0, 42)
  should.be_true(is_close(noise1, noise2, 0.0001))
}

pub fn stochastic_cusp_different_steps_test() {
  // Different steps should produce different noise
  let noise1 = common.deterministic_noise(0, 42)
  let noise2 = common.deterministic_noise(1, 42)
  should.be_false(is_close(noise1, noise2, 0.0001))
}

pub fn stochastic_cusp_range_test() {
  // Noise should be in [-1, 1]
  let noise = common.deterministic_noise(100, 999)
  should.be_true(noise >=. -1.0 && noise <=. 1.0)
}

pub fn stochastic_simulation_length_test() {
  // Simulation should return correct number of steps
  let params =
    cusp.StochasticCuspParams(alpha: -1.0, beta: 0.0, sigma: 0.1, seed: 42)
  let trajectory = cusp.simulate_stochastic(0.0, params, 0.01, 10)
  should.equal(list.length(trajectory), 11)
  // initial + 10 steps
}

// ============================================================================
// NEW: Basin weights with exp(-γd) tests
// ============================================================================

pub fn basin_weights_exp_sum_to_one_test() {
  // Basin weights should sum to 1.0
  let attractors = attractor.emotional_attractors()
  let point = vector.Vec3(0.0, 0.0, 0.0)
  let weights = attractor.basin_weights(point, attractors, 1.0)
  let sum = list.fold(weights, 0.0, fn(acc, pair) { acc +. pair.1 })
  should.be_true(is_close(sum, 1.0, 0.01))
}

pub fn basin_weights_temperature_effect_test() {
  // Lower temperature should make weights sharper (max weight higher)
  let attractors = attractor.emotional_attractors()
  let point = vector.Vec3(0.7, 0.4, 0.3)
  // Near joy

  let weights_warm = attractor.basin_weights(point, attractors, 2.0)
  let weights_cold = attractor.basin_weights(point, attractors, 0.5)

  let max_warm =
    list.fold(weights_warm, 0.0, fn(acc, p) { float.max(acc, p.1) })
  let max_cold =
    list.fold(weights_cold, 0.0, fn(acc, p) { float.max(acc, p.1) })

  // Cold (low temp) should have higher max weight
  should.be_true(max_cold >. max_warm)
}

// ============================================================================
// NEW: Hybrid entropy tests
// ============================================================================

pub fn hybrid_entropy_blend_test() {
  // Blend of two distributions
  let p1 = [0.5, 0.5]
  // H = 1.0
  let p2 = [1.0, 0.0]
  // H = 0.0

  let h_blend = entropy.hybrid_shannon(p1, p2, 0.5)
  // Should be average: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
  should.be_true(is_close(h_blend, 0.5, 0.01))
}

pub fn hybrid_entropy_alpha_zero_test() {
  // Alpha = 0 should give H(p2)
  let p1 = [0.5, 0.5]
  // H = 1.0
  let p2 = [1.0, 0.0]
  // H = 0.0

  let h = entropy.hybrid_shannon(p1, p2, 0.0)
  should.be_true(is_close(h, 0.0, 0.01))
}

pub fn hybrid_entropy_alpha_one_test() {
  // Alpha = 1 should give H(p1)
  let p1 = [0.5, 0.5]
  // H = 1.0
  let p2 = [1.0, 0.0]
  // H = 0.0

  let h = entropy.hybrid_shannon(p1, p2, 1.0)
  should.be_true(is_close(h, 1.0, 0.01))
}

// ============================================================================
// NEW: KL with sensitivity tests
// ============================================================================

pub fn kl_sensitivity_standard_test() {
  // Standard sensitivity should match regular KL
  let p = [0.5, 0.5]
  let q = [0.6, 0.4]

  let assert Ok(kl_standard) = entropy.kl_divergence(p, q)
  let assert Ok(kl_sens) =
    entropy.kl_divergence_with_sensitivity(p, q, entropy.Standard)

  should.be_true(is_close(kl_standard, kl_sens, 0.001))
}

pub fn kl_sensitivity_arousal_increases_test() {
  // Higher arousal should increase KL (more sensitive)
  let p = [0.9, 0.1]
  let q = [0.5, 0.5]

  let assert Ok(kl_low) =
    entropy.kl_divergence_with_sensitivity(p, q, entropy.ArousalWeighted(0.2))
  let assert Ok(kl_high) =
    entropy.kl_divergence_with_sensitivity(p, q, entropy.ArousalWeighted(0.8))

  should.be_true(kl_high >. kl_low)
}

// ============================================================================
// NEW: Renyi entropy tests
// ============================================================================

pub fn renyi_order_one_is_shannon_test() {
  // Renyi entropy with α=1 should equal Shannon entropy
  let p = [0.5, 0.5]
  let assert Ok(h_renyi) = entropy.renyi(p, 1.0)
  let h_shannon = entropy.shannon(p)
  should.be_true(is_close(h_renyi, h_shannon, 0.001))
}

pub fn renyi_order_two_collision_test() {
  // Renyi entropy with α=2 (collision entropy)
  // For uniform [0.5, 0.5]: H_2 = -log2(0.5² + 0.5²) = -log2(0.5) = 1.0
  let p = [0.5, 0.5]
  let assert Ok(h2) = entropy.renyi(p, 2.0)
  should.be_true(is_close(h2, 1.0, 0.01))
}

// ============================================================================
// NEW: Full KL divergence tests
// ============================================================================

pub fn gaussian_kl_full_equal_variance_test() {
  // When variances are equal, full KL should reduce to simple form
  let mean1 = vector.Vec3(0.5, 0.0, 0.0)
  let mean2 = vector.Vec3(0.0, 0.0, 0.0)

  let kl_simple = free_energy.gaussian_kl_divergence(mean1, mean2, 1.0)
  let kl_full = free_energy.gaussian_kl_divergence_full(mean1, mean2, 1.0, 1.0)

  // Full KL with equal variances = log(1) + (σ² + d²)/(2σ²) - 0.5
  // = 0 + (1 + 0.25)/2 - 0.5 = 0.625 - 0.5 = 0.125
  // Simple KL = d²/(2σ²) = 0.25/2 = 0.125
  should.be_true(is_close(kl_simple, kl_full, 0.01))
}

// ============================================================================
// scalar.gleam tests
// ============================================================================

pub fn scalar_erf_zero_test() {
  should.be_true(is_close(scalar.erf(0.0), 0.0, 1.0e-9))
}

pub fn scalar_erf_one_test() {
  // erf(1) ≈ 0.8427007929
  should.be_true(is_close(scalar.erf(1.0), 0.8427007929, 1.0e-6))
}

pub fn scalar_erfc_complement_test() {
  // erf(x) + erfc(x) = 1
  let x = 0.5
  should.be_true(is_close(scalar.erf(x) +. scalar.erfc(x), 1.0, 1.0e-9))
}

pub fn scalar_gelu_zero_test() {
  // GELU(0) = 0
  should.be_true(is_close(scalar.gelu(0.0), 0.0, 1.0e-9))
}

pub fn scalar_gelu_large_test() {
  // GELU(large positive) ≈ identity
  should.be_true(is_close(scalar.gelu(10.0), 10.0, 1.0e-3))
}

pub fn scalar_silu_zero_test() {
  // SiLU(0) = 0 · σ(0) = 0
  should.be_true(is_close(scalar.silu(0.0), 0.0, 1.0e-9))
}

pub fn scalar_softplus_zero_test() {
  // softplus(0) = ln(2)
  should.be_true(is_close(scalar.softplus(0.0), constants.ln_2, 1.0e-6))
}

pub fn scalar_logsumexp_test() {
  // logsumexp([0, 0]) = ln(2)
  should.be_true(is_close(scalar.logsumexp([0.0, 0.0]), constants.ln_2, 1.0e-6))
}

pub fn scalar_relu_test() {
  scalar.relu(-1.0) |> should.equal(0.0)
  scalar.relu(2.5) |> should.equal(2.5)
}

pub fn scalar_hypot_test() {
  // 3-4-5 right triangle
  should.be_true(is_close(scalar.hypot(3.0, 4.0), 5.0, 1.0e-9))
}

// ============================================================================
// constants.gleam tests
// ============================================================================

pub fn constants_pi_test() {
  should.be_true(is_close(constants.pi, 3.14159265, 1.0e-6))
}

pub fn constants_tau_test() {
  should.be_true(is_close(constants.tau, 2.0 *. constants.pi, 1.0e-9))
}

pub fn constants_sqrt_2_squared_test() {
  should.be_true(is_close(constants.sqrt_2 *. constants.sqrt_2, 2.0, 1.0e-9))
}

// ============================================================================
// statistics.gleam tests
// ============================================================================

pub fn statistics_mean_test() {
  let assert Ok(m) = statistics.mean([1.0, 2.0, 3.0, 4.0, 5.0])
  should.be_true(is_close(m, 3.0, 1.0e-9))
}

pub fn statistics_variance_test() {
  // Var([2, 4, 4, 4, 5, 5, 7, 9]) = 4 (population)
  let assert Ok(v) =
    statistics.variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  should.be_true(is_close(v, 4.0, 1.0e-6))
}

pub fn statistics_median_odd_test() {
  let assert Ok(m) = statistics.median([1.0, 3.0, 2.0])
  should.be_true(is_close(m, 2.0, 1.0e-9))
}

pub fn statistics_median_even_test() {
  let assert Ok(m) = statistics.median([1.0, 2.0, 3.0, 4.0])
  should.be_true(is_close(m, 2.5, 1.0e-9))
}

pub fn statistics_ema_test() {
  let assert Ok(series) = statistics.ema([1.0, 2.0, 3.0], 0.5)
  // y0 = 1, y1 = 0.5·2 + 0.5·1 = 1.5, y2 = 0.5·3 + 0.5·1.5 = 2.25
  should.equal(list.length(series), 3)
}

pub fn statistics_percentile_test() {
  let assert Ok(q3) = statistics.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.75)
  should.be_true(is_close(q3, 4.0, 1.0e-6))
}

// ============================================================================
// random.gleam tests
// ============================================================================

pub fn random_uniform_reproducible_test() {
  let seed = random.from_int(42)
  let #(a, _) = random.uniform(seed)
  let seed2 = random.from_int(42)
  let #(b, _) = random.uniform(seed2)
  // Same seed -> identical sequence
  should.equal(a, b)
}

pub fn random_uniform_in_range_test() {
  let seed = random.from_int(7)
  let #(x, _) = random.uniform(seed)
  should.be_true(x >=. 0.0 && x <. 1.0)
}

pub fn random_normal_works_test() {
  let seed = random.from_int(123)
  let #(_, _) = random.normal(seed, 0.0, 1.0)
  should.be_true(True)
}

pub fn random_categorical_test() {
  let seed = random.from_int(1)
  let assert Ok(#(idx, _)) = random.categorical(seed, [0.0, 1.0, 0.0])
  // Only second category has mass
  should.equal(idx, 1)
}

// ============================================================================
// distributions.gleam tests
// ============================================================================

pub fn distributions_gaussian_pdf_at_mean_test() {
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  // PDF at mean = 1/√(2π) ≈ 0.3989
  should.be_true(is_close(
    distributions.gaussian_pdf(g, 0.0),
    constants.inv_sqrt_2pi,
    1.0e-6,
  ))
}

pub fn distributions_gaussian_cdf_half_test() {
  // F(μ) = 0.5
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  should.be_true(is_close(distributions.gaussian_cdf(g, 0.0), 0.5, 1.0e-9))
}

pub fn distributions_uniform_pdf_test() {
  let u = distributions.Uniform(low: 0.0, high: 2.0)
  should.be_true(is_close(distributions.uniform_pdf(u, 1.0), 0.5, 1.0e-9))
}

// ============================================================================
// scheduler.gleam tests
// ============================================================================

pub fn scheduler_linear_warmup_test() {
  scheduler.linear_warmup(1.0, 0, 10) |> should.equal(0.0)
  scheduler.linear_warmup(1.0, 10, 10) |> should.equal(1.0)
  should.be_true(is_close(scheduler.linear_warmup(1.0, 5, 10), 0.5, 1.0e-9))
}

pub fn scheduler_cosine_at_zero_test() {
  // At step 0, cosine annealing = base_lr
  should.be_true(is_close(
    scheduler.cosine_annealing(1.0, 0, 100, 0.0),
    1.0,
    1.0e-9,
  ))
}

pub fn scheduler_cosine_at_end_test() {
  // At step T_max, cosine annealing = min_lr
  should.be_true(is_close(
    scheduler.cosine_annealing(1.0, 100, 100, 0.0),
    0.0,
    1.0e-9,
  ))
}

pub fn scheduler_exponential_test() {
  should.be_true(is_close(scheduler.exponential(1.0, 2, 0.5), 0.25, 1.0e-9))
}

// ============================================================================
// ode.gleam tests
// ============================================================================

pub fn ode_euler_constant_test() {
  // dx/dt = 1, x(0) = 0, dt = 1 -> x(1) = 1
  let f = fn(_t: Float, _x: Float) { 1.0 }
  should.be_true(is_close(ode.euler(f, 0.0, 0.0, 1.0), 1.0, 1.0e-9))
}

pub fn ode_rk4_quadratic_test() {
  // dx/dt = 2t, x(0) = 0 -> x(1) = 1 (exact integration of 2t)
  let f = fn(t: Float, _x: Float) { 2.0 *. t }
  let result = ode.rk4(f, 0.0, 0.0, 1.0)
  should.be_true(is_close(result, 1.0, 1.0e-6))
}

pub fn ode_integrate_length_test() {
  let f = fn(_t: Float, _x: Float) { 1.0 }
  let traj = ode.integrate(ode.euler, f, 0.0, 0.0, 0.1, 10)
  // 1 initial + 10 steps
  should.equal(list.length(traj), 11)
}

// ============================================================================
// calculus.gleam tests
// ============================================================================

pub fn calculus_central_diff_test() {
  // d/dx (x²) at x=2 is 4
  let f = fn(x: Float) { x *. x }
  should.be_true(is_close(calculus.central_diff(f, 2.0, 1.0e-4), 4.0, 1.0e-4))
}

pub fn calculus_trapezoid_test() {
  // ∫₀¹ x dx = 0.5
  let f = fn(x: Float) { x }
  should.be_true(is_close(calculus.trapezoid(f, 0.0, 1.0, 100), 0.5, 1.0e-6))
}

pub fn calculus_simpson_test() {
  // ∫₀² x² dx = 8/3
  let f = fn(x: Float) { x *. x }
  let assert Ok(result) = calculus.simpson(f, 0.0, 2.0, 100)
  should.be_true(is_close(result, 8.0 /. 3.0, 1.0e-6))
}

// ============================================================================
// matrix.gleam tests
// ============================================================================

pub fn matrix_mat2_identity_mul_test() {
  let i = matrix.mat2_identity()
  let m = matrix.Mat2(2.0, 3.0, 4.0, 5.0)
  let result = matrix.mat2_mul(i, m)
  should.equal(result.m11, 2.0)
  should.equal(result.m22, 5.0)
}

pub fn matrix_mat2_inverse_test() {
  let m = matrix.Mat2(4.0, 7.0, 2.0, 6.0)
  let assert Ok(inv) = matrix.mat2_inverse(m)
  // M · M⁻¹ should be identity
  let prod = matrix.mat2_mul(m, inv)
  should.be_true(is_close(prod.m11, 1.0, 1.0e-9))
  should.be_true(is_close(prod.m22, 1.0, 1.0e-9))
  should.be_true(is_close(prod.m12, 0.0, 1.0e-9))
}

pub fn matrix_mat3_determinant_identity_test() {
  should.be_true(is_close(
    matrix.mat3_determinant(matrix.mat3_identity()),
    1.0,
    1.0e-9,
  ))
}

pub fn matrix_mat3_mul_vec3_test() {
  let i = matrix.mat3_identity()
  let v = Vec3(1.0, 2.0, 3.0)
  let result = matrix.mat3_mul_vec3(i, v)
  should.equal(result, v)
}

pub fn matrix_matn_transpose_test() {
  let assert Ok(m) = matrix.matn_from_rows([[1.0, 2.0], [3.0, 4.0]])
  let t = matrix.matn_transpose(m)
  should.equal(t.rows, 2)
  should.equal(t.cols, 2)
}

// ============================================================================
// vec2 / vec4 / vecn tests
// ============================================================================

pub fn vec2_length_test() {
  should.be_true(is_close(vec2.length(vec2.Vec2(3.0, 4.0)), 5.0, 1.0e-9))
}

pub fn vec2_rotate_quarter_turn_test() {
  // Rotating (1, 0) by π/2 gives ~(0, 1)
  let rotated = vec2.rotate(vec2.Vec2(1.0, 0.0), constants.half_pi)
  should.be_true(vec2.is_close(rotated, vec2.Vec2(0.0, 1.0), 1.0e-9))
}

pub fn vec4_dot_test() {
  let a = vec4.Vec4(1.0, 2.0, 3.0, 4.0)
  let b = vec4.Vec4(1.0, 1.0, 1.0, 1.0)
  should.be_true(is_close(vec4.dot(a, b), 10.0, 1.0e-9))
}

pub fn vecn_add_test() {
  let assert Ok(result) = vecn.add([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
  should.equal(result, [11.0, 22.0, 33.0])
}

pub fn vecn_dot_test() {
  let assert Ok(d) = vecn.dot([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
  should.be_true(is_close(d, 6.0, 1.0e-9))
}

// ============================================================================
// entropy extensions tests
// ============================================================================

pub fn entropy_tsallis_q_one_test() {
  // Tsallis with q → 1 should approach Shannon
  let probs = [0.5, 0.5]
  let assert Ok(t) = entropy.tsallis(probs, 1.0)
  should.be_true(is_close(t, entropy.shannon(probs), 1.0e-6))
}

pub fn entropy_fisher_test() {
  let assert Ok(i) = entropy.fisher_information_gaussian(2.0)
  should.be_true(is_close(i, 0.25, 1.0e-9))
}

// ============================================================================
// free_energy extensions tests
// ============================================================================

pub fn expected_free_energy_zero_distance_test() {
  let g =
    free_energy.expected_free_energy(
      vector.Vec3(0.0, 0.0, 0.0),
      vector.Vec3(0.0, 0.0, 0.0),
      0.5,
    )
  should.be_true(is_close(g.epistemic, 0.5, 1.0e-9))
  should.be_true(is_close(g.pragmatic, 0.0, 1.0e-9))
}

// ============================================================================
// Helper functions
// ============================================================================

fn is_close(a: Float, b: Float, tolerance: Float) -> Bool {
  float.absolute_value(a -. b) <=. tolerance
}

// ============================================================================
// transport.gleam tests
// ============================================================================

pub fn wasserstein_1_identical_test() {
  let assert Ok(distance) =
    transport.wasserstein_1_empirical([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

  should.be_true(is_close(distance, 0.0, 1.0e-9))
}

pub fn wasserstein_1_translation_test() {
  let assert Ok(distance) =
    transport.wasserstein_1_empirical([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

  should.be_true(is_close(distance, 1.0, 1.0e-9))
}

pub fn wasserstein_2_gaussian_identical_test() {
  let distance =
    transport.wasserstein_2_gaussian(
      distributions.Gaussian(mean: 0.0, stddev: 1.0),
      distributions.Gaussian(mean: 0.0, stddev: 1.0),
    )

  should.be_true(is_close(distance, 0.0, 1.0e-9))
}

pub fn wasserstein_2_gaussian_known_test() {
  let distance =
    transport.wasserstein_2_gaussian(
      distributions.Gaussian(mean: 0.0, stddev: 1.0),
      distributions.Gaussian(mean: 2.0, stddev: 1.0),
    )

  should.be_true(is_close(distance, 2.0, 1.0e-9))
}

pub fn wasserstein_pad_zero_test() {
  let pads = [
    vector.pad(0.2, -0.1, 0.7),
    vector.pad(-0.4, 0.5, -0.2),
  ]
  let assert Ok(distance) = transport.wasserstein_pad(pads, pads)

  should.be_true(is_close(distance, 0.0, 1.0e-9))
}

pub fn wasserstein_empty_error_test() {
  transport.wasserstein_1_empirical([], [])
  |> should.equal(Error(Nil))
}

// ============================================================================
// ou.gleam tests
// ============================================================================

pub fn ou_mean_at_initial_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 1.0, sigma: 0.2)
  // E[X_0] = x_0
  ou.mean_at(params, 0.7, 0.0)
  |> is_close(0.7, 1.0e-12)
  |> should.be_true
}

pub fn ou_mean_at_converges_to_mu_test() {
  let params = ou.OUParams1D(theta: 1.0, mu: 2.5, sigma: 0.5)
  // After many time constants, mean → μ
  ou.mean_at(params, 0.0, 50.0)
  |> is_close(2.5, 1.0e-12)
  |> should.be_true
}

pub fn ou_variance_at_initial_is_zero_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 0.0, sigma: 1.0)
  ou.variance_at(params, 0.3, 0.0)
  |> is_close(0.0, 1.0e-12)
  |> should.be_true
}

pub fn ou_stationary_variance_test() {
  let params = ou.OUParams1D(theta: 2.0, mu: 0.0, sigma: 4.0)
  // σ²/(2θ) = 16/4 = 4.0
  ou.stationary_variance(params)
  |> is_close(4.0, 1.0e-12)
  |> should.be_true
}

pub fn ou_autocovariance_at_zero_lag_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 0.0, sigma: 1.0)
  let stationary = ou.stationary_variance(params)
  ou.autocovariance(params, 0.0)
  |> is_close(stationary, 1.0e-12)
  |> should.be_true
}

pub fn ou_half_life_test() {
  let params = ou.OUParams1D(theta: 1.0, mu: 0.0, sigma: 1.0)
  // ln(2)/θ ≈ 0.6931
  ou.half_life(params)
  |> is_close(0.693_147_180_559_945, 1.0e-9)
  |> should.be_true
}

pub fn ou_step_zero_sigma_is_deterministic_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 1.0, sigma: 0.0)
  let seed = random.from_int(42)
  let #(x_next, _) = ou.step(params, 0.0, 1.0, seed)
  // With σ=0, step equals mean_at exactly
  let expected = ou.mean_at(params, 0.0, 1.0)
  x_next
  |> is_close(expected, 1.0e-12)
  |> should.be_true
}

pub fn ou_simulate_length_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 0.0, sigma: 0.1)
  let seed = random.from_int(7)
  let #(traj, _) = ou.simulate(params, 0.0, 0.1, 100, seed)
  list.length(traj)
  |> should.equal(100)
}

pub fn ou_vec3_mean_converges_test() {
  let params =
    ou.OUParamsVec3(
      theta: Vec3(1.0, 1.0, 1.0),
      mu: Vec3(0.3, -0.2, 0.5),
      sigma: Vec3(0.1, 0.1, 0.1),
    )
  let x0 = Vec3(0.0, 0.0, 0.0)
  let m = ou.mean_at_vec3(params, x0, 50.0)
  is_close(m.x, 0.3, 1.0e-12)
  |> should.be_true
  is_close(m.y, -0.2, 1.0e-12)
  |> should.be_true
  is_close(m.z, 0.5, 1.0e-12)
  |> should.be_true
}

pub fn ou_is_valid_test() {
  ou.is_valid(ou.OUParams1D(theta: 1.0, mu: 0.0, sigma: 0.5))
  |> should.be_true
  ou.is_valid(ou.OUParams1D(theta: -1.0, mu: 0.0, sigma: 0.5))
  |> should.be_false
  ou.is_valid(ou.OUParams1D(theta: 1.0, mu: 0.0, sigma: -0.1))
  |> should.be_false
}

// ============================================================================
// free_energy.gleam — Variational (Bayesian) extension tests
// ============================================================================

pub fn vfe_scalar_gaussian_kl_self_zero_test() {
  free_energy.scalar_gaussian_kl(0.5, 1.0, 0.5, 1.0)
  |> is_close(0.0, 1.0e-12)
  |> should.be_true
}

pub fn vfe_scalar_gaussian_kl_known_test() {
  // D_KL(N(1,1) || N(0,1)) = (1-0)² / (2·1) = 0.5
  free_energy.scalar_gaussian_kl(1.0, 1.0, 0.0, 1.0)
  |> is_close(0.5, 1.0e-12)
  |> should.be_true
}

pub fn vfe_mean_field_update_no_observations_test() {
  // Sem observações: posterior = prior
  let assert Ok(mf) = free_energy.mean_field_update([], 2.0, 1.0, 0.5)
  is_close(mf.q_mean, 2.0, 1.0e-12)
  |> should.be_true
  is_close(mf.q_var, 1.0, 1.0e-12)
  |> should.be_true
}

pub fn vfe_mean_field_update_one_obs_test() {
  // Prior N(0,1), likelihood var=1, observation=2
  // posterior_prec = 1 + 1·1 = 2 → posterior_var = 0.5
  // posterior_mean = 0.5 · (1·0 + 1·2) = 1.0
  let assert Ok(mf) = free_energy.mean_field_update([2.0], 0.0, 1.0, 1.0)
  is_close(mf.q_mean, 1.0, 1.0e-12)
  |> should.be_true
  is_close(mf.q_var, 0.5, 1.0e-12)
  |> should.be_true
}

pub fn vfe_mean_field_update_reduces_variance_test() {
  let assert Ok(mf_few) = free_energy.mean_field_update([1.0], 0.0, 1.0, 1.0)
  let assert Ok(mf_many) =
    free_energy.mean_field_update(
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      0.0,
      1.0,
      1.0,
    )
  // mais observações → menor variância posterior
  { mf_many.q_var <. mf_few.q_var }
  |> should.be_true
}

pub fn vfe_mean_field_update_invalid_variance_test() {
  free_energy.mean_field_update([1.0], 0.0, -1.0, 1.0)
  |> should.equal(Error(Nil))
}

pub fn vfe_elbo_lower_bounds_log_evidence_test() {
  // Sob conjugado Gaussiano, posterior exato = MAP estimator.
  // ELBO @ posterior exato = log p(x). ELBO @ outro q ≤ log p(x).
  let obs = 1.5
  let prior_mean = 0.0
  let prior_var = 1.0
  let lik_var = 1.0
  let assert Ok(mf_exact) =
    free_energy.mean_field_update([obs], prior_mean, prior_var, lik_var)
  let log_evidence =
    free_energy.log_evidence_gaussian(obs, prior_mean, prior_var, lik_var)
  let elbo_exact =
    free_energy.elbo(
      obs,
      mf_exact.q_mean,
      mf_exact.q_var,
      prior_mean,
      prior_var,
      lik_var,
    )
  // Posterior exato satura o bound: ELBO == log p(x).
  is_close(elbo_exact.total, log_evidence, 1.0e-9)
  |> should.be_true
  // Posterior subótimo é estritamente menor.
  let elbo_bad = free_energy.elbo(obs, 5.0, 0.1, prior_mean, prior_var, lik_var)
  { elbo_bad.total <. log_evidence }
  |> should.be_true
}

pub fn vfe_laplace_quadratic_test() {
  // Para log_posterior(z) = -½ (z - 2)² / 0.25 (≡ N(2, 0.25)), Laplace
  // recupera mean=2.0, var=0.25 exatamente.
  let log_post = fn(z: Float) {
    let d = z -. 2.0
    0.0 -. 0.5 *. d *. d /. 0.25
  }
  let assert Ok(mf) = free_energy.laplace_approximation(log_post, 0.0, 0.1, 200)
  is_close(mf.q_mean, 2.0, 1.0e-3)
  |> should.be_true
  is_close(mf.q_var, 0.25, 1.0e-3)
  |> should.be_true
}

pub fn vfe_log_evidence_gaussian_known_test() {
  // log p(x) when x=0, prior=N(0,1), lik var=1 → marginal=N(0,2)
  // log p(0) = -½·ln(2π·2) - 0 = -½·ln(4π)
  let lp = free_energy.log_evidence_gaussian(0.0, 0.0, 1.0, 1.0)
  let expected =
    0.0
    -. 0.5
    *. {
      case scalar.try_ln(4.0 *. constants.pi) {
        Ok(l) -> l
        Error(_) -> 0.0
      }
    }
  is_close(lp, expected, 1.0e-12)
  |> should.be_true
}

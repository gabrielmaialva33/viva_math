//// Invariant tests — mathematical identities that must hold across the
//// input domain. Renamed from `property_test.gleam` to disambiguate from
//// `qcheck_test.gleam`, which uses generative property-based testing.
////
//// Unlike `precision_test.gleam` (which checks specific golden values), this
//// file verifies universal identities like "PDF integrates to 1" or
//// "softmax sums to 1" by sampling across the input domain.

import gleam/float
import gleam/list
import gleeunit/should
import viva_math/calculus
import viva_math/constants
import viva_math/distributions
import viva_math/entropy
import viva_math/free_energy
import viva_math/matrix
import viva_math/random
import viva_math/scalar
import viva_math/statistics
import viva_math/vec2
import viva_math/vec4
import viva_math/vecn
import viva_math/vector

const tight: Float = 1.0e-9

const loose: Float = 1.0e-6

fn approx(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

// ============================================================================
// scalar - functional identities
// ============================================================================

pub fn property_relu_idempotent_test() {
  // relu(relu(x)) = relu(x)
  let xs = [-3.0, -0.5, 0.0, 1.0, 7.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.relu(scalar.relu(x)), scalar.relu(x), tight))
  })
}

pub fn property_sign_consistency_test() {
  // sign(x) * |x| = x
  let xs = [-7.0, -1.5, 0.0, 1.5, 7.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.sign(x) *. float.absolute_value(x), x, tight))
  })
}

pub fn property_smoothstep_endpoints_test() {
  // smoothstep(a, b, a) = 0, smoothstep(a, b, b) = 1, regardless of a,b
  let edges = [#(0.0, 1.0), #(-2.0, 3.0), #(10.0, 20.0)]
  list.each(edges, fn(pair) {
    let #(a, b) = pair
    should.be_true(approx(scalar.smoothstep(a, b, a), 0.0, tight))
    should.be_true(approx(scalar.smoothstep(a, b, b), 1.0, tight))
  })
}

pub fn property_smoothstep_degenerate_test() {
  // Degenerate edges (a == b) behave as step function
  should.be_true(approx(scalar.smoothstep(5.0, 5.0, 3.0), 0.0, tight))
  should.be_true(approx(scalar.smoothstep(5.0, 5.0, 7.0), 1.0, tight))
}

pub fn property_gelu_zero_at_zero_test() {
  should.be_true(approx(scalar.gelu(0.0), 0.0, tight))
  should.be_true(approx(scalar.gelu_approx(0.0), 0.0, tight))
}

pub fn property_gelu_approx_close_to_exact_test() {
  // GELU exact vs tanh approximation differ by less than 5e-4 in practice
  let xs = [-3.0, -1.0, -0.1, 0.1, 1.0, 3.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.gelu(x), scalar.gelu_approx(x), 5.0e-4))
  })
}

pub fn property_softplus_positive_test() {
  // softplus(x) > 0 for all finite x
  let xs = [-100.0, -1.0, 0.0, 1.0, 100.0]
  list.each(xs, fn(x) { should.be_true(scalar.softplus(x) >. 0.0) })
}

pub fn property_softplus_monotonic_test() {
  // softplus is strictly increasing
  let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
  let sps = list.map(xs, scalar.softplus)
  let pairs = list.zip(sps, list.drop(sps, 1))
  list.each(pairs, fn(p) { should.be_true(p.0 <. p.1) })
}

pub fn property_sigmoid_bounded_test() {
  // 0 ≤ σ(x) ≤ 1 for all finite x. Mathematically the bounds are strict,
  // but IEEE-754 saturates to exact 0.0 / 1.0 for arguments past ±745 or so.
  let xs = [-1000.0, -1.0, 0.0, 1.0, 1000.0]
  list.each(xs, fn(x) {
    let s = scalar.sigmoid(x)
    should.be_true(s >=. 0.0)
    should.be_true(s <=. 1.0)
  })
}

pub fn property_tanh_bounded_test() {
  // |tanh(x)| ≤ 1
  let xs = [-1000.0, -1.0, 0.0, 1.0, 1000.0]
  list.each(xs, fn(x) {
    should.be_true(float.absolute_value(scalar.tanh(x)) <=. 1.0)
  })
}

pub fn property_logit_sigmoid_inverse_test() {
  // logit(σ(x)) ≈ x for moderate x
  let xs = [-5.0, -1.0, 0.0, 1.0, 5.0]
  list.each(xs, fn(x) {
    should.be_true(approx(scalar.logit(scalar.sigmoid(x)), x, 1.0e-7))
  })
}

// ============================================================================
// vector / vec2 / vec4 / vecn - length, normalize invariants
// ============================================================================

pub fn property_vec3_normalize_unit_length_test() {
  let vs = [
    vector.Vec3(1.0, 0.0, 0.0),
    vector.Vec3(3.0, 4.0, 0.0),
    vector.Vec3(1.0, 1.0, 1.0),
    vector.Vec3(-2.0, 5.0, -1.0),
  ]
  list.each(vs, fn(v) {
    let n = vector.normalize(v)
    should.be_true(approx(vector.length(n), 1.0, tight))
  })
}

pub fn property_vec3_dot_commutative_test() {
  let pairs = [
    #(vector.Vec3(1.0, 2.0, 3.0), vector.Vec3(4.0, 5.0, 6.0)),
    #(vector.Vec3(-1.0, 0.5, 7.0), vector.Vec3(2.0, -3.0, 1.0)),
  ]
  list.each(pairs, fn(p) {
    let a = p.0
    let b = p.1
    should.be_true(approx(vector.dot(a, b), vector.dot(b, a), tight))
  })
}

pub fn property_vec3_cross_perpendicular_test() {
  // a × b is perpendicular to both a and b
  let a = vector.Vec3(1.0, 2.0, 3.0)
  let b = vector.Vec3(-1.0, 4.0, 2.0)
  let c = vector.cross(a, b)
  should.be_true(approx(vector.dot(a, c), 0.0, tight))
  should.be_true(approx(vector.dot(b, c), 0.0, tight))
}

pub fn property_vec2_rotation_preserves_length_test() {
  let v = vec2.Vec2(3.0, 4.0)
  let l0 = vec2.length(v)
  let angles = [0.0, constants.quarter_pi, constants.half_pi, constants.pi]
  list.each(angles, fn(a) {
    should.be_true(approx(vec2.length(vec2.rotate(v, a)), l0, tight))
  })
}

pub fn property_vec4_normalize_unit_length_test() {
  let v = vec4.Vec4(1.0, 2.0, 3.0, 4.0)
  let n = vec4.normalize(v)
  should.be_true(approx(vec4.length(n), 1.0, tight))
}

pub fn property_vecn_dot_self_equals_length_squared_test() {
  let v = [1.0, 2.0, 3.0, 4.0]
  let assert Ok(d) = vecn.dot(v, v)
  should.be_true(approx(d, vecn.length_squared(v), tight))
}

// ============================================================================
// matrix - identity / transpose / determinant invariants
// ============================================================================

pub fn property_mat2_transpose_involutive_test() {
  let m = matrix.Mat2(1.0, 2.0, 3.0, 4.0)
  let mt = matrix.mat2_transpose(matrix.mat2_transpose(m))
  should.equal(mt, m)
}

pub fn property_mat3_identity_preserves_test() {
  let i = matrix.mat3_identity()
  let m =
    matrix.Mat3(
      m11: 1.0,
      m12: 2.0,
      m13: 3.0,
      m21: 4.0,
      m22: 5.0,
      m23: 6.0,
      m31: 7.0,
      m32: 8.0,
      m33: 9.0,
    )
  should.equal(matrix.mat3_mul(i, m), m)
  should.equal(matrix.mat3_mul(m, i), m)
}

pub fn property_mat3_det_product_test() {
  // det(AB) = det(A) · det(B)
  let a = matrix.mat3_diagonal(2.0, 3.0, 4.0)
  let b = matrix.mat3_diagonal(0.5, 0.5, 0.5)
  let ab = matrix.mat3_mul(a, b)
  let da = matrix.mat3_determinant(a)
  let db = matrix.mat3_determinant(b)
  let dab = matrix.mat3_determinant(ab)
  should.be_true(approx(dab, da *. db, tight))
}

pub fn property_mat3_rotation_det_is_one_test() {
  // det of rotation matrices is +1
  let r = matrix.mat3_rot_z(0.42)
  should.be_true(approx(matrix.mat3_determinant(r), 1.0, tight))
}

// ============================================================================
// statistics - invariants
// ============================================================================

pub fn property_mean_constant_test() {
  // Mean of a constant list = constant
  let assert Ok(m) = statistics.mean([3.5, 3.5, 3.5, 3.5])
  should.be_true(approx(m, 3.5, tight))
}

pub fn property_variance_zero_for_constant_test() {
  let assert Ok(v) = statistics.variance([42.0, 42.0, 42.0, 42.0])
  should.be_true(approx(v, 0.0, tight))
}

pub fn property_ema_step_extremes_test() {
  // α=1 → ema picks observation; α=0 → ema stays at previous
  should.be_true(approx(statistics.ema_step(10.0, 99.0, 1.0), 99.0, tight))
  should.be_true(approx(statistics.ema_step(10.0, 99.0, 0.0), 10.0, tight))
}

pub fn property_z_score_mean_zero_test() {
  // z-score of any non-constant list has mean 0 and stddev 1
  let assert Ok(zs) = statistics.z_score([1.0, 3.0, 5.0, 7.0, 9.0])
  let assert Ok(mz) = statistics.mean(zs)
  let assert Ok(sz) = statistics.stddev(zs)
  should.be_true(approx(mz, 0.0, tight))
  should.be_true(approx(sz, 1.0, tight))
}

// ============================================================================
// distributions - PDF integrates to 1
// ============================================================================

pub fn property_gaussian_pdf_integrates_to_one_test() {
  // ∫ pdf(x) dx ≈ 1 over [-6, 6] for N(0, 1)
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  let pdf = fn(x: Float) { distributions.gaussian_pdf(g, x) }
  let assert Ok(integral) = calculus.simpson(pdf, -6.0, 6.0, 200)
  should.be_true(approx(integral, 1.0, loose))
}

pub fn property_gaussian_cdf_bounded_test() {
  // 0 ≤ CDF(x) ≤ 1 always
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  let xs = [-100.0, -2.0, 0.0, 2.0, 100.0]
  list.each(xs, fn(x) {
    let c = distributions.gaussian_cdf(g, x)
    should.be_true(c >=. 0.0)
    should.be_true(c <=. 1.0)
  })
}

pub fn property_gaussian_cdf_monotone_test() {
  let g = distributions.Gaussian(mean: 0.0, stddev: 1.0)
  let xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
  let cdfs = list.map(xs, fn(x) { distributions.gaussian_cdf(g, x) })
  let pairs = list.zip(cdfs, list.drop(cdfs, 1))
  list.each(pairs, fn(p) { should.be_true(p.0 <=. p.1) })
}

pub fn property_uniform_pdf_integrates_to_one_test() {
  let u = distributions.Uniform(low: 0.0, high: 4.0)
  let pdf = fn(x: Float) { distributions.uniform_pdf(u, x) }
  let assert Ok(integral) = calculus.simpson(pdf, 0.0, 4.0, 100)
  should.be_true(approx(integral, 1.0, loose))
}

pub fn property_exponential_pdf_integrates_to_one_test() {
  let e = distributions.Exponential(rate: 1.5)
  let pdf = fn(x: Float) { distributions.exponential_pdf(e, x) }
  let assert Ok(integral) = calculus.simpson(pdf, 0.0, 30.0, 600)
  should.be_true(approx(integral, 1.0, loose))
}

// ============================================================================
// entropy - non-negativity, additivity
// ============================================================================

pub fn property_shannon_non_negative_test() {
  // H ≥ 0 for any valid distribution
  let probs = [0.1, 0.2, 0.3, 0.4]
  should.be_true(entropy.shannon(probs) >=. 0.0)
}

pub fn property_shannon_max_uniform_test() {
  // Maximum entropy for n outcomes = log₂(n), achieved by uniform distribution
  let uniform = [0.25, 0.25, 0.25, 0.25]
  let max_h = entropy.shannon(uniform)
  // log₂(4) = 2
  should.be_true(approx(max_h, 2.0, tight))
}

pub fn property_shannon_zero_for_certain_test() {
  // Distribution concentrated on one outcome has zero entropy
  should.be_true(approx(entropy.shannon([1.0, 0.0, 0.0]), 0.0, tight))
}

pub fn property_kl_zero_self_test() {
  // D_KL(P || P) = 0
  let p = [0.2, 0.3, 0.5]
  let assert Ok(kl) = entropy.kl_divergence(p, p)
  should.be_true(approx(kl, 0.0, tight))
}

pub fn property_kl_non_negative_test() {
  // D_KL ≥ 0 (Gibbs' inequality)
  let p = [0.2, 0.3, 0.5]
  let q = [0.5, 0.3, 0.2]
  let assert Ok(kl) = entropy.kl_divergence(p, q)
  should.be_true(kl >=. 0.0)
}

pub fn property_jensen_shannon_symmetric_test() {
  // JS(P, Q) = JS(Q, P)
  let p = [0.4, 0.3, 0.3]
  let q = [0.2, 0.5, 0.3]
  let assert Ok(js_pq) = entropy.jensen_shannon(p, q)
  let assert Ok(js_qp) = entropy.jensen_shannon(q, p)
  should.be_true(approx(js_pq, js_qp, tight))
}

pub fn property_tsallis_concentrates_to_zero_test() {
  // Concentrated distribution -> S_q = 0 for q != 1
  let assert Ok(t) = entropy.tsallis([1.0, 0.0, 0.0], 2.0)
  should.be_true(approx(t, 0.0, tight))
}

// ============================================================================
// free_energy - invariants
// ============================================================================

pub fn property_kl_self_zero_test() {
  // KL(N(μ, σ²) || N(μ, σ²)) = 0
  let mu = vector.Vec3(0.5, 1.0, -0.3)
  let kl = free_energy.gaussian_kl_divergence_full(mu, mu, 1.0, 1.0)
  should.be_true(approx(kl, 0.0, tight))
}

pub fn property_kl_full_3d_test() {
  // For Vec3 with equal variances and zero mean difference, KL = 0.
  let v = vector.Vec3(0.0, 0.0, 0.0)
  let kl = free_energy.gaussian_kl_divergence_full(v, v, 2.0, 2.0)
  should.be_true(approx(kl, 0.0, tight))
}

pub fn property_policy_posterior_sums_to_one_test() {
  // Probabilities of selecting each policy must sum to 1.
  let policies = [
    #("a", vector.Vec3(1.0, 0.0, 0.0), 0.5),
    #("b", vector.Vec3(0.0, 1.0, 0.0), 0.3),
    #("c", vector.Vec3(0.0, 0.0, 1.0), 0.8),
  ]
  let preferred = vector.Vec3(0.5, 0.5, 0.5)
  let dist = free_energy.policy_posterior(policies, preferred, 1.0)
  let total = list.fold(dist, 0.0, fn(acc, p) { acc +. p.1 })
  should.be_true(approx(total, 1.0, tight))
}

pub fn property_policy_posterior_empty_test() {
  // Empty policy list returns empty distribution.
  let dist = free_energy.policy_posterior([], vector.Vec3(0.0, 0.0, 0.0), 1.0)
  should.equal(dist, [])
}

// ============================================================================
// scalar.logsumexp - max-subtract identity
// ============================================================================

pub fn property_logsumexp_shift_invariance_test() {
  // logsumexp(x + c) = c + logsumexp(x) for any constant c
  let xs = [0.5, 1.5, -2.0, 3.5]
  let c = 5.0
  let shifted = list.map(xs, fn(x) { x +. c })
  let lhs = scalar.logsumexp(shifted)
  let rhs = c +. scalar.logsumexp(xs)
  should.be_true(approx(lhs, rhs, tight))
}

// ============================================================================
// random - statistical properties (loose tolerance)
// ============================================================================

pub fn property_random_uniform_in_range_test() {
  // Draw many samples — all should land in [0, 1)
  let seed = random.from_int(7)
  let #(samples, _) = random.uniforms(seed, 1000)
  list.each(samples, fn(x) { should.be_true(x >=. 0.0 && x <. 1.0) })
}

pub fn property_random_normal_zero_finite_test() {
  // Sampling N(0, 0) should give exactly mu (variance 0 collapses).
  let seed = random.from_int(0)
  let #(x, _) = random.normal(seed, 7.5, 0.0)
  should.be_true(approx(x, 7.5, tight))
}

pub fn property_random_bernoulli_extremes_test() {
  // p=1.0 always True, p=0.0 always False.
  let seed = random.from_int(0)
  let #(t, _) = random.bernoulli(seed, 1.0)
  should.equal(t, True)
  let #(f, _) = random.bernoulli(seed, 0.0)
  should.equal(f, False)
}

pub fn property_random_bernoulli_clamps_test() {
  // p < 0 treated as 0, p > 1 treated as 1.
  let seed = random.from_int(0)
  let #(t, _) = random.bernoulli(seed, 5.0)
  should.equal(t, True)
  let #(f, _) = random.bernoulli(seed, -1.0)
  should.equal(f, False)
}

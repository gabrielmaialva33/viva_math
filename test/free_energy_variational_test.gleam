//// Tests for the Variational Inference (Bayesian deepening) block of
//// `viva_math/free_energy` — ELBO, mean-field updates, Laplace, and
//// log evidence under Gaussian conjugate models. The non-variational
//// surface of `free_energy` is exercised by `viva_math_test.gleam`.

import gleeunit/should
import test_support.{is_close}
import viva_math/constants
import viva_math/free_energy
import viva_math/scalar

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
  let assert Ok(mf) = free_energy.mean_field_update([], 2.0, 1.0, 0.5)
  is_close(mf.q_mean, 2.0, 1.0e-12) |> should.be_true
  is_close(mf.q_var, 1.0, 1.0e-12) |> should.be_true
}

pub fn vfe_mean_field_update_one_obs_test() {
  // Prior N(0,1), likelihood var=1, observation=2 → q ~ N(1.0, 0.5).
  let assert Ok(mf) = free_energy.mean_field_update([2.0], 0.0, 1.0, 1.0)
  is_close(mf.q_mean, 1.0, 1.0e-12) |> should.be_true
  is_close(mf.q_var, 0.5, 1.0e-12) |> should.be_true
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
  { mf_many.q_var <. mf_few.q_var } |> should.be_true
}

pub fn vfe_mean_field_update_invalid_variance_test() {
  free_energy.mean_field_update([1.0], 0.0, -1.0, 1.0)
  |> should.equal(Error(Nil))
}

pub fn vfe_elbo_lower_bounds_log_evidence_test() {
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
  // Exact posterior saturates the Jensen bound.
  is_close(elbo_exact.total, log_evidence, 1.0e-9) |> should.be_true
  // Sub-optimal q is strictly below.
  let elbo_bad = free_energy.elbo(obs, 5.0, 0.1, prior_mean, prior_var, lik_var)
  { elbo_bad.total <. log_evidence } |> should.be_true
}

pub fn vfe_laplace_quadratic_test() {
  // log_posterior(z) = -½ (z - 2)² / 0.25  ≡  N(2, 0.25)
  let log_post = fn(z: Float) {
    let d = z -. 2.0
    0.0 -. 0.5 *. d *. d /. 0.25
  }
  let assert Ok(mf) = free_energy.laplace_approximation(log_post, 0.0, 0.1, 200)
  is_close(mf.q_mean, 2.0, 1.0e-3) |> should.be_true
  is_close(mf.q_var, 0.25, 1.0e-3) |> should.be_true
}

pub fn vfe_log_evidence_gaussian_known_test() {
  // x=0, prior=N(0,1), lik var=1 → marginal=N(0,2) → log p(0) = -½·ln(4π).
  let lp = free_energy.log_evidence_gaussian(0.0, 0.0, 1.0, 1.0)
  let expected =
    0.0
    -. 0.5
    *. case scalar.try_ln(4.0 *. constants.pi) {
      Ok(l) -> l
      Error(_) -> 0.0
    }
  is_close(lp, expected, 1.0e-12) |> should.be_true
}

// ============================================================================
// Deep-audit regression tests (CHANGELOG 1.2.102)
// ============================================================================

// ELBO must not exceed log p(x) even with q_var <= 0 (Jensen bound).
pub fn vfe_elbo_negative_q_var_does_not_break_bound_test() {
  let e = free_energy.elbo(0.0, 0.0, -10.0, 0.0, 1.0, 1.0)
  let logp = free_energy.log_evidence_gaussian(0.0, 0.0, 1.0, 1.0)
  should.be_true(e.total <=. logp +. 1.0e-9)
}

// log_evidence_gaussian must reject component-wise invalid variances even if
// their sum is positive.
pub fn vfe_log_evidence_rejects_componentwise_invalid_variance_test() {
  let lp = free_energy.log_evidence_gaussian(0.0, 0.0, 2.0, -1.0)
  should.be_true(lp <. -999_999.0)
}

// Sequential Bayes (mean_field_iterate) matches flat-batch mean_field_update
// (Bishop §2.3.6 — associativity of Gaussian-Gaussian conjugate updates).
pub fn vfe_mean_field_iterate_matches_flat_batch_test() {
  let prior = free_energy.MeanFieldParams(q_mean: 0.0, q_var: 1.0)
  let assert Ok(seq) =
    free_energy.mean_field_iterate([[1.0, 2.0], [3.0]], prior, 0.5)
  let assert Ok(flat) =
    free_energy.mean_field_update([1.0, 2.0, 3.0], 0.0, 1.0, 0.5)
  should.be_true(is_close(seq.q_mean, flat.q_mean, 1.0e-12))
  should.be_true(is_close(seq.q_var, flat.q_var, 1.0e-12))
}

// Laplace must reject near-zero curvature (would yield q_var = +∞).
pub fn vfe_laplace_rejects_flat_log_posterior_test() {
  let nearly_flat = fn(z: Float) { 0.0 -. 1.0e-300 *. z *. z }
  free_energy.laplace_approximation(nearly_flat, 0.0, 0.1, 50)
  |> should.equal(Error(Nil))
}

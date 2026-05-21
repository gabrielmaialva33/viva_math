//// Tests for `viva_math/transport` — 1D Wasserstein (W₁, W₂), closed-form
//// W₂ for Gaussians, and componentwise PAD aggregation.

import gleeunit/should
import test_support.{is_close, is_close_hybrid, loose, tight}
import viva_math/distributions
import viva_math/transport
import viva_math/vector

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
  let pads = [vector.pad(0.2, -0.1, 0.7), vector.pad(-0.4, 0.5, -0.2)]
  let assert Ok(distance) = transport.wasserstein_pad(pads, pads)
  should.be_true(is_close(distance, 0.0, 1.0e-9))
}

pub fn wasserstein_empty_error_test() {
  transport.wasserstein_1_empirical([], [])
  |> should.equal(Error(Nil))
}

// Triangle inequality across unequal sample sizes.
pub fn wasserstein_2_triangle_unequal_sizes_test() {
  let assert Ok(d_pr) = transport.wasserstein_2_empirical([0.0], [4.0])
  let assert Ok(d_pq) = transport.wasserstein_2_empirical([0.0], [1.0, 2.0])
  let assert Ok(d_qr) = transport.wasserstein_2_empirical([1.0, 2.0], [4.0])
  should.be_true(d_pr <=. d_pq +. d_qr +. 1.0e-9)
}

// Regression — see CHANGELOG 1.2.102: `wasserstein_2_gaussian` must normalise
// negative `stddev` via `abs` (no Gaussian meaning otherwise).
pub fn wasserstein_2_gaussian_negative_stddev_test() {
  let d =
    transport.wasserstein_2_gaussian(
      distributions.Gaussian(mean: 0.0, stddev: -1.0),
      distributions.Gaussian(mean: 0.0, stddev: 1.0),
    )
  should.be_true(is_close(d, 0.0, 1.0e-12))
}

pub fn wasserstein_2_multivariate_empty_error_test() {
  transport.wasserstein_2_multivariate([], [vector.zero()], 0.1, 100)
  |> should.equal(Error(Nil))
}

pub fn wasserstein_2_multivariate_identity_test() {
  let samples = [
    vector.pad(0.0, 0.0, 0.0),
    vector.pad(1.0, 0.0, 0.0),
    vector.pad(0.0, 1.0, 0.0),
  ]
  let assert Ok(distance) =
    transport.wasserstein_2_multivariate(samples, samples, 0.001, 100)
  should.be_true(is_close(distance, 0.0, loose))
}

pub fn wasserstein_2_multivariate_single_point_translation_test() {
  let assert Ok(distance) =
    transport.wasserstein_2_multivariate(
      [vector.zero()],
      [vector.pad(1.0, 0.0, 0.0)],
      0.1,
      100,
    )
  should.be_true(is_close(distance, 1.0, tight))
}

pub fn wasserstein_2_multivariate_symmetry_test() {
  let p = [
    vector.pad(0.0, 0.1, 0.2),
    vector.pad(0.4, 0.5, 0.6),
    vector.pad(0.8, 0.9, 1.0),
  ]
  let q = [
    vector.pad(0.2, 0.0, 0.1),
    vector.pad(0.6, 0.4, 0.5),
    vector.pad(1.0, 0.8, 0.9),
  ]
  let assert Ok(pq) = transport.wasserstein_2_multivariate(p, q, 0.05, 150)
  let assert Ok(qp) = transport.wasserstein_2_multivariate(q, p, 0.05, 150)
  should.be_true(is_close(pq, qp, loose))
}

pub fn wasserstein_2_multivariate_matches_pad_for_translation_test() {
  let p = [
    vector.pad(0.0, 0.0, 0.0),
    vector.pad(0.0, 1.0, 0.0),
    vector.pad(1.0, 0.0, 0.0),
    vector.pad(1.0, 1.0, 0.0),
  ]
  let q = [
    vector.pad(0.5, -0.25, 0.75),
    vector.pad(0.5, 0.75, 0.75),
    vector.pad(1.5, -0.25, 0.75),
    vector.pad(1.5, 0.75, 0.75),
  ]
  let assert Ok(multivariate) =
    transport.wasserstein_2_multivariate(p, q, 0.01, 150)
  let assert Ok(pad) = transport.wasserstein_pad(p, q)
  should.be_true(is_close_hybrid(multivariate, pad, loose, loose))
}

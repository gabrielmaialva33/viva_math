//// Tests for `viva_math/transport` — 1D Wasserstein (W₁, W₂), closed-form
//// W₂ for Gaussians, and componentwise PAD aggregation.

import gleeunit/should
import test_support.{is_close}
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

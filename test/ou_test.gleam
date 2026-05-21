//// Tests for `viva_math/ou` — exact Doob transition kernel, analytical
//// moments, autocovariance, and componentwise PAD `OUParamsVec3`.

import gleam/list
import gleeunit/should
import test_support.{is_close}
import viva_math/ou
import viva_math/random
import viva_math/vector.{Vec3}

pub fn ou_mean_at_initial_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 1.0, sigma: 0.2)
  ou.mean_at(params, 0.7, 0.0)
  |> is_close(0.7, 1.0e-12)
  |> should.be_true
}

pub fn ou_mean_at_converges_to_mu_test() {
  let params = ou.OUParams1D(theta: 1.0, mu: 2.5, sigma: 0.5)
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
  ou.half_life(params)
  |> is_close(0.693_147_180_559_945, 1.0e-9)
  |> should.be_true
}

pub fn ou_step_zero_sigma_is_deterministic_test() {
  let params = ou.OUParams1D(theta: 0.5, mu: 1.0, sigma: 0.0)
  let seed = random.from_int(42)
  let #(x_next, _) = ou.step(params, 0.0, 1.0, seed)
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
  is_close(m.x, 0.3, 1.0e-12) |> should.be_true
  is_close(m.y, -0.2, 1.0e-12) |> should.be_true
  is_close(m.z, 0.5, 1.0e-12) |> should.be_true
}

pub fn ou_is_valid_test() {
  ou.is_valid(ou.OUParams1D(theta: 1.0, mu: 0.0, sigma: 0.5))
  |> should.be_true
  ou.is_valid(ou.OUParams1D(theta: -1.0, mu: 0.0, sigma: 0.5))
  |> should.be_false
  ou.is_valid(ou.OUParams1D(theta: 1.0, mu: 0.0, sigma: -0.1))
  |> should.be_false
}

// Semigroup property of the deterministic mean kernel:
// `mean_at(t₁ + t₂) = mean_at(t₂, mean_at(t₁))`.
pub fn ou_mean_at_composes_over_time_test() {
  let params = ou.OUParams1D(theta: 0.7, mu: -0.3, sigma: 0.2)
  let x0 = 1.4
  let t1 = 0.8
  let t2 = 1.2
  let once = ou.mean_at(params, x0, t1 +. t2)
  let twice = ou.mean_at(params, ou.mean_at(params, x0, t1), t2)
  should.be_true(is_close(once, twice, 1.0e-12))
}

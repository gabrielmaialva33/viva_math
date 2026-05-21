import gleam/list
import gleeunit/should
import test_support.{is_close, is_close_hybrid, loose, tight}
import viva_math/ode
import viva_math/random
import viva_math/scalar

pub fn euler_maruyama_uses_gaussian_variance_scale_test() {
  let seed = random.from_int(42)
  let sigma = 0.7
  let dt = 0.01
  let drift = fn(_t: Float, _x: Float) { 0.0 }
  let diffusion = fn(_t: Float, _x: Float) { sigma }

  let #(z, _) = random.standard_normal(seed)
  let #(x1, _) = ode.euler_maruyama(drift, diffusion, 0.0, 0.0, dt, seed)

  // One deterministic sample validates dW = sqrt(dt) * Z, so Var[sigma*dW]
  // is sigma^2 * dt by construction without a flaky Monte Carlo tolerance.
  x1 |> is_close(sigma *. scalar.sqrt(dt) *. z, loose) |> should.be_true
}

pub fn milstein_matches_euler_maruyama_for_additive_noise_test() {
  let seed = random.from_int(123)
  let drift = fn(_t: Float, x: Float) { 0.2 *. x }
  let diffusion = fn(_t: Float, _x: Float) { 0.4 }
  let dg_dx = fn(_t: Float, _x: Float) { 0.0 }

  let #(em, _) = ode.euler_maruyama(drift, diffusion, 0.0, 1.0, 0.01, seed)
  let #(mi, _) = ode.milstein(drift, diffusion, dg_dx, 0.0, 1.0, 0.01, seed)
  mi |> is_close(em, tight) |> should.be_true
}

pub fn velocity_verlet_preserves_harmonic_energy_test() {
  let #(q, v) = step_symplectic(ode.velocity_verlet, 1000, 1.0, 0.0, 0.001)
  energy(q, v)
  |> is_close_hybrid(0.5, loose, loose)
  |> should.be_true
}

pub fn leapfrog_preserves_harmonic_energy_test() {
  let #(q, v) = step_symplectic(ode.leapfrog, 1000, 1.0, 0.0, 0.001)
  energy(q, v)
  |> is_close_hybrid(0.5, loose, loose)
  |> should.be_true
}

pub fn yoshida4_is_more_accurate_than_velocity_verlet_test() {
  let dt = 0.2
  let steps = 10
  let t = 2.0
  let #(q_vv, v_vv) = step_symplectic(ode.velocity_verlet, steps, 1.0, 0.0, dt)
  let #(q_y4, v_y4) = step_symplectic(ode.yoshida4, steps, 1.0, 0.0, dt)
  let q_exact = scalar.cos(t)
  let v_exact = 0.0 -. scalar.sin(t)

  let vv_error = scalar.hypot(q_vv -. q_exact, v_vv -. v_exact)
  let y4_error = scalar.hypot(q_y4 -. q_exact, v_y4 -. v_exact)
  should.be_true(y4_error <. vv_error)
}

pub fn integrate_sde_returns_expected_length_test() {
  let drift = fn(_t: Float, _x: Float) { 0.0 }
  let diffusion = fn(_t: Float, _x: Float) { 0.1 }
  let #(trajectory, _) =
    ode.integrate_sde(drift, diffusion, 0.0, 1.0, 0.01, 12, random.from_int(9))
  should.equal(list.length(trajectory), 13)
}

pub fn integrate_symplectic_matches_manual_steps_test() {
  let trajectory =
    ode.integrate_symplectic(
      ode.velocity_verlet,
      harmonic_force,
      0.0,
      1.0,
      0.0,
      0.1,
      2,
    )
  should.equal(list.length(trajectory), 3)

  let #(q1, v1) = ode.velocity_verlet(harmonic_force, 1.0, 0.0, 0.1)
  let #(q2, v2) = ode.velocity_verlet(harmonic_force, q1, v1, 0.1)
  let #(t_last, q_last, v_last) = last3(trajectory)

  t_last |> is_close(0.2, tight) |> should.be_true
  q_last |> is_close(q2, tight) |> should.be_true
  v_last |> is_close(v2, tight) |> should.be_true
}

fn step_symplectic(
  method: fn(ode.ForceLaw, Float, Float, Float) -> #(Float, Float),
  steps: Int,
  q: Float,
  v: Float,
  dt: Float,
) -> #(Float, Float) {
  case steps <= 0 {
    True -> #(q, v)
    False -> {
      let #(q_next, v_next) = method(harmonic_force, q, v, dt)
      step_symplectic(method, steps - 1, q_next, v_next, dt)
    }
  }
}

fn harmonic_force(q: Float) -> Float {
  0.0 -. q
}

fn energy(q: Float, v: Float) -> Float {
  0.5 *. q *. q +. 0.5 *. v *. v
}

fn last3(xs: List(#(Float, Float, Float))) -> #(Float, Float, Float) {
  case xs {
    [x] -> x
    [_, ..rest] -> last3(rest)
    [] -> #(0.0, 0.0, 0.0)
  }
}

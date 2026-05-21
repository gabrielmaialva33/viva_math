import gleeunit/should
import test_support.{is_close, is_close_hybrid, tight, transcendental}
import viva_math/constants
import viva_math/scalar

pub fn scalar_remainder_roots_and_powers_test() {
  scalar.fmod(7.5, 2.0)
  |> is_close(1.5, tight)
  |> should.be_true

  let xs = [0.0, 0.25, 2.0, 9.0, 144.0]
  list_all_close_sqrt_identity(xs)

  scalar.pow(9.0, 0.5)
  |> is_close(scalar.sqrt(9.0), transcendental)
  |> should.be_true

  scalar.pow(3.0, 2.0)
  |> is_close(9.0, tight)
  |> should.be_true
}

pub fn scalar_trig_identities_test() {
  let x = 0.37
  scalar.tan(x)
  |> is_close(scalar.sin(x) /. scalar.cos(x), transcendental)
  |> should.be_true

  let asin_x = 0.5
  scalar.asin(scalar.sin(asin_x))
  |> is_close(asin_x, transcendental)
  |> should.be_true

  let acos_x = 2.2
  scalar.acos(scalar.cos(acos_x))
  |> is_close(acos_x, transcendental)
  |> should.be_true

  scalar.atan(1.0)
  |> is_close(constants.quarter_pi, transcendental)
  |> should.be_true
}

pub fn scalar_atan2_quadrants_test() {
  scalar.atan2(1.0, 1.0)
  |> is_close(constants.quarter_pi, transcendental)
  |> should.be_true

  scalar.atan2(1.0, -1.0)
  |> is_close(3.0 *. constants.quarter_pi, transcendental)
  |> should.be_true

  scalar.atan2(-1.0, -1.0)
  |> is_close(0.0 -. 3.0 *. constants.quarter_pi, transcendental)
  |> should.be_true

  scalar.atan2(-1.0, 1.0)
  |> is_close(0.0 -. constants.quarter_pi, transcendental)
  |> should.be_true
}

pub fn scalar_log_and_root_wrappers_test() {
  scalar.log2(8.0)
  |> is_close(3.0, tight)
  |> should.be_true

  scalar.log10(1000.0)
  |> is_close(3.0, tight)
  |> should.be_true

  scalar.logarithm_2(32.0)
  |> should.equal(Ok(5.0))

  scalar.logarithm_10(100.0)
  |> should.equal(Ok(2.0))

  scalar.square_root(16.0)
  |> should.equal(Ok(4.0))

  let assert Ok(cube) = scalar.cube_root(-27.0)
  cube |> is_close(-3.0, transcendental) |> should.be_true

  let assert Ok(fourth) = scalar.nth_root(81.0, 4)
  fourth |> is_close(3.0, transcendental) |> should.be_true

  scalar.logarithm_2(0.0) |> should.equal(Error(Nil))
  scalar.logarithm_10(-1.0) |> should.equal(Error(Nil))
  scalar.square_root(-1.0) |> should.equal(Error(Nil))
  scalar.nth_root(-16.0, 4) |> should.equal(Error(Nil))
  scalar.nth_root(16.0, 0) |> should.equal(Error(Nil))
}

pub fn scalar_safe_variants_test() {
  scalar.safe_log(-2.0, -99.0)
  |> is_close(-99.0, tight)
  |> should.be_true

  scalar.safe_exp(-800.0)
  |> is_close(0.0, tight)
  |> should.be_true

  should.be_true(scalar.safe_exp(800.0) >. 1.0e300)

  scalar.safe_sqrt(-4.0)
  |> is_close(0.0, tight)
  |> should.be_true

  scalar.safe_div(5.0, 0.0, -1.0)
  |> is_close(-1.0, tight)
  |> should.be_true
}

pub fn scalar_activation_known_values_test() {
  scalar.sigmoid_k(2.0, 0.5)
  |> is_close(scalar.sigmoid(1.0), tight)
  |> should.be_true

  scalar.leaky_relu(-4.0, 0.01)
  |> is_close(-0.04, tight)
  |> should.be_true

  scalar.elu(-1.0, 2.0)
  |> is_close(2.0 *. scalar.expm1(-1.0), transcendental)
  |> should.be_true

  scalar.selu(0.0)
  |> is_close(0.0, tight)
  |> should.be_true

  scalar.swish(1.0)
  |> is_close(scalar.silu(1.0), tight)
  |> should.be_true
}

pub fn scalar_hard_activation_known_values_test() {
  scalar.hard_sigmoid(-3.0) |> is_close(0.0, tight) |> should.be_true
  scalar.hard_sigmoid(0.0) |> is_close(0.5, tight) |> should.be_true
  scalar.hard_sigmoid(3.0) |> is_close(1.0, tight) |> should.be_true

  scalar.hard_swish(3.0) |> is_close(3.0, tight) |> should.be_true
  scalar.hard_swish(-3.0) |> is_close(0.0, tight) |> should.be_true

  scalar.hard_tanh(-2.0) |> is_close(-1.0, tight) |> should.be_true
  scalar.hard_tanh(0.25) |> is_close(0.25, tight) |> should.be_true
  scalar.hard_tanh(2.0) |> is_close(1.0, tight) |> should.be_true
}

pub fn scalar_interpolation_and_step_test() {
  scalar.step(0.5, 0.49) |> is_close(0.0, tight) |> should.be_true
  scalar.step(0.5, 0.5) |> is_close(1.0, tight) |> should.be_true
  scalar.step(0.5, 0.51) |> is_close(1.0, tight) |> should.be_true

  let a = scalar.smootherstep(0.0, 1.0, 0.0)
  let b = scalar.smootherstep(0.0, 1.0, 0.25)
  let c = scalar.smootherstep(0.0, 1.0, 0.5)
  let d = scalar.smootherstep(0.0, 1.0, 0.75)
  let e = scalar.smootherstep(0.0, 1.0, 1.0)

  should.be_true(a <=. b && b <=. c && c <=. d && d <=. e)
  a |> is_close(0.0, tight) |> should.be_true
  e |> is_close(1.0, tight) |> should.be_true
}

pub fn scalar_degree_radian_roundtrip_test() {
  let xs = [
    0.0,
    constants.quarter_pi,
    constants.half_pi,
    constants.pi,
    0.0 -. constants.pi,
  ]
  list_all_close_angle_roundtrip(xs)

  scalar.deg_to_rad(180.0)
  |> is_close(constants.pi, tight)
  |> should.be_true

  scalar.rad_to_deg(constants.half_pi)
  |> is_close(90.0, tight)
  |> should.be_true
}

fn list_all_close_sqrt_identity(xs: List(Float)) {
  case xs {
    [] -> Nil
    [x, ..rest] -> {
      let r = scalar.sqrt(x)
      r *. r |> is_close_hybrid(x, tight, tight) |> should.be_true
      list_all_close_sqrt_identity(rest)
    }
  }
}

fn list_all_close_angle_roundtrip(xs: List(Float)) {
  case xs {
    [] -> Nil
    [x, ..rest] -> {
      scalar.deg_to_rad(scalar.rad_to_deg(x))
      |> is_close(x, tight)
      |> should.be_true
      list_all_close_angle_roundtrip(rest)
    }
  }
}

//// Benchmarks for ODE solvers.
////
//// Compares accuracy/cost of Euler, RK2, RK4, RKF45 and DOP54 on the
//// canonical scaling problem `dx/dt = x` with `x(0) = 1`, target `x(1) = e`.

import gleam/float
import gleam/io
import gleam/list
import viva_math/constants
import viva_math/ode

pub fn main() {
  io.println("\n=== viva_math ODE solver benchmarks ===\n")

  let f = fn(_t: Float, x: Float) { x }

  io.println("Single step accuracy (dt = 0.1, expected = e^0.1):")
  let expected_short = 1.1051709180756477
  step_report("euler           ", ode.euler(f, 0.0, 1.0, 0.1), expected_short)
  step_report(
    "rk2_midpoint    ",
    ode.rk2_midpoint(f, 0.0, 1.0, 0.1),
    expected_short,
  )
  step_report(
    "rk2_heun        ",
    ode.rk2_heun(f, 0.0, 1.0, 0.1),
    expected_short,
  )
  step_report("rk4             ", ode.rk4(f, 0.0, 1.0, 0.1), expected_short)
  let #(rkf45_x, _) = ode.rkf45(f, 0.0, 1.0, 0.1)
  step_report("rkf45           ", rkf45_x, expected_short)
  let #(dop54_x, _) = ode.dop54(f, 0.0, 1.0, 0.1)
  step_report("dop54           ", dop54_x, expected_short)

  io.println("\nFull-trajectory integration to t = 1.0, dt = 0.01:")
  let dt = 0.01
  let steps = 100
  bench_traj("euler       ", ode.euler, f, dt, steps, constants.e)
  bench_traj("rk4         ", ode.rk4, f, dt, steps, constants.e)
}

fn step_report(label: String, got: Float, expected: Float) {
  let err = float.absolute_value(got -. expected)
  io.println("  " <> label <> " err = " <> float_to_str(err))
}

fn bench_traj(
  label: String,
  method: fn(fn(Float, Float) -> Float, Float, Float, Float) -> Float,
  f: fn(Float, Float) -> Float,
  dt: Float,
  steps: Int,
  expected: Float,
) {
  let t0 = monotonic_time_ns()
  let traj = ode.integrate(method, f, 0.0, 1.0, dt, steps)
  let t1 = monotonic_time_ns()
  let assert Ok(last_pair) = list.last(traj)
  let err = float.absolute_value(last_pair.1 -. expected)
  io.println(
    "  "
    <> label
    <> " err = "
    <> float_to_str(err)
    <> "  time = "
    <> int_to_str(t1 - t0)
    <> "ns",
  )
}

@external(erlang, "erlang", "monotonic_time")
fn monotonic_time_ns() -> Int

@external(erlang, "erlang", "integer_to_binary")
fn int_to_str(n: Int) -> String

@external(erlang, "erlang", "float_to_binary")
fn float_to_str(f: Float) -> String

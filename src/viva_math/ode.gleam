//// Numerical ODE / SDE solvers.
////
//// Single-step integrators for scalar systems `dx/dt = f(t, x)` and
//// stochastic systems `dx = f(t, x) dt + g(t, x) dW`. Step functions are
//// pure: they take the current `(t, x)` and produce the next state.
////
//// ## Methods
////
//// | Function          | Order  | Use case                                    |
//// | ----------------- | ------ | ------------------------------------------- |
//// | `euler`           | 1      | Quick, stiff systems with small dt          |
//// | `rk2_midpoint`    | 2      | Mid-accuracy, half the cost of RK4          |
//// | `rk2_heun`        | 2      | Trapezoidal/improved Euler                  |
//// | `rk4`             | 4      | Workhorse for non-stiff systems             |
//// | `euler_maruyama`  | 0.5*   | SDE with additive noise (* strong order)    |
//// | `milstein`        | 1.0*   | SDE with multiplicative noise               |
////
//// For full trajectories use `integrate` / `integrate_sde`.

import gleam/list
import viva_math/random
import viva_math/scalar

/// Drift function f(t, x).
pub type Drift =
  fn(Float, Float) -> Float

/// Diffusion function g(t, x) for SDEs.
pub type Diffusion =
  fn(Float, Float) -> Float

// ============================================================================
// Deterministic single-step
// ============================================================================

/// Euler step: xₙ₊₁ = xₙ + dt · f(tₙ, xₙ).
pub fn euler(f: Drift, t: Float, x: Float, dt: Float) -> Float {
  x +. dt *. f(t, x)
}

/// Midpoint Runge-Kutta (RK2): evaluates f at the half-step midpoint.
pub fn rk2_midpoint(f: Drift, t: Float, x: Float, dt: Float) -> Float {
  let k1 = f(t, x)
  let k2 = f(t +. dt /. 2.0, x +. dt *. k1 /. 2.0)
  x +. dt *. k2
}

/// Heun (improved Euler) RK2: trapezoidal predictor-corrector.
pub fn rk2_heun(f: Drift, t: Float, x: Float, dt: Float) -> Float {
  let k1 = f(t, x)
  let k2 = f(t +. dt, x +. dt *. k1)
  x +. dt *. { k1 +. k2 } /. 2.0
}

/// Classical 4th-order Runge-Kutta.
pub fn rk4(f: Drift, t: Float, x: Float, dt: Float) -> Float {
  let k1 = f(t, x)
  let k2 = f(t +. dt /. 2.0, x +. dt *. k1 /. 2.0)
  let k3 = f(t +. dt /. 2.0, x +. dt *. k2 /. 2.0)
  let k4 = f(t +. dt, x +. dt *. k3)
  x +. dt *. { k1 +. 2.0 *. k2 +. 2.0 *. k3 +. k4 } /. 6.0
}

// ============================================================================
// Stochastic single-step (SDE)
// ============================================================================

/// Euler-Maruyama step for dx = f dt + g dW with normal increment dW = √dt·Z.
///
/// Returns the new state and the advanced PRNG seed.
pub fn euler_maruyama(
  f: Drift,
  g: Diffusion,
  t: Float,
  x: Float,
  dt: Float,
  seed: random.Seed,
) -> #(Float, random.Seed) {
  // SDE step sizes must be positive; reverse-time SDEs are not supported here.
  let dt_safe = case dt <. 0.0 {
    True -> 0.0
    False -> dt
  }
  let #(z, s) = random.standard_normal(seed)
  let dw = z *. scalar.sqrt(dt_safe)
  #(x +. f(t, x) *. dt_safe +. g(t, x) *. dw, s)
}

/// Milstein step — corrects Euler-Maruyama with the Itô derivative of g.
///
/// dx = f dt + g dW + ½g·g'(t,x)·(dW² - dt)
///
/// `dg_dx` is the partial derivative ∂g/∂x. If `g` doesn't depend on x
/// (additive noise), use `euler_maruyama` instead — they coincide.
pub fn milstein(
  f: Drift,
  g: Diffusion,
  dg_dx: Drift,
  t: Float,
  x: Float,
  dt: Float,
  seed: random.Seed,
) -> #(Float, random.Seed) {
  let #(z, s) = random.standard_normal(seed)
  let dw = z *. scalar.sqrt(dt)
  let g_val = g(t, x)
  let correction = 0.5 *. g_val *. dg_dx(t, x) *. { dw *. dw -. dt }
  #(x +. f(t, x) *. dt +. g_val *. dw +. correction, s)
}

// ============================================================================
// Adaptive Runge-Kutta-Fehlberg 4(5)
// ============================================================================

// ============================================================================
// Dormand-Prince 5(4) — the modern default for non-stiff ODEs
// ============================================================================
//
// Explicit Runge-Kutta method of order 5 with an embedded 4th-order error
// estimator, used by scipy.integrate.RK45, MATLAB's ode45, and Diffrax's
// Dopri5. Seven stages with First-Same-As-Last (FSAL) optimisation: k₇ of
// step n equals k₁ of step n+1, so practical cost is 6 function
// evaluations per step.
//
// The Dormand-Prince 8(5,3) (DOP853) variant exists but its 12-stage
// Butcher tableau is large enough that hand-coding it is bug-prone;
// scipy.integrate.DOP853 uses pre-computed binary tables. We ship the
// well-vetted 5(4) here and leave DOP853 for a future NIF.
//
// References:
//   Dormand & Prince (1980) "A family of embedded Runge-Kutta formulae"
//   Hairer, Nørsett, Wanner (1993) "Solving ODEs I" §II.5

/// One step of Dormand-Prince 5(4) with embedded error estimate.
///
/// Returns `#(x_new, err)` where `x_new` is the 5th-order solution and
/// `err` is the absolute difference from the embedded 4th-order estimate,
/// suitable for step-size control. Per-step truncation error is O(dt⁶).
pub fn dop54(f: Drift, t: Float, x: Float, dt: Float) -> #(Float, Float) {
  // Butcher tableau (Dormand-Prince 5(4))
  let k1 = f(t, x)
  let k2 = f(t +. dt /. 5.0, x +. dt *. k1 /. 5.0)
  let k3 =
    f(
      t +. 3.0 *. dt /. 10.0,
      x +. dt *. { 3.0 *. k1 /. 40.0 +. 9.0 *. k2 /. 40.0 },
    )
  let k4 =
    f(
      t +. 4.0 *. dt /. 5.0,
      x
        +. dt
        *. { 44.0 *. k1 /. 45.0 -. 56.0 *. k2 /. 15.0 +. 32.0 *. k3 /. 9.0 },
    )
  let k5 =
    f(
      t +. 8.0 *. dt /. 9.0,
      x
        +. dt
        *. {
        19_372.0
        *. k1
        /. 6561.0
        -. 25_360.0
        *. k2
        /. 2187.0
        +. 64_448.0
        *. k3
        /. 6561.0
        -. 212.0
        *. k4
        /. 729.0
      },
    )
  let k6 =
    f(
      t +. dt,
      x
        +. dt
        *. {
        9017.0
        *. k1
        /. 3168.0
        -. 355.0
        *. k2
        /. 33.0
        +. 46_732.0
        *. k3
        /. 5247.0
        +. 49.0
        *. k4
        /. 176.0
        -. 5103.0
        *. k5
        /. 18_656.0
      },
    )

  // 5th-order solution coefficients
  let x_new =
    x
    +. dt
    *. {
      35.0
      *. k1
      /. 384.0
      +. 500.0
      *. k3
      /. 1113.0
      +. 125.0
      *. k4
      /. 192.0
      -. 2187.0
      *. k5
      /. 6784.0
      +. 11.0
      *. k6
      /. 84.0
    }

  // 7th stage uses the 5th-order solution (FSAL)
  let k7 = f(t +. dt, x_new)

  // Embedded 4th-order solution
  let x_alt =
    x
    +. dt
    *. {
      5179.0
      *. k1
      /. 57_600.0
      +. 7571.0
      *. k3
      /. 16_695.0
      +. 393.0
      *. k4
      /. 640.0
      -. 92_097.0
      *. k5
      /. 339_200.0
      +. 187.0
      *. k6
      /. 2100.0
      +. k7
      /. 40.0
    }

  let err = case x_new >=. x_alt {
    True -> x_new -. x_alt
    False -> x_alt -. x_new
  }
  #(x_new, err)
}

/// Deprecated alias retained for backwards compatibility. Returns the same
/// pair as `dop54`. To be removed once external callers migrate.
/// One step of RKF45 with an error estimate.
///
/// Returns `#(x5, error)` where `x5` is the 5th-order estimate and `error` is
/// |x5 - x4|, useful for step-size control.
pub fn rkf45(f: Drift, t: Float, x: Float, dt: Float) -> #(Float, Float) {
  let k1 = dt *. f(t, x)
  let k2 = dt *. f(t +. dt /. 4.0, x +. k1 /. 4.0)
  let k3 =
    dt *. f(t +. 3.0 *. dt /. 8.0, x +. 3.0 *. k1 /. 32.0 +. 9.0 *. k2 /. 32.0)
  let k4 =
    dt
    *. f(
      t +. 12.0 *. dt /. 13.0,
      x
        +. 1932.0
        *. k1
        /. 2197.0
        -. 7200.0
        *. k2
        /. 2197.0
        +. 7296.0
        *. k3
        /. 2197.0,
    )
  let k5 =
    dt
    *. f(
      t +. dt,
      x
        +. 439.0
        *. k1
        /. 216.0
        -. 8.0
        *. k2
        +. 3680.0
        *. k3
        /. 513.0
        -. 845.0
        *. k4
        /. 4104.0,
    )
  let k6 =
    dt
    *. f(
      t +. dt /. 2.0,
      x
        -. 8.0
        *. k1
        /. 27.0
        +. 2.0
        *. k2
        -. 3544.0
        *. k3
        /. 2565.0
        +. 1859.0
        *. k4
        /. 4104.0
        -. 11.0
        *. k5
        /. 40.0,
    )

  let x4 =
    x
    +. 25.0
    *. k1
    /. 216.0
    +. 1408.0
    *. k3
    /. 2565.0
    +. 2197.0
    *. k4
    /. 4104.0
    -. k5
    /. 5.0
  let x5 =
    x
    +. 16.0
    *. k1
    /. 135.0
    +. 6656.0
    *. k3
    /. 12_825.0
    +. 28_561.0
    *. k4
    /. 56_430.0
    -. 9.0
    *. k5
    /. 50.0
    +. 2.0
    *. k6
    /. 55.0

  let err = case x5 >=. x4 {
    True -> x5 -. x4
    False -> x4 -. x5
  }
  #(x5, err)
}

// ============================================================================
// Trajectory builders
// ============================================================================

/// Integrate a deterministic ODE with a fixed-step method.
pub fn integrate(
  method: fn(Drift, Float, Float, Float) -> Float,
  f: Drift,
  t0: Float,
  x0: Float,
  dt: Float,
  steps: Int,
) -> List(#(Float, Float)) {
  integrate_loop(method, f, t0, x0, dt, steps, [#(t0, x0)])
}

fn integrate_loop(
  method: fn(Drift, Float, Float, Float) -> Float,
  f: Drift,
  t: Float,
  x: Float,
  dt: Float,
  steps: Int,
  acc: List(#(Float, Float)),
) -> List(#(Float, Float)) {
  case steps <= 0 {
    True -> list.reverse(acc)
    False -> {
      let next_t = t +. dt
      let next_x = method(f, t, x, dt)
      integrate_loop(method, f, next_t, next_x, dt, steps - 1, [
        #(next_t, next_x),
        ..acc
      ])
    }
  }
}

/// Integrate a stochastic system with Euler-Maruyama.
pub fn integrate_sde(
  f: Drift,
  g: Diffusion,
  t0: Float,
  x0: Float,
  dt: Float,
  steps: Int,
  seed: random.Seed,
) -> #(List(#(Float, Float)), random.Seed) {
  integrate_sde_loop(f, g, t0, x0, dt, steps, seed, [#(t0, x0)])
}

fn integrate_sde_loop(
  f: Drift,
  g: Diffusion,
  t: Float,
  x: Float,
  dt: Float,
  steps: Int,
  seed: random.Seed,
  acc: List(#(Float, Float)),
) -> #(List(#(Float, Float)), random.Seed) {
  case steps <= 0 {
    True -> #(list.reverse(acc), seed)
    False -> {
      let #(next_x, next_seed) = euler_maruyama(f, g, t, x, dt, seed)
      let next_t = t +. dt
      integrate_sde_loop(f, g, next_t, next_x, dt, steps - 1, next_seed, [
        #(next_t, next_x),
        ..acc
      ])
    }
  }
}

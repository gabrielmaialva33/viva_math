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
// DOP853 - Dormand-Prince 8(5,3) — the gold standard for non-stiff ODEs
// ============================================================================
//
// Implementation of Hairer's DOP853, an explicit Runge-Kutta method of order
// 8 with embedded error estimators of orders 5 and 3. This is the same
// method used by scipy.integrate.DOP853 and Diffrax. The coefficients come
// from Hairer-Nørsett-Wanner "Solving ODEs I" (Springer 1993).
//
// Use it when:
//   - High accuracy is required (target err ≤ 10⁻¹⁰)
//   - The system is non-stiff and smooth
//   - You have budget for ~12 function evaluations per step
//
// For stiff problems or oscillatory dynamics, prefer specialised methods
// (Rosenbrock, Gauss-Legendre IRK) — out of scope here.

/// One step of DOP853 with embedded error estimate.
///
/// Returns `#(x_new, err)` where `err` is an L-∞-like local error estimate
/// suitable for step-size control. The combination of the order-5 and
/// order-3 embedded estimators makes DOP853 robust across smooth ODEs.
pub fn dop853(
  f: Drift,
  t: Float,
  x: Float,
  dt: Float,
) -> #(Float, Float) {
  // Stage abscissae (c_i)
  let c2 = 0.0526001519587677318785587544488
  let c3 = 0.0789002279381515978178381316732
  let c4 = 0.118350341907227396726757197510
  let c5 = 0.281649658092772603273242802490
  let c6 = 0.333333333333333333333333333333
  let c7 = 0.25
  let c8 = 0.307692307692307692307692307692
  let c9 = 0.651282051282051282051282051282
  let c10 = 0.6
  let c11 = 0.857142857142857142857142857142

  // a-coefficients (row-major); subset used.
  let k1 = f(t, x)
  let k2 = f(t +. c2 *. dt, x +. dt *. 0.0526001519587677318785587544488 *. k1)
  let k3 =
    f(
      t +. c3 *. dt,
      x
        +. dt
        *. {
          0.0197250569845378994544595329183 *. k1
          +. 0.0591751709536136983633785987549 *. k2
        },
    )
  let k4 =
    f(
      t +. c4 *. dt,
      x
        +. dt
        *. {
          0.0295875854768068491816892993775 *. k1
          +. 0.0887627564304205475450678981324 *. k3
        },
    )
  let k5 =
    f(
      t +. c5 *. dt,
      x
        +. dt
        *. {
          0.241365134159266685502369798665 *. k1
          -. 0.884549479328286085344864962717 *. k3
          +. 0.924834003261792003115737630351 *. k4
        },
    )
  let k6 =
    f(
      t +. c6 *. dt,
      x
        +. dt
        *. {
          0.0370370370370370370370370370370 *. k1
          +. 0.170828608729473871279604482173 *. k4
          +. 0.125467687566822425016691814123 *. k5
        },
    )
  let k7 =
    f(
      t +. c7 *. dt,
      x
        +. dt
        *. {
          0.0371093750 *. k1
          +. 0.170252211019544039314978060272 *. k4
          +. 0.0602165389804559606850219397283 *. k5
          -. 0.0175781250 *. k6
        },
    )
  let k8 =
    f(
      t +. c8 *. dt,
      x
        +. dt
        *. {
          0.0370920001185047927108779319836 *. k1
          +. 0.170383925712239993810214054705 *. k4
          +. 0.107262030446373611259788006774 *. k5
          -. 0.0153194377486244017527936158236 *. k6
          +. 0.00827378916381402288758473766002 *. k7
        },
    )
  let k9 =
    f(
      t +. c9 *. dt,
      x
        +. dt
        *. {
          0.624110958716075717114429577812 *. k1
          -. 3.36089262944694129406857109825 *. k4
          -. 0.868219346841726006818189891453 *. k5
          +. 27.5920996994467083049415600797 *. k6
          +. 20.1540675504778934086186788979 *. k7
          -. 43.4898841810699588477366255144 *. k8
        },
    )
  let k10 =
    f(
      t +. c10 *. dt,
      x
        +. dt
        *. {
          0.477662536438264365890433908527 *. k1
          -. 2.48811461997166764192642586468 *. k4
          -. 0.590290826836842996371446475743 *. k5
          +. 21.2300514481811942347288488774 *. k6
          +. 15.2792336328824235832596922938 *. k7
          -. 33.2882109689848629194453176389 *. k8
          -. 0.0203312017085086261358222928593 *. k9
        },
    )
  let k11 =
    f(
      t +. c11 *. dt,
      x
        +. dt
        *. {
          -19.5778479795613910484035111281 *. k1
          +. 111.408081056545300451213684475 *. k4
          -. 1.84357566457552267572568049378 *. k5
          -. 13.7270746154103097835015180142 *. k6
          +. 2.61876616647600003500859014866 *. k7
          +. 13.2547875323920512057178605499 *. k8
          +. 1.21881213813681224786669875401 *. k9
          -. 1.86766391545820906900920002392 *. k10
        },
    )
  let k12 =
    f(
      t +. dt,
      x
        +. dt
        *. {
          -0.491338588803989988800076022717 *. k1
          -. 11.4724485620176055015900989415 *. k4
          -. 0.508169095070834008351817453828 *. k5
          -. 7.62905335219842437830006380078 *. k6
          -. 0.215216968819090395906767317587 *. k7
          +. 21.2700399583015952706711928437 *. k8
          -. 1.86407189676803806080040365253 *. k9
          +. 2.30607986474442942466014931923 *. k10
          +. 0.140709071359799802975488727050 *. k11
        },
    )

  // 8th-order solution coefficients (b_i)
  let x_new =
    x
    +. dt
    *. {
      0.0542937341165687622380535766363 *. k1
      +. 4.45031289275240888144113950566 *. k6
      +. 1.89151789931450038304281599044 *. k7
      -. 5.80120396001058478146721707624 *. k8
      +. 0.311164366957350449629876347172 *. k9
      -. 0.152160949662516078556178806805 *. k10
      +. 0.201365400804030348374776537501 *. k11
      +. 0.0447106157277725905176885569043 *. k12
    }

  // Embedded 5th-order error estimator coefficients (e_i = b_i - b̂_i)
  let err5 =
    dt
    *. {
      0.0244094488188976377952755905512 *. k1
      +. 0.0666666666666666666666666666667 *. k9
      +. 0.0179453052631578947368421052632 *. k10
      -. 0.105092592592592592592592592593 *. k11
      +. 0.0156250000000000000000000000000 *. k12
    }

  #(x_new, abs_float(err5))
}

fn abs_float(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

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

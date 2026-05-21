//// Ornstein-Uhlenbeck mood dynamics.
////
//// Mean-reverting stochastic process for affective dynamics — the canonical
//// model for emotion regulation toward a baseline. Underlies VIVA's
//// homeostatic emotional decay.
////
//// **SDE**: `dX_t = θ(μ - X_t) dt + σ dW_t`
////
//// Parameters:
//// - `theta` (θ) — mean-reversion speed (> 0). Larger = faster return to μ.
//// - `mu`    (μ) — long-run mean (the attractor).
//// - `sigma` (σ) — diffusion (volatility, ≥ 0).
////
//// ## Analytical properties
////
//// Given `X_0 = x_0`:
////   - `E[X_t]   = μ + (x_0 − μ) · e^(−θt)`
////   - `Var[X_t] = σ² / (2θ) · (1 − e^(−2θt))`
////   - Stationary: `X_∞ ~ N(μ, σ²/(2θ))`
////   - Autocovariance at lag `τ`: `σ²/(2θ) · e^(−θ|τ|)`
////   - Half-life of expectation: `ln(2) / θ`
////
//// ## Integration
////
//// `step` uses the **exact transition kernel** (Doob 1942) — no
//// discretization error regardless of `dt`. Closed form:
////
//// ```
//// X_{t+Δ} = μ + (X_t − μ)·e^(−θΔ) + σ·sqrt((1 − e^(−2θΔ))/(2θ)) · Z
//// ```
////
//// where `Z ~ N(0, 1)`. For Euler-Maruyama on the same SDE, use
//// `ode.euler_maruyama` directly with a custom drift/diffusion.
////
//// ## References
////
//// - Uhlenbeck & Ornstein (1930) — *On the theory of Brownian motion*
//// - Oravecz, Tuerlinckx & Vandekerckhove (2009) — *Ornstein-Uhlenbeck
////   Process in Affective Dynamics*
//// - Doob (1942) — *The Brownian Movement and Stochastic Equations*

import gleam/float
import gleam/list
import viva_math/random.{type Seed}
import viva_math/scalar
import viva_math/vector.{type Vec3}

// ============================================================================
// Types
// ============================================================================

/// Scalar Ornstein-Uhlenbeck parameters.
pub type OUParams1D {
  OUParams1D(theta: Float, mu: Float, sigma: Float)
}

/// Componentwise Ornstein-Uhlenbeck parameters over PAD.
///
/// Each axis (pleasure, arousal, dominance) gets its own θ, μ, σ — no
/// cross-correlation between axes (diagonal covariance). Sufficient for
/// affective dynamics where each dimension regulates independently.
pub type OUParamsVec3 {
  OUParamsVec3(theta: Vec3, mu: Vec3, sigma: Vec3)
}

// ============================================================================
// Validation
// ============================================================================

/// Check whether 1D parameters are physically meaningful.
///
/// Requires `theta > 0` (otherwise no mean-reversion) and `sigma >= 0`.
pub fn is_valid(params: OUParams1D) -> Bool {
  params.theta >. 0.0 && params.sigma >=. 0.0
}

/// Vec3 validity — every component of `theta` strictly positive and every
/// component of `sigma` non-negative.
pub fn is_valid_vec3(params: OUParamsVec3) -> Bool {
  params.theta.x >. 0.0
  && params.theta.y >. 0.0
  && params.theta.z >. 0.0
  && params.sigma.x >=. 0.0
  && params.sigma.y >=. 0.0
  && params.sigma.z >=. 0.0
}

// ============================================================================
// 1D — exact transition step (Doob 1942)
// ============================================================================

/// One step of the exact OU transition kernel.
///
/// `X_{t+dt} = μ + (X_t − μ)·e^(−θ·dt) + σ·sqrt((1 − e^(−2θ·dt))/(2θ)) · Z`
///
/// `Z ~ N(0, 1)`. No discretization error: works correctly even for large `dt`.
///
/// **Caller-validated inputs**: callers must ensure `theta > 0`, `sigma >= 0`,
/// and `dt >= 0` (use `is_valid` for the params). With `dt < 0` the variance
/// term goes negative and `std_term` silently collapses to `0.0`, producing
/// a deterministic backward step that consumes a normal draw without using
/// it — not a physically meaningful transition.
pub fn step(
  params: OUParams1D,
  x: Float,
  dt: Float,
  seed: Seed,
) -> #(Float, Seed) {
  let OUParams1D(theta, mu, sigma) = params
  let decay = scalar.exp(0.0 -. theta *. dt)
  let drift_term = mu +. { x -. mu } *. decay
  // `1 − e^(−2θdt) = −expm1(−2θdt)` — avoids catastrophic cancellation when
  // `θ·dt → 0` (reduces correctly to the Brownian limit `σ²·dt`).
  let var_term =
    sigma
    *. sigma
    *. { 0.0 -. scalar.expm1(0.0 -. 2.0 *. theta *. dt) }
    /. { 2.0 *. theta }
  let std_term = case float.square_root(var_term) {
    Ok(s) -> s
    Error(_) -> 0.0
  }
  let #(z, new_seed) = random.standard_normal(seed)
  #(drift_term +. std_term *. z, new_seed)
}

/// Simulate `n` steps starting from `x0` with constant time-step `dt`.
///
/// Returns the trajectory **excluding** the initial point (length `n`) and the
/// final seed for chaining.
///
/// Pre-computes the transition kernel (`decay`, `std`) once — the loop only
/// does a multiply-add and a normal draw per step.
pub fn simulate(
  params: OUParams1D,
  x0: Float,
  dt: Float,
  n: Int,
  seed: Seed,
) -> #(List(Float), Seed) {
  let OUParams1D(theta, mu, sigma) = params
  let decay = scalar.exp(0.0 -. theta *. dt)
  let var_term =
    sigma
    *. sigma
    *. { 0.0 -. scalar.expm1(0.0 -. 2.0 *. theta *. dt) }
    /. { 2.0 *. theta }
  let std_term = case float.square_root(var_term) {
    Ok(s) -> s
    Error(_) -> 0.0
  }
  let #(traj, s) = simulate_loop(decay, mu, std_term, x0, n, seed, [])
  #(list.reverse(traj), s)
}

fn simulate_loop(
  decay: Float,
  mu: Float,
  std: Float,
  x: Float,
  n: Int,
  seed: Seed,
  acc: List(Float),
) -> #(List(Float), Seed) {
  case n <= 0 {
    True -> #(acc, seed)
    False -> {
      let #(z, s_next) = random.standard_normal(seed)
      let x_next = mu +. { x -. mu } *. decay +. std *. z
      simulate_loop(decay, mu, std, x_next, n - 1, s_next, [x_next, ..acc])
    }
  }
}

// ============================================================================
// 1D — analytical moments
// ============================================================================

/// Closed-form `E[X_t | X_0 = x0]`.
///
/// `μ + (x0 − μ) · e^(−θ·t)`
pub fn mean_at(params: OUParams1D, x0: Float, t: Float) -> Float {
  let OUParams1D(theta, mu, _) = params
  let decay = scalar.exp(0.0 -. theta *. t)
  mu +. { x0 -. mu } *. decay
}

/// Closed-form `Var[X_t | X_0 = x0]`.
///
/// `σ² / (2θ) · (1 − e^(−2θ·t))`
///
/// **Note**: the conditional variance is **independent of `x0`** — OU's noise
/// is additive Brownian, so all `x0`-dependence is absorbed into the mean.
/// The parameter is kept in the signature only to mirror `mean_at` and
/// `variance_at_vec3` for API symmetry; pass any value.
///
/// Routed through `scalar.expm1` so the Brownian limit `σ²·t` (as `θ·t → 0`)
/// is recovered without catastrophic cancellation.
pub fn variance_at(params: OUParams1D, _x0: Float, t: Float) -> Float {
  let OUParams1D(theta, _, sigma) = params
  // `1 − e^(−2θt) = −expm1(−2θt)` — avoids cancellation as `θ·t → 0`.
  sigma
  *. sigma
  *. { 0.0 -. scalar.expm1(0.0 -. 2.0 *. theta *. t) }
  /. { 2.0 *. theta }
}

/// Stationary variance `σ² / (2θ)` — the variance of `X_∞ ~ N(μ, σ²/(2θ))`.
pub fn stationary_variance(params: OUParams1D) -> Float {
  let OUParams1D(theta, _, sigma) = params
  sigma *. sigma /. { 2.0 *. theta }
}

/// Stationary standard deviation.
pub fn stationary_std(params: OUParams1D) -> Float {
  case float.square_root(stationary_variance(params)) {
    Ok(s) -> s
    Error(_) -> 0.0
  }
}

/// Autocovariance at lag `τ` (in time units).
///
/// `Cov(X_s, X_{s+τ}) = σ²/(2θ) · e^(−θ·|τ|)` (stationary regime).
pub fn autocovariance(params: OUParams1D, lag: Float) -> Float {
  let abs_lag = float.absolute_value(lag)
  stationary_variance(params) *. scalar.exp(0.0 -. params.theta *. abs_lag)
}

/// Half-life of mean reversion: `ln(2) / θ`.
///
/// Time at which `E[X_t]` has covered half the gap toward `μ`.
pub fn half_life(params: OUParams1D) -> Float {
  case scalar.logarithm(2.0) {
    Ok(l) -> l /. params.theta
    Error(_) -> 0.0
  }
}

// ============================================================================
// Vec3 — componentwise (PAD)
// ============================================================================

/// One Vec3 OU step. Each axis (P, A, D) updated independently via the exact
/// 1D kernel. Three normals drawn from the seed in sequence.
pub fn step_vec3(
  params: OUParamsVec3,
  x: Vec3,
  dt: Float,
  seed: Seed,
) -> #(Vec3, Seed) {
  let OUParamsVec3(th, mu, sg) = params
  let #(px, s1) = step(OUParams1D(th.x, mu.x, sg.x), x.x, dt, seed)
  let #(py, s2) = step(OUParams1D(th.y, mu.y, sg.y), x.y, dt, s1)
  let #(pz, s3) = step(OUParams1D(th.z, mu.z, sg.z), x.z, dt, s2)
  #(vector.Vec3(px, py, pz), s3)
}

/// Simulate Vec3 trajectory. Returns `n` Vec3 points excluding initial.
///
/// Pre-computes the three componentwise transition kernels once — the loop
/// only does multiply-adds and three normal draws per step.
pub fn simulate_vec3(
  params: OUParamsVec3,
  x0: Vec3,
  dt: Float,
  n: Int,
  seed: Seed,
) -> #(List(Vec3), Seed) {
  let OUParamsVec3(th, mu, sg) = params
  let kx = build_kernel(th.x, sg.x, dt)
  let ky = build_kernel(th.y, sg.y, dt)
  let kz = build_kernel(th.z, sg.z, dt)
  let #(traj, s) = simulate_vec3_loop(kx, ky, kz, mu, x0, n, seed, [])
  #(list.reverse(traj), s)
}

/// Pre-computed scalar OU kernel `(decay, std)` for one axis.
type Kernel {
  Kernel(decay: Float, std: Float)
}

fn build_kernel(theta: Float, sigma: Float, dt: Float) -> Kernel {
  let decay = scalar.exp(0.0 -. theta *. dt)
  let var_term =
    sigma
    *. sigma
    *. { 0.0 -. scalar.expm1(0.0 -. 2.0 *. theta *. dt) }
    /. { 2.0 *. theta }
  let std = case float.square_root(var_term) {
    Ok(s) -> s
    Error(_) -> 0.0
  }
  Kernel(decay: decay, std: std)
}

fn simulate_vec3_loop(
  kx: Kernel,
  ky: Kernel,
  kz: Kernel,
  mu: Vec3,
  x: Vec3,
  n: Int,
  seed: Seed,
  acc: List(Vec3),
) -> #(List(Vec3), Seed) {
  case n <= 0 {
    True -> #(acc, seed)
    False -> {
      let #(zx, s1) = random.standard_normal(seed)
      let #(zy, s2) = random.standard_normal(s1)
      let #(zz, s3) = random.standard_normal(s2)
      let nx = mu.x +. { x.x -. mu.x } *. kx.decay +. kx.std *. zx
      let ny = mu.y +. { x.y -. mu.y } *. ky.decay +. ky.std *. zy
      let nz = mu.z +. { x.z -. mu.z } *. kz.decay +. kz.std *. zz
      let x_next = vector.Vec3(nx, ny, nz)
      simulate_vec3_loop(kx, ky, kz, mu, x_next, n - 1, s3, [x_next, ..acc])
    }
  }
}

/// Closed-form `E[X_t | X_0 = x0]` componentwise.
pub fn mean_at_vec3(params: OUParamsVec3, x0: Vec3, t: Float) -> Vec3 {
  vector.Vec3(
    mean_at(OUParams1D(params.theta.x, params.mu.x, params.sigma.x), x0.x, t),
    mean_at(OUParams1D(params.theta.y, params.mu.y, params.sigma.y), x0.y, t),
    mean_at(OUParams1D(params.theta.z, params.mu.z, params.sigma.z), x0.z, t),
  )
}

/// Closed-form `Var[X_t | X_0 = x0]` componentwise.
pub fn variance_at_vec3(params: OUParamsVec3, x0: Vec3, t: Float) -> Vec3 {
  vector.Vec3(
    variance_at(
      OUParams1D(params.theta.x, params.mu.x, params.sigma.x),
      x0.x,
      t,
    ),
    variance_at(
      OUParams1D(params.theta.y, params.mu.y, params.sigma.y),
      x0.y,
      t,
    ),
    variance_at(
      OUParams1D(params.theta.z, params.mu.z, params.sigma.z),
      x0.z,
      t,
    ),
  )
}

/// Stationary variance per axis.
pub fn stationary_variance_vec3(params: OUParamsVec3) -> Vec3 {
  vector.Vec3(
    stationary_variance(OUParams1D(params.theta.x, params.mu.x, params.sigma.x)),
    stationary_variance(OUParams1D(params.theta.y, params.mu.y, params.sigma.y)),
    stationary_variance(OUParams1D(params.theta.z, params.mu.z, params.sigma.z)),
  )
}

//// Probability distributions.
////
//// Closed-form PDF / CDF / log-PDF + samplers backed by `viva_math/random`.
//// All samplers are pure: they take a `Seed` and return a new seed.
////
//// ## Scope
////
//// Continuous: `gaussian`, `uniform`, `exponential`, `cauchy`, `laplace`.
//// Discrete: `bernoulli`, `categorical`.
////
//// For heavier statistics (gamma, beta, chi-square) prefer `gleam_stats`
//// or `gleam_community_maths`. Here we ship only what `viva_emotion` and
//// `viva_tensor` consume directly.

import gleam/list
import viva_math/constants
import viva_math/random
import viva_math/scalar

// ============================================================================
// Gaussian (univariate)
// ============================================================================

/// Gaussian parameters.
pub type Gaussian {
  Gaussian(mean: Float, stddev: Float)
}

/// Standard normal N(0, 1).
pub fn standard_normal() -> Gaussian {
  Gaussian(mean: 0.0, stddev: 1.0)
}

/// Probability density f(x; μ, σ) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²)).
pub fn gaussian_pdf(g: Gaussian, x: Float) -> Float {
  let z = { x -. g.mean } /. g.stddev
  let coeff = constants.inv_sqrt_2pi /. g.stddev
  coeff *. scalar.exp(-0.5 *. z *. z)
}

/// Log-density. More numerically stable than `ln(gaussian_pdf)`.
pub fn gaussian_log_pdf(g: Gaussian, x: Float) -> Float {
  let z = { x -. g.mean } /. g.stddev
  -0.5 *. z *. z -. scalar.ln(g.stddev) -. 0.5 *. scalar.ln(constants.tau)
}

/// Cumulative distribution F(x; μ, σ) = ½ · (1 + erf((x - μ)/(σ√2))).
pub fn gaussian_cdf(g: Gaussian, x: Float) -> Float {
  0.5
  *. { 1.0 +. scalar.erf({ x -. g.mean } /. { g.stddev *. constants.sqrt_2 }) }
}

/// Sample from a Gaussian.
pub fn gaussian_sample(
  g: Gaussian,
  seed: random.Seed,
) -> #(Float, random.Seed) {
  random.normal(seed, g.mean, g.stddev)
}

// ============================================================================
// Uniform (continuous)
// ============================================================================

pub type Uniform {
  Uniform(low: Float, high: Float)
}

/// PDF on [low, high]; zero elsewhere.
pub fn uniform_pdf(u: Uniform, x: Float) -> Float {
  case x <. u.low || x >. u.high {
    True -> 0.0
    False -> 1.0 /. { u.high -. u.low }
  }
}

/// CDF.
pub fn uniform_cdf(u: Uniform, x: Float) -> Float {
  case x <. u.low, x >. u.high {
    True, _ -> 0.0
    _, True -> 1.0
    _, _ -> { x -. u.low } /. { u.high -. u.low }
  }
}

pub fn uniform_sample(u: Uniform, seed: random.Seed) -> #(Float, random.Seed) {
  random.uniform_in(seed, u.low, u.high)
}

// ============================================================================
// Exponential
// ============================================================================

pub type Exponential {
  Exponential(rate: Float)
}

/// PDF f(x; λ) = λ · e^(-λx) for x ≥ 0.
pub fn exponential_pdf(e: Exponential, x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0
    False -> e.rate *. scalar.exp(0.0 -. e.rate *. x)
  }
}

/// CDF.
pub fn exponential_cdf(e: Exponential, x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0
    False -> 1.0 -. scalar.exp(0.0 -. e.rate *. x)
  }
}

/// Inverse-CDF sampling.
///
/// Uses x = -log1p(-U) / λ, which is numerically equivalent to
/// -ln(1 - U) / λ but avoids cancellation when U is near zero. Since
/// `random.uniform` returns U ∈ [0, 1), 1 - U ∈ (0, 1] never hits zero.
pub fn exponential_sample(
  e: Exponential,
  seed: random.Seed,
) -> #(Float, random.Seed) {
  let #(u, s) = random.uniform(seed)
  #(0.0 -. scalar.log1p(0.0 -. u) /. e.rate, s)
}

// ============================================================================
// Laplace
// ============================================================================

pub type Laplace {
  Laplace(location: Float, scale: Float)
}

pub fn laplace_pdf(l: Laplace, x: Float) -> Float {
  let abs_dev = case x >=. l.location {
    True -> x -. l.location
    False -> l.location -. x
  }
  scalar.exp(0.0 -. abs_dev /. l.scale) /. { 2.0 *. l.scale }
}

pub fn laplace_sample(l: Laplace, seed: random.Seed) -> #(Float, random.Seed) {
  let #(u, s) = random.uniform(seed)
  // u in [0, 1); shift to [-0.5, 0.5).
  let shifted = u -. 0.5
  let abs_shift = case shifted >=. 0.0 {
    True -> shifted
    False -> 0.0 -. shifted
  }
  let sign_shift = scalar.sign(shifted)
  // log1p(-2|shifted|) is finite for |shifted| < 0.5 (always true since
  // u ∈ [0, 1)). Guard against the boundary just in case.
  let inner = 0.0 -. 2.0 *. abs_shift
  let log_factor = case inner <=. -1.0 {
    True -> 0.0 -. 1.0e300
    False -> scalar.log1p(inner)
  }
  #(l.location -. l.scale *. sign_shift *. log_factor, s)
}

// ============================================================================
// Cauchy (Lorentzian)
// ============================================================================

pub type Cauchy {
  Cauchy(location: Float, scale: Float)
}

pub fn cauchy_pdf(c: Cauchy, x: Float) -> Float {
  let z = { x -. c.location } /. c.scale
  1.0 /. { constants.pi *. c.scale *. { 1.0 +. z *. z } }
}

pub fn cauchy_sample(c: Cauchy, seed: random.Seed) -> #(Float, random.Seed) {
  let #(u, s) = random.uniform(seed)
  // Inverse CDF: location + scale · tan(π(u - ½))
  let arg = constants.pi *. { u -. 0.5 }
  #(c.location +. c.scale *. tangent(arg), s)
}

// ============================================================================
// Bernoulli (discrete)
// ============================================================================

pub type Bernoulli {
  Bernoulli(p: Float)
}

pub fn bernoulli_pmf(b: Bernoulli, k: Int) -> Float {
  case k {
    1 -> b.p
    0 -> 1.0 -. b.p
    _ -> 0.0
  }
}

pub fn bernoulli_sample(
  b: Bernoulli,
  seed: random.Seed,
) -> #(Bool, random.Seed) {
  random.bernoulli(seed, b.p)
}

// ============================================================================
// Categorical
// ============================================================================

pub type Categorical {
  Categorical(probs: List(Float))
}

/// PMF for a category index (zero-based).
pub fn categorical_pmf(c: Categorical, k: Int) -> Float {
  case list_at(c.probs, k) {
    Ok(p) -> p
    Error(_) -> 0.0
  }
}

/// Entropy H(X) = -Σ pᵢ log pᵢ.
pub fn categorical_entropy(c: Categorical) -> Float {
  list.fold(c.probs, 0.0, fn(acc, p) {
    case p <=. 0.0 {
      True -> acc
      False -> acc -. p *. scalar.ln(p)
    }
  })
}

pub fn categorical_sample(
  c: Categorical,
  seed: random.Seed,
) -> Result(#(Int, random.Seed), Nil) {
  random.categorical(seed, c.probs)
}

// ============================================================================
// Helpers
// ============================================================================

fn list_at(xs: List(Float), idx: Int) -> Result(Float, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at(rest, n - 1)
  }
}

@external(erlang, "math", "tan")
@external(javascript, "../viva_math_random_ffi.mjs", "tan")
fn tangent(x: Float) -> Float

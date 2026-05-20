//// Descriptive statistics over `List(Float)`.
////
//// All functions are pure and operate on lists. For large datasets prefer
//// `viva_tensor` and stream reductions; this module targets emotion-scale
//// data (10²–10⁴ samples).
////
//// ## Numerical notes
////
//// - Variance uses Welford's online algorithm (`welford`) to avoid
////   catastrophic cancellation. `variance/1` materialises the result.
//// - Population (N) and sample (N-1) variants are provided where the
////   distinction matters.
//// - Percentile uses linear interpolation between adjacent order statistics
////   (NumPy-style "linear" method).

import gleam/float
import gleam/list
import gleam/order
import viva_math/precision
import viva_math/scalar

// ============================================================================
// Sums & means
// ============================================================================

/// Sum of a list using Neumaier compensated summation. Returns 0.0 for empty.
///
/// Recovers up to 16 extra bits of precision vs naive `fold (+.)`.
pub fn sum(xs: List(Float)) -> Float {
  precision.neumaier_sum(xs)
}

/// Arithmetic mean using Neumaier-compensated sum.
pub fn mean(xs: List(Float)) -> Result(Float, Nil) {
  precision.neumaier_mean(xs)
}

/// Geometric mean: ⁿ√(∏ xᵢ). Requires strictly positive inputs.
pub fn geometric_mean(xs: List(Float)) -> Result(Float, Nil) {
  case xs {
    [] -> Error(Nil)
    _ -> {
      let n = int_to_float(list.length(xs))
      let all_positive = list.all(xs, fn(x) { x >. 0.0 })
      case all_positive {
        False -> Error(Nil)
        True -> {
          let log_sum = list.fold(xs, 0.0, fn(acc, x) { acc +. scalar.ln(x) })
          Ok(scalar.exp(log_sum /. n))
        }
      }
    }
  }
}

/// Harmonic mean: n / Σ(1/xᵢ). Requires non-zero inputs.
pub fn harmonic_mean(xs: List(Float)) -> Result(Float, Nil) {
  case xs {
    [] -> Error(Nil)
    _ -> {
      let n = int_to_float(list.length(xs))
      let inv_sum =
        list.fold(xs, Ok(0.0), fn(acc, x) {
          case acc, x == 0.0 {
            Error(_), _ -> Error(Nil)
            _, True -> Error(Nil)
            Ok(s), False -> Ok(s +. 1.0 /. x)
          }
        })
      case inv_sum {
        Ok(s) ->
          case s == 0.0 {
            True -> Error(Nil)
            False -> Ok(n /. s)
          }
        Error(_) -> Error(Nil)
      }
    }
  }
}

/// Weighted mean: Σ(wᵢ · xᵢ) / Σwᵢ.
pub fn weighted_mean(xs: List(Float), ws: List(Float)) -> Result(Float, Nil) {
  case list.length(xs) == list.length(ws), xs {
    False, _ -> Error(Nil)
    _, [] -> Error(Nil)
    True, _ -> {
      let pairs = list.zip(xs, ws)
      let total_w = list.fold(pairs, 0.0, fn(acc, p) { acc +. p.1 })
      case total_w == 0.0 {
        True -> Error(Nil)
        False -> {
          let num = list.fold(pairs, 0.0, fn(acc, p) { acc +. p.0 *. p.1 })
          Ok(num /. total_w)
        }
      }
    }
  }
}

// ============================================================================
// Variance & spread (Welford's online algorithm)
// ============================================================================

/// Online (mean, variance, count) accumulator from Welford's algorithm.
pub type Welford {
  Welford(count: Int, mean: Float, m2: Float)
}

/// Empty Welford accumulator.
pub fn welford_empty() -> Welford {
  Welford(count: 0, mean: 0.0, m2: 0.0)
}

/// Update a Welford accumulator with a new sample.
pub fn welford_update(w: Welford, x: Float) -> Welford {
  let count = w.count + 1
  let delta = x -. w.mean
  let new_mean = w.mean +. delta /. int_to_float(count)
  let delta2 = x -. new_mean
  Welford(count: count, mean: new_mean, m2: w.m2 +. delta *. delta2)
}

/// Build a Welford accumulator from a list.
pub fn welford(xs: List(Float)) -> Welford {
  list.fold(xs, welford_empty(), welford_update)
}

/// Population variance σ² = M₂ / N.
pub fn variance(xs: List(Float)) -> Result(Float, Nil) {
  let w = welford(xs)
  case w.count {
    0 -> Error(Nil)
    n -> Ok(w.m2 /. int_to_float(n))
  }
}

/// Sample variance s² = M₂ / (N - 1).
pub fn sample_variance(xs: List(Float)) -> Result(Float, Nil) {
  let w = welford(xs)
  case w.count {
    0 | 1 -> Error(Nil)
    n -> Ok(w.m2 /. int_to_float(n - 1))
  }
}

/// Population standard deviation.
pub fn stddev(xs: List(Float)) -> Result(Float, Nil) {
  case variance(xs) {
    Ok(v) -> Ok(scalar.sqrt(v))
    Error(_) -> Error(Nil)
  }
}

/// Sample standard deviation.
pub fn sample_stddev(xs: List(Float)) -> Result(Float, Nil) {
  case sample_variance(xs) {
    Ok(v) -> Ok(scalar.sqrt(v))
    Error(_) -> Error(Nil)
  }
}

// ============================================================================
// Covariance & correlation
// ============================================================================

/// Population covariance Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)].
pub fn covariance(xs: List(Float), ys: List(Float)) -> Result(Float, Nil) {
  case list.length(xs) == list.length(ys), xs, ys {
    False, _, _ -> Error(Nil)
    _, [], _ -> Error(Nil)
    _, _, [] -> Error(Nil)
    True, _, _ -> {
      let n = int_to_float(list.length(xs))
      case mean(xs), mean(ys) {
        Ok(mx), Ok(my) -> {
          let s =
            list.fold(list.zip(xs, ys), 0.0, fn(acc, p) {
              acc +. { p.0 -. mx } *. { p.1 -. my }
            })
          Ok(s /. n)
        }
        _, _ -> Error(Nil)
      }
    }
  }
}

/// Pearson correlation coefficient. Result in [-1, 1].
pub fn pearson(xs: List(Float), ys: List(Float)) -> Result(Float, Nil) {
  case covariance(xs, ys), stddev(xs), stddev(ys) {
    Ok(c), Ok(sx), Ok(sy) -> {
      let denom = sx *. sy
      case denom == 0.0 {
        True -> Error(Nil)
        False -> Ok(c /. denom)
      }
    }
    _, _, _ -> Error(Nil)
  }
}

// ============================================================================
// Order statistics
// ============================================================================

/// Minimum of a list.
pub fn min(xs: List(Float)) -> Result(Float, Nil) {
  case xs {
    [] -> Error(Nil)
    [x, ..rest] -> Ok(list.fold(rest, x, float.min))
  }
}

/// Maximum of a list.
pub fn max(xs: List(Float)) -> Result(Float, Nil) {
  case xs {
    [] -> Error(Nil)
    [x, ..rest] -> Ok(list.fold(rest, x, float.max))
  }
}

/// Range = max - min.
pub fn range(xs: List(Float)) -> Result(Float, Nil) {
  case min(xs), max(xs) {
    Ok(lo), Ok(hi) -> Ok(hi -. lo)
    _, _ -> Error(Nil)
  }
}

/// Median (50th percentile). Linearly interpolates for even-length lists.
pub fn median(xs: List(Float)) -> Result(Float, Nil) {
  percentile(xs, 0.5)
}

/// Percentile q ∈ [0, 1] using linear interpolation (NumPy default).
pub fn percentile(xs: List(Float), q: Float) -> Result(Float, Nil) {
  case xs, q <. 0.0 || q >. 1.0 {
    [], _ -> Error(Nil)
    _, True -> Error(Nil)
    _, False -> {
      let sorted = list.sort(xs, sort_floats)
      let n = list.length(sorted)
      let pos = q *. int_to_float(n - 1)
      let lo = float_floor(pos)
      let hi = float_ceil(pos)
      let frac = pos -. lo
      case
        list_at(sorted, float_to_int(lo)),
        list_at(sorted, float_to_int(hi))
      {
        Ok(a), Ok(b) -> Ok(a +. frac *. { b -. a })
        _, _ -> Error(Nil)
      }
    }
  }
}

/// Quartiles Q1, Q2 (median), Q3 as a tuple.
pub fn quartiles(xs: List(Float)) -> Result(#(Float, Float, Float), Nil) {
  case percentile(xs, 0.25), percentile(xs, 0.5), percentile(xs, 0.75) {
    Ok(q1), Ok(q2), Ok(q3) -> Ok(#(q1, q2, q3))
    _, _, _ -> Error(Nil)
  }
}

/// Interquartile range Q3 - Q1.
pub fn iqr(xs: List(Float)) -> Result(Float, Nil) {
  case percentile(xs, 0.25), percentile(xs, 0.75) {
    Ok(q1), Ok(q3) -> Ok(q3 -. q1)
    _, _ -> Error(Nil)
  }
}

// ============================================================================
// Normalisation & standardisation
// ============================================================================

/// Z-score: (x - μ) / σ. Returns an error if the input has zero variance.
pub fn z_score(xs: List(Float)) -> Result(List(Float), Nil) {
  case mean(xs), stddev(xs) {
    Ok(mu), Ok(sigma) ->
      case sigma == 0.0 {
        True -> Error(Nil)
        False -> Ok(list.map(xs, fn(x) { { x -. mu } /. sigma }))
      }
    _, _ -> Error(Nil)
  }
}

/// Min-max normalisation to [0, 1].
pub fn min_max_normalize(xs: List(Float)) -> Result(List(Float), Nil) {
  case min(xs), max(xs) {
    Ok(lo), Ok(hi) -> {
      let span = hi -. lo
      case span == 0.0 {
        True -> Error(Nil)
        False -> Ok(list.map(xs, fn(x) { { x -. lo } /. span }))
      }
    }
    _, _ -> Error(Nil)
  }
}

// ============================================================================
// Moving averages
// ============================================================================

/// Simple moving average over a sliding window of size `window`.
///
/// Output has length `len(xs) - window + 1`. Errors if window > len or < 1.
pub fn moving_average(
  xs: List(Float),
  window: Int,
) -> Result(List(Float), Nil) {
  let n = list.length(xs)
  case window < 1 || window > n {
    True -> Error(Nil)
    False -> {
      // Rolling sum in O(n): seed with sum of first `window`, then for each
      // step subtract the leaving element and add the entering one.
      let head = list.take(xs, window)
      let body = list.drop(xs, window)
      let initial_sum = sum(head)
      let window_float = int_to_float(window)
      Ok(
        roll_loop(head, body, initial_sum, window_float, [
          initial_sum /. window_float,
        ]),
      )
    }
  }
}

fn roll_loop(
  leaving: List(Float),
  entering: List(Float),
  running_sum: Float,
  window_float: Float,
  acc: List(Float),
) -> List(Float) {
  case leaving, entering {
    [out, ..lrest], [in_v, ..erest] -> {
      let new_sum = running_sum -. out +. in_v
      roll_loop(lrest, erest, new_sum, window_float, [
        new_sum /. window_float,
        ..acc
      ])
    }
    _, _ -> list.reverse(acc)
  }
}

/// Exponential moving average with smoothing factor α ∈ (0, 1].
///
/// Recurrence: yₜ = α·xₜ + (1-α)·yₜ₋₁, y₀ = x₀.
pub fn ema(xs: List(Float), alpha: Float) -> Result(List(Float), Nil) {
  case xs, alpha <=. 0.0 || alpha >. 1.0 {
    [], _ -> Error(Nil)
    _, True -> Error(Nil)
    [x0, ..rest], False -> Ok(ema_loop(rest, alpha, x0, [x0]))
  }
}

fn ema_loop(
  xs: List(Float),
  alpha: Float,
  prev: Float,
  acc: List(Float),
) -> List(Float) {
  case xs {
    [] -> list.reverse(acc)
    [x, ..rest] -> {
      let next = alpha *. x +. { 1.0 -. alpha } *. prev
      ema_loop(rest, alpha, next, [next, ..acc])
    }
  }
}

/// Single-step EMA update. Useful for streaming.
pub fn ema_step(previous: Float, observation: Float, alpha: Float) -> Float {
  alpha *. observation +. { 1.0 -. alpha } *. previous
}

// ============================================================================
// Higher moments
// ============================================================================

/// Skewness (Fisher-Pearson) via Pébay online accumulator.
///
/// Numerically stable: avoids the cancellation of `Σ(x - μ)³` over a list.
pub fn skewness(xs: List(Float)) -> Result(Float, Nil) {
  precision.moments_skewness(precision.moments_from_list(xs))
}

/// Excess kurtosis via Pébay online accumulator. Zero for a Gaussian.
pub fn kurtosis(xs: List(Float)) -> Result(Float, Nil) {
  precision.moments_excess_kurtosis(precision.moments_from_list(xs))
}

// ============================================================================
// Helpers
// ============================================================================

fn sort_floats(a: Float, b: Float) -> order.Order {
  case a <. b, a >. b {
    True, _ -> order.Lt
    _, True -> order.Gt
    _, _ -> order.Eq
  }
}

fn list_at(xs: List(Float), idx: Int) -> Result(Float, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at(rest, n - 1)
  }
}

@external(erlang, "erlang", "float")
fn int_to_float_erl(n: Int) -> Float

fn int_to_float(n: Int) -> Float {
  int_to_float_erl(n)
}

@external(erlang, "erlang", "trunc")
fn float_trunc(x: Float) -> Int

fn float_to_int(x: Float) -> Int {
  float_trunc(x)
}

fn float_floor(x: Float) -> Float {
  float.floor(x)
}

fn float_ceil(x: Float) -> Float {
  float.ceiling(x)
}

//// High-precision numerical primitives.
////
//// IEEE-754 doubles lose precision when summing values that differ widely
//// in magnitude. This module provides compensated and exact summation
//// algorithms plus Pébay's online higher-order moment accumulators.
////
//// ## Algorithms shipped
////
//// | Function           | Worst-case error | Cost   | When to use                |
//// | ------------------ | ---------------- | ------ | -------------------------- |
//// | `neumaier_sum`     | O(ε)             | ~3x    | **Default — fast + safe**  |
//// | `kahan_sum`        | O(ε)             | ~4x    | Reference / didactic       |
//// | `pairwise_sum`     | O(log n · ε)     | ~1x    | Reductions on long lists   |
//// | `fsum`             | round-once exact | ~10x   | Critical accuracy           |
////
//// `neumaier_sum` is what CPython 3.12 `sum()` uses by default and matches
//// the Gonum and Julia recommendations. Catastrophic example:
////
//// ```gleam
//// import viva_math/precision
////
//// precision.neumaier_sum([1.0, 1.0e100, 1.0, -1.0e100])
//// // -> 2.0  (naive sum would give 0.0)
//// ```
////
//// ## References
////
//// - Kahan (1965) "Pracniques: further remarks on reducing truncation errors"
//// - Neumaier (1974) "Rundungsfehleranalyse einiger Verfahren zur Summation
////   endlicher Summen"
//// - Shewchuk (1997) "Adaptive Precision Floating-Point Arithmetic"
//// - Pébay (2008) "Formulas for robust, one-pass parallel computation of
////   covariances and arbitrary-order statistical moments" (Sandia)
//// - Pébay, Terriberry et al. (2016) "Numerically Stable, Scalable Formulas
////   for Parallel and Online Computation of Higher-Order Multivariate
////   Central Moments with Arbitrary Weights"
//// - CPython issue #100425 (Python 3.12 switched to Neumaier)

import gleam/float
import gleam/list
import gleam/order

// ============================================================================
// Compensated summation
// ============================================================================

/// Neumaier (Kahan-Babuška improved) compensated sum.
///
/// Maintains a separate compensation accumulator and swaps the role of the
/// running sum / next term depending on magnitude. Recovers about 16 extra
/// bits of precision over naive summation at ~3x the cost.
///
/// ## Example
///
/// ```gleam
/// precision.neumaier_sum([1.0, 1.0e100, 1.0, -1.0e100])
/// // -> 2.0   (naive sum returns 0.0)
/// ```
pub fn neumaier_sum(xs: List(Float)) -> Float {
  neumaier_loop(xs, 0.0, 0.0)
}

fn neumaier_loop(xs: List(Float), sum: Float, comp: Float) -> Float {
  case xs {
    [] -> sum +. comp
    [x, ..rest] -> {
      let t = sum +. x
      let new_comp = case
        float.absolute_value(sum) >=. float.absolute_value(x)
      {
        True -> comp +. { sum -. t +. x }
        False -> comp +. { x -. t +. sum }
      }
      neumaier_loop(rest, t, new_comp)
    }
  }
}

/// Classical Kahan compensated sum.
///
/// Slightly less accurate than `neumaier_sum` (fails on the pathological
/// `[1, 1e100, 1, -1e100]` example) but easier to reason about. Useful as a
/// reference implementation.
pub fn kahan_sum(xs: List(Float)) -> Float {
  kahan_loop(xs, 0.0, 0.0)
}

fn kahan_loop(xs: List(Float), sum: Float, comp: Float) -> Float {
  case xs {
    [] -> sum
    [x, ..rest] -> {
      let y = x -. comp
      let t = sum +. y
      let new_comp = { t -. sum } -. y
      kahan_loop(rest, t, new_comp)
    }
  }
}

/// Pairwise sum: recursively splits and combines.
///
/// Error grows as O(log n · ε) instead of O(n · ε) for naive sum, at the
/// same asymptotic cost. NumPy's default. Good for very long lists where
/// the constant factor of Neumaier matters.
pub fn pairwise_sum(xs: List(Float)) -> Float {
  case xs {
    [] -> 0.0
    [x] -> x
    [a, b] -> a +. b
    _ -> {
      let n = list.length(xs)
      let mid = n / 2
      let left = list.take(xs, mid)
      let right = list.drop(xs, mid)
      pairwise_sum(left) +. pairwise_sum(right)
    }
  }
}

/// Shewchuk's fsum: maintains a list of non-overlapping partial sums.
///
/// Round-to-nearest exact result up to a final rounding. Matches Python's
/// `math.fsum`. The most accurate option available, at ~10x the cost of
/// naive summation.
pub fn fsum(xs: List(Float)) -> Float {
  let partials = list.fold(xs, [], fsum_add)
  list.fold(partials, 0.0, fn(acc, p) { acc +. p })
}

fn fsum_add(partials: List(Float), x: Float) -> List(Float) {
  fsum_merge(partials, x, [])
}

fn fsum_merge(
  partials: List(Float),
  x: Float,
  acc: List(Float),
) -> List(Float) {
  case partials {
    [] ->
      case x == 0.0 {
        True -> list.reverse(acc)
        False -> list.reverse([x, ..acc])
      }
    [p, ..rest] -> {
      let #(hi, lo) = two_sum(p, x)
      case lo == 0.0 {
        True -> fsum_merge(rest, hi, acc)
        False -> fsum_merge(rest, hi, [lo, ..acc])
      }
    }
  }
}

/// Two-sum: returns the exact sum `a+b` as a non-overlapping pair `(hi, lo)`
/// where `hi = round(a+b)` and `lo = (a+b) - hi`.
pub fn two_sum(a: Float, b: Float) -> #(Float, Float) {
  let s = a +. b
  let bp = s -. a
  let ap = s -. bp
  let lo = { a -. ap } +. { b -. bp }
  #(s, lo)
}

// ============================================================================
// Mean from compensated sum
// ============================================================================

/// Mean using Neumaier-compensated sum. Errors on empty input.
pub fn neumaier_mean(xs: List(Float)) -> Result(Float, Nil) {
  case xs {
    [] -> Error(Nil)
    _ -> Ok(neumaier_sum(xs) /. int_to_float(list.length(xs)))
  }
}

// ============================================================================
// Pébay higher-order moments (online accumulator up to M₄)
// ============================================================================

/// Online accumulator for moments up to fourth order.
///
/// Updates Pébay-style central moments without storing the data. Numerically
/// stable through cancellation tricks; computes variance, skewness, and
/// kurtosis from streaming input.
pub type Moments {
  Moments(count: Int, mean: Float, m2: Float, m3: Float, m4: Float)
}

/// Empty accumulator.
pub fn moments_empty() -> Moments {
  Moments(count: 0, mean: 0.0, m2: 0.0, m3: 0.0, m4: 0.0)
}

/// Update with a single sample (Pébay 2008 recurrence).
///
/// Let n be the new count, δ = x - mean, δ_n = δ/n, δ_n² and term1 = δ·δ_n·(n-1).
/// Then:
///   M₂ ← M₂ + term1
///   M₃ ← M₃ + term1·δ_n·(n-2) - 3·δ_n·M₂
///   M₄ ← M₄ + term1·δ_n²·(n²-3n+3) + 6·δ_n²·M₂ - 4·δ_n·M₃
///
/// Updates the mean last to preserve the previous-iteration moments above.
pub fn moments_update(m: Moments, x: Float) -> Moments {
  let n1 = m.count + 1
  let n1_float = int_to_float(n1)
  let delta = x -. m.mean
  let delta_n = delta /. n1_float
  let delta_n_sq = delta_n *. delta_n
  let n_prev = m.count
  let n_prev_float = int_to_float(n_prev)
  let term1 = delta *. delta_n *. n_prev_float

  let new_m4 =
    m.m4
    +. term1
    *. delta_n_sq
    *. { n1_float *. n1_float -. 3.0 *. n1_float +. 3.0 }
    +. 6.0
    *. delta_n_sq
    *. m.m2
    -. 4.0
    *. delta_n
    *. m.m3

  let new_m3 =
    m.m3 +. term1 *. delta_n *. { n1_float -. 2.0 } -. 3.0 *. delta_n *. m.m2

  let new_m2 = m.m2 +. term1
  let new_mean = m.mean +. delta_n

  Moments(count: n1, mean: new_mean, m2: new_m2, m3: new_m3, m4: new_m4)
}

/// Build accumulator from a list.
pub fn moments_from_list(xs: List(Float)) -> Moments {
  list.fold(xs, moments_empty(), moments_update)
}

/// Population mean.
pub fn moments_mean(m: Moments) -> Result(Float, Nil) {
  case m.count {
    0 -> Error(Nil)
    _ -> Ok(m.mean)
  }
}

/// Population variance σ² = M₂ / n.
pub fn moments_variance(m: Moments) -> Result(Float, Nil) {
  case m.count {
    0 -> Error(Nil)
    _ -> Ok(m.m2 /. int_to_float(m.count))
  }
}

/// Sample variance s² = M₂ / (n-1).
pub fn moments_sample_variance(m: Moments) -> Result(Float, Nil) {
  case m.count {
    0 | 1 -> Error(Nil)
    _ -> Ok(m.m2 /. int_to_float(m.count - 1))
  }
}

/// Population skewness γ₁ = (M₃/n) / (M₂/n)^(3/2) = √n · M₃ / M₂^(3/2).
pub fn moments_skewness(m: Moments) -> Result(Float, Nil) {
  case m.count {
    0 -> Error(Nil)
    n -> {
      case m.m2 <=. 0.0 {
        True -> Error(Nil)
        False -> {
          let n_float = int_to_float(n)
          // sqrt(n) · M3 / M2^(3/2)
          let m2_sqrt = sqrt(m.m2)
          let m2_pow_3_2 = m2_sqrt *. m.m2
          Ok(sqrt(n_float) *. m.m3 /. m2_pow_3_2)
        }
      }
    }
  }
}

/// Excess kurtosis γ₂ = n · M₄ / M₂² - 3.
pub fn moments_excess_kurtosis(m: Moments) -> Result(Float, Nil) {
  case m.count {
    0 -> Error(Nil)
    n -> {
      case m.m2 <=. 0.0 {
        True -> Error(Nil)
        False -> {
          let n_float = int_to_float(n)
          Ok(n_float *. m.m4 /. { m.m2 *. m.m2 } -. 3.0)
        }
      }
    }
  }
}

/// Combine two Pébay accumulators computed in parallel (Chan formula).
///
/// Useful for distributed / parallel reductions: split a stream across
/// workers, then merge.
pub fn moments_combine(a: Moments, b: Moments) -> Moments {
  case a.count, b.count {
    0, _ -> b
    _, 0 -> a
    na, nb -> {
      let na_f = int_to_float(na)
      let nb_f = int_to_float(nb)
      let n = na + nb
      let n_f = int_to_float(n)
      let delta = b.mean -. a.mean
      let delta2 = delta *. delta
      let delta3 = delta2 *. delta
      let delta4 = delta2 *. delta2
      let new_mean = a.mean +. delta *. nb_f /. n_f
      let new_m2 = a.m2 +. b.m2 +. delta2 *. na_f *. nb_f /. n_f
      let new_m3 =
        a.m3
        +. b.m3
        +. delta3
        *. na_f
        *. nb_f
        *. { na_f -. nb_f }
        /. { n_f *. n_f }
        +. 3.0
        *. delta
        *. { na_f *. b.m2 -. nb_f *. a.m2 }
        /. n_f
      let new_m4 =
        a.m4
        +. b.m4
        +. delta4
        *. na_f
        *. nb_f
        *. { na_f *. na_f -. na_f *. nb_f +. nb_f *. nb_f }
        /. { n_f *. n_f *. n_f }
        +. 6.0
        *. delta2
        *. { na_f *. na_f *. b.m2 +. nb_f *. nb_f *. a.m2 }
        /. { n_f *. n_f }
        +. 4.0
        *. delta
        *. { na_f *. b.m3 -. nb_f *. a.m3 }
        /. n_f
      Moments(count: n, mean: new_mean, m2: new_m2, m3: new_m3, m4: new_m4)
    }
  }
}

// ============================================================================
// Internal helpers
// ============================================================================

@external(erlang, "erlang", "float")
@external(javascript, "../viva_math_random_ffi.mjs", "int_to_float")
fn int_to_float(n: Int) -> Float

@external(erlang, "math", "sqrt")
@external(javascript, "../viva_math_random_ffi.mjs", "sqrt")
fn sqrt(x: Float) -> Float

/// Stable cmp by absolute value (descending). Useful for sorting before
/// summation in worst-case scenarios.
pub fn cmp_abs_desc(a: Float, b: Float) -> order.Order {
  case
    float.absolute_value(a) >. float.absolute_value(b),
    float.absolute_value(a) <. float.absolute_value(b)
  {
    True, _ -> order.Lt
    _, True -> order.Gt
    _, _ -> order.Eq
  }
}

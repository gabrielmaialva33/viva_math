//// Optimal transport distances for empirical affective distributions.
////
//// Implements one-dimensional empirical Wasserstein distances and PAD
//// component aggregation following Villani (2008), *Optimal Transport*, and
//// Peyre & Cuturi (2019), *Computational Optimal Transport*.

import gleam/float
import gleam/list
import gleam/order
import viva_math/distributions
import viva_math/scalar
import viva_math/vector

/// Wasserstein-1 (Earth Mover) between 1D empirical samples.
/// W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx.
/// Uses O(n log n) sorting plus pairwise quantile differences when sample sizes
/// match; otherwise integrates empirical CDF gaps over the union of samples
/// (Villani 2008; Peyre & Cuturi 2019).
pub fn wasserstein_1_empirical(
  p: List(Float),
  q: List(Float),
) -> Result(Float, Nil) {
  case p, q {
    [], _ -> Error(Nil)
    _, [] -> Error(Nil)
    _, _ -> {
      let p_sorted = sort_samples(p)
      let q_sorted = sort_samples(q)
      let p_len = list.length(p_sorted)
      let q_len = list.length(q_sorted)

      case p_len == q_len {
        True -> {
          let total =
            list.zip(p_sorted, q_sorted)
            |> list.fold(0.0, fn(acc, pair) {
              let #(a, b) = pair
              acc +. float.absolute_value(a -. b)
            })

          Ok(total /. int_to_float(p_len))
        }
        False -> Ok(integrate_cdf_gap(p_sorted, q_sorted, False))
      }
    }
  }
}

/// Wasserstein-2 between 1D empirical samples.
///
/// `W_2²(P, Q) = ∫_0^1 (F_P⁻¹(u) − F_Q⁻¹(u))² du`.
///
/// For equal sample sizes, the inverse-CDF integral reduces to the sorted
/// pairwise mean-square difference. For unequal sizes we integrate over the
/// union of quantile breakpoints `{i/n} ∪ {j/m}`: in each slab `[u_prev, u_next]`
/// both inverse CDFs are constant at `p_sorted[i]` and `q_sorted[j]`, so the
/// contribution is `(p_sorted[i] − q_sorted[j])² · (u_next − u_prev)`.
///
/// Returns `W_2`, not `W_2²`. References: Villani (2008); Peyré & Cuturi (2019).
///
/// **Note**: a CDF-based path `∫(F_P−F_Q)² dx` would be wrong for unequal
/// sample sizes — the `W_1`/`W_2` duality via integration by parts only holds
/// for `p=1` (absolute value), not for the quadratic kernel.
pub fn wasserstein_2_empirical(
  p: List(Float),
  q: List(Float),
) -> Result(Float, Nil) {
  case p, q {
    [], _ -> Error(Nil)
    _, [] -> Error(Nil)
    _, _ -> {
      let p_sorted = sort_samples(p)
      let q_sorted = sort_samples(q)
      let p_len = list.length(p_sorted)
      let q_len = list.length(q_sorted)

      case p_len == q_len {
        True -> {
          let total =
            list.zip(p_sorted, q_sorted)
            |> list.fold(0.0, fn(acc, pair) {
              let #(a, b) = pair
              let delta = a -. b
              acc +. delta *. delta
            })
          Ok(scalar.sqrt(total /. int_to_float(p_len)))
        }
        False ->
          Ok(
            scalar.sqrt(quantile_integral_squared(
              p_sorted,
              q_sorted,
              int_to_float(p_len),
              int_to_float(q_len),
              0,
              0,
              0.0,
              0.0,
            )),
          )
      }
    }
  }
}

/// Quantile-based integral `∫_0^1 (F_P⁻¹(u) − F_Q⁻¹(u))² du` for sorted
/// empirical samples of unequal sizes. Walks both breakpoint sequences
/// `(i+1)/n` and `(j+1)/m` in tandem, accumulating squared-difference slabs.
fn quantile_integral_squared(
  p_sorted: List(Float),
  q_sorted: List(Float),
  n: Float,
  m: Float,
  i: Int,
  j: Int,
  u_prev: Float,
  acc: Float,
) -> Float {
  case nth(p_sorted, i), nth(q_sorted, j) {
    Ok(p_val), Ok(q_val) -> {
      let u_p = int_to_float(i + 1) /. n
      let u_q = int_to_float(j + 1) /. m
      let u_next = float.min(u_p, u_q)
      let delta = p_val -. q_val
      let new_acc = acc +. delta *. delta *. { u_next -. u_prev }
      let next_i = case u_p <=. u_q {
        True -> i + 1
        False -> i
      }
      let next_j = case u_q <=. u_p {
        True -> j + 1
        False -> j
      }
      quantile_integral_squared(
        p_sorted,
        q_sorted,
        n,
        m,
        next_i,
        next_j,
        u_next,
        new_acc,
      )
    }
    _, _ -> acc
  }
}

fn nth(xs: List(a), idx: Int) -> Result(a, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n if n > 0 -> nth(rest, n - 1)
    _, _ -> Error(Nil)
  }
}

/// Closed form W_2 for scalar Gaussians.
/// W_2(N(mu_1, sigma_1^2), N(mu_2, sigma_2^2))
/// = sqrt((mu_1 - mu_2)^2 + (sigma_1 - sigma_2)^2).
pub fn wasserstein_2_gaussian(
  g1: distributions.Gaussian,
  g2: distributions.Gaussian,
) -> Float {
  let mean_delta = g1.mean -. g2.mean
  let stddev_delta = g1.stddev -. g2.stddev
  scalar.sqrt(mean_delta *. mean_delta +. stddev_delta *. stddev_delta)
}

/// Component-wise (marginal) Wasserstein-2 over PAD dimensions.
///
/// `D(P, Q) = √( W_2²(P_P, Q_P) + W_2²(P_A, Q_A) + W_2²(P_D, Q_D) )`.
///
/// This is **not** the multivariate W_2 (which requires solving a full
/// optimal-transport assignment with cost `‖x − y‖²`). It is the Euclidean
/// norm of the per-axis marginal Wasserstein distances — equivalent to the
/// Sliced Wasserstein along the canonical PAD basis.
///
/// Triangle inequality holds (Minkowski over the marginal vector), so this
/// is a **pseudo-metric** on PAD distributions: `D(P, Q) = 0` does **not**
/// imply `P = Q` as joints — two distributions with identical marginals but
/// different correlations are tied. Useful as a fast lower bound on the true
/// multivariate W_2; tight when marginals are product distributions.
pub fn wasserstein_pad(
  p: List(vector.Vec3),
  q: List(vector.Vec3),
) -> Result(Float, Nil) {
  case p, q {
    [], _ -> Error(Nil)
    _, [] -> Error(Nil)
    _, _ -> {
      let p_pleasure = list.map(p, vector.pleasure)
      let q_pleasure = list.map(q, vector.pleasure)
      let p_arousal = list.map(p, vector.arousal)
      let q_arousal = list.map(q, vector.arousal)
      let p_dominance = list.map(p, vector.dominance)
      let q_dominance = list.map(q, vector.dominance)

      case
        wasserstein_2_empirical(p_pleasure, q_pleasure),
        wasserstein_2_empirical(p_arousal, q_arousal),
        wasserstein_2_empirical(p_dominance, q_dominance)
      {
        Ok(pleasure), Ok(arousal), Ok(dominance) ->
          Ok(scalar.sqrt(
            pleasure *. pleasure +. arousal *. arousal +. dominance *. dominance,
          ))
        _, _, _ -> Error(Nil)
      }
    }
  }
}

fn integrate_cdf_gap(
  p_sorted: List(Float),
  q_sorted: List(Float),
  squared: Bool,
) -> Float {
  let points = list.append(p_sorted, q_sorted) |> sort_samples
  let p_len = int_to_float(list.length(p_sorted))
  let q_len = int_to_float(list.length(q_sorted))

  case points {
    [] -> 0.0
    [first, ..rest] ->
      integrate_cdf_gap_from(
        rest,
        first,
        p_sorted,
        q_sorted,
        p_len,
        q_len,
        squared,
        0.0,
      )
  }
}

fn integrate_cdf_gap_from(
  points: List(Float),
  current: Float,
  p_sorted: List(Float),
  q_sorted: List(Float),
  p_len: Float,
  q_len: Float,
  squared: Bool,
  acc: Float,
) -> Float {
  let delta =
    empirical_cdf_at(p_sorted, current, p_len)
    -. empirical_cdf_at(q_sorted, current, q_len)
  let height = case squared {
    True -> delta *. delta
    False -> float.absolute_value(delta)
  }

  case points {
    [] -> acc
    [next, ..rest] ->
      integrate_cdf_gap_from(
        rest,
        next,
        p_sorted,
        q_sorted,
        p_len,
        q_len,
        squared,
        acc +. height *. { next -. current },
      )
  }
}

fn empirical_cdf_at(sorted: List(Float), x: Float, len: Float) -> Float {
  int_to_float(count_less_or_equal(sorted, x, 0)) /. len
}

fn count_less_or_equal(sorted: List(Float), x: Float, count: Int) -> Int {
  case sorted {
    [] -> count
    [value, ..rest] -> {
      case value <=. x {
        True -> count_less_or_equal(rest, x, count + 1)
        False -> count
      }
    }
  }
}

fn sort_samples(samples: List(Float)) -> List(Float) {
  list.sort(samples, fn(a, b) {
    case a <. b, a >. b {
      True, _ -> order.Lt
      _, True -> order.Gt
      _, _ -> order.Eq
    }
  })
}

@external(erlang, "erlang", "float")
fn int_to_float(n: Int) -> Float

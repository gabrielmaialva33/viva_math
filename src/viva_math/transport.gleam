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
/// W_2^2(P, Q) = integral (F_P^-1(t) - F_Q^-1(t))^2 dt.
/// Uses sorted empirical quantiles for equal sample sizes; for unequal sample
/// sizes it integrates squared empirical CDF gaps over the sample union
/// (Villani 2008; Peyre & Cuturi 2019). Returns W_2, not W_2^2.
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
        False -> Ok(scalar.sqrt(integrate_cdf_gap(p_sorted, q_sorted, True)))
      }
    }
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

/// Component-wise W_2 over PAD dimensions.
/// Sums W_2^2(p_i, q_i) across pleasure, arousal, and dominance, then returns
/// the square root (Villani 2008; Peyre & Cuturi 2019).
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

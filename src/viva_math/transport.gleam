//// Optimal transport distances for empirical affective distributions.
////
//// 1D empirical Wasserstein distances and PAD componentwise aggregation
//// following Villani (2008), *Optimal Transport*, and Peyré & Cuturi (2019),
//// *Computational Optimal Transport*.
////
//// **Complexity**: O((n+m)·log(n+m)) dominated by sort. The post-sort
//// integral walks both quantile sequences in linear time without random
//// indexing — see `walk_quantile_*` helpers below.

import gleam/float
import gleam/list
import gleam/order
import viva_math/distributions
import viva_math/scalar
import viva_math/vector

/// Wasserstein-1 (Earth Mover) between 1D empirical samples.
///
/// `W_1(P, Q) = ∫_0^1 |F_P⁻¹(u) − F_Q⁻¹(u)| du`.
///
/// Equivalent to `∫_ℝ |F_P(x) − F_Q(x)| dx` (the `W_1` duality holds via
/// integration by parts; the absolute-value identity is symmetric). For
/// equal sample sizes reduces to `(1/n)·Σ |p_(i) − q_(i)|`. For unequal sizes
/// integrates `|p_i − q_j|·Δu` across the union of quantile breakpoints
/// `{i/n} ∪ {j/m}` in linear time after sorting.
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
        True ->
          Ok(walk_pair_abs(p_sorted, q_sorted, 0.0) /. int_to_float(p_len))
        False ->
          Ok(walk_quantile(
            p_sorted,
            q_sorted,
            int_to_float(p_len),
            int_to_float(q_len),
            0,
            0,
            0.0,
            0.0,
            False,
          ))
      }
    }
  }
}

/// Wasserstein-2 between 1D empirical samples.
///
/// `W_2²(P, Q) = ∫_0^1 (F_P⁻¹(u) − F_Q⁻¹(u))² du`.
///
/// For equal sample sizes the inverse-CDF integral reduces to the sorted
/// pairwise mean-square difference. For unequal sizes we integrate over the
/// union of quantile breakpoints `{i/n} ∪ {j/m}` in linear time: in each
/// slab `[u_prev, u_next]` both inverse CDFs are constant, contributing
/// `(p_sorted[i] − q_sorted[j])² · (u_next − u_prev)`.
///
/// Returns `W_2`, not `W_2²`. References: Villani (2008); Peyré & Cuturi (2019).
///
/// **Note**: a CDF-based path `∫(F_P−F_Q)² dx` is **incorrect** for unequal
/// sample sizes — the `W_1`/`W_2` duality via integration by parts only
/// holds for `p=1` (absolute value), not for the quadratic kernel.
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
        True ->
          Ok(scalar.sqrt(
            walk_pair_squared(p_sorted, q_sorted, 0.0) /. int_to_float(p_len),
          ))
        False ->
          Ok(
            scalar.sqrt(walk_quantile(
              p_sorted,
              q_sorted,
              int_to_float(p_len),
              int_to_float(q_len),
              0,
              0,
              0.0,
              0.0,
              True,
            )),
          )
      }
    }
  }
}

/// Closed form W_2 for scalar Gaussians.
///
/// `W_2(N(μ₁, σ₁²), N(μ₂, σ₂²)) = √((μ₁ − μ₂)² + (σ₁ − σ₂)²)`.
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
      let #(p_p, p_a, p_d) = split_pad(p, [], [], [])
      let #(q_p, q_a, q_d) = split_pad(q, [], [], [])
      case
        wasserstein_2_empirical(p_p, q_p),
        wasserstein_2_empirical(p_a, q_a),
        wasserstein_2_empirical(p_d, q_d)
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

// ============================================================================
// Internal — linear-time walks
// ============================================================================

/// Equal-size W_2 numerator `Σ (p_i − q_i)²` via single-pass cons traversal,
/// no `list.zip` intermediate.
fn walk_pair_squared(p: List(Float), q: List(Float), acc: Float) -> Float {
  case p, q {
    [a, ..rest_p], [b, ..rest_q] -> {
      let d = a -. b
      walk_pair_squared(rest_p, rest_q, acc +. d *. d)
    }
    _, _ -> acc
  }
}

/// Equal-size W_1 numerator `Σ |p_i − q_i|` via single-pass cons traversal.
fn walk_pair_abs(p: List(Float), q: List(Float), acc: Float) -> Float {
  case p, q {
    [a, ..rest_p], [b, ..rest_q] ->
      walk_pair_abs(rest_p, rest_q, acc +. float.absolute_value(a -. b))
    _, _ -> acc
  }
}

/// O(n+m) quantile-integral walk over the breakpoint union `{(i+1)/n} ∪
/// `{(j+1)/m}`. `squared=True` gives the W_2² integrand; otherwise W_1.
///
/// Both lists are consumed by their heads — never indexed by position — so
/// each step is O(1) and the total is O(n+m).
fn walk_quantile(
  p_sorted: List(Float),
  q_sorted: List(Float),
  n: Float,
  m: Float,
  i: Int,
  j: Int,
  u_prev: Float,
  acc: Float,
  squared: Bool,
) -> Float {
  case p_sorted, q_sorted {
    [p_val, ..p_tail], [q_val, ..q_tail] -> {
      let u_p = int_to_float(i + 1) /. n
      let u_q = int_to_float(j + 1) /. m
      let u_next = float.min(u_p, u_q)
      let delta = p_val -. q_val
      let height = case squared {
        True -> delta *. delta
        False -> float.absolute_value(delta)
      }
      let new_acc = acc +. height *. { u_next -. u_prev }
      // Advance whichever list has the smaller breakpoint (or both on tie).
      let advance_p = u_p <=. u_q
      let advance_q = u_q <=. u_p
      let next_p = case advance_p {
        True -> p_tail
        False -> p_sorted
      }
      let next_q = case advance_q {
        True -> q_tail
        False -> q_sorted
      }
      let next_i = case advance_p {
        True -> i + 1
        False -> i
      }
      let next_j = case advance_q {
        True -> j + 1
        False -> j
      }
      walk_quantile(
        next_p,
        next_q,
        n,
        m,
        next_i,
        next_j,
        u_next,
        new_acc,
        squared,
      )
    }
    _, _ -> acc
  }
}

/// Single-pass PAD projection — avoids three `list.map` passes.
fn split_pad(
  xs: List(vector.Vec3),
  p_acc: List(Float),
  a_acc: List(Float),
  d_acc: List(Float),
) -> #(List(Float), List(Float), List(Float)) {
  case xs {
    [] -> #(list.reverse(p_acc), list.reverse(a_acc), list.reverse(d_acc))
    [v, ..rest] ->
      split_pad(
        rest,
        [vector.pleasure(v), ..p_acc],
        [vector.arousal(v), ..a_acc],
        [vector.dominance(v), ..d_acc],
      )
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

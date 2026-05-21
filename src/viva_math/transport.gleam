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
          Ok(walk_quantile_abs(
            p_sorted,
            q_sorted,
            int_to_float(p_len),
            int_to_float(q_len),
            0,
            0,
            0.0,
            0.0,
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
            scalar.sqrt(walk_quantile_squared(
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

/// Closed form W_2 for scalar Gaussians.
///
/// `W_2(N(μ₁, σ₁²), N(μ₂, σ₂²)) = √((μ₁ − μ₂)² + (σ₁ − σ₂)²)`.
///
/// `stddev` is normalised via `abs` before subtracting — a negative `stddev`
/// has no Gaussian meaning (`N(μ, σ²)` depends only on `σ²`), and silently
/// using the signed value would yield `W_2(N(0, 1), N(0, 1))` of `2.0` for
/// `Gaussian(0, -1) vs Gaussian(0, 1)`, which is wrong.
pub fn wasserstein_2_gaussian(
  g1: distributions.Gaussian,
  g2: distributions.Gaussian,
) -> Float {
  let mean_delta = g1.mean -. g2.mean
  let stddev_delta =
    float.absolute_value(g1.stddev) -. float.absolute_value(g2.stddev)
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

/// Multivariate Wasserstein-2 distance via Sinkhorn-Knopp entropic
/// regularization (Cuturi 2013).
///
/// Solves the entropic OT problem
/// `min_π ⟨π, C⟩ + ε·H(π)` subject to `π·1 = a`, `πᵀ·1 = b`,
/// where `C[i,j] = ‖x_i − y_j‖²` and `π` is the transport plan.
///
/// Returns `√(⟨π, C⟩)`, the W₂ distance induced by the regularized plan,
/// dropping the entropic term itself.
///
/// - `xs`, `ys`: empirical PAD samples with uniform weights.
/// - `epsilon`: regularization strength, typically `0.01` to `1.0`.
/// - `max_iter`: Sinkhorn iterations, typically `100` to `1000`.
/// - Returns `Error(Nil)` if either list is empty.
pub fn wasserstein_2_multivariate(
  xs: List(vector.Vec3),
  ys: List(vector.Vec3),
  epsilon: Float,
  max_iter: Int,
) -> Result(Float, Nil) {
  case xs, ys {
    [], _ -> Error(Nil)
    _, [] -> Error(Nil)
    _, _ -> {
      let n = int_to_float(list.length(xs))
      let m = int_to_float(list.length(ys))
      let a = 1.0 /. n
      let b = 1.0 /. m
      let costs = cost_matrix(xs, ys)
      let kernel = sinkhorn_kernel(costs, float.max(epsilon, 1.0e-12))
      let u0 = fill(list.length(xs), a, [])
      let v0 = fill(list.length(ys), b, [])
      let #(u, v) = sinkhorn_iter(kernel, max_iter, a, b, u0, v0)
      let cost = transport_cost(costs, kernel, u, v)
      Ok(scalar.sqrt(float.max(cost, 0.0)))
    }
  }
}

// ============================================================================
// Internal — linear-time walks
// ============================================================================

fn cost_matrix(
  xs: List(vector.Vec3),
  ys: List(vector.Vec3),
) -> List(List(Float)) {
  list.map(xs, fn(x) { list.map(ys, fn(y) { vector.distance_squared(x, y) }) })
}

fn sinkhorn_kernel(
  costs: List(List(Float)),
  epsilon: Float,
) -> List(List(Float)) {
  list.map(costs, fn(row) {
    list.map(row, fn(cost) { scalar.exp(0.0 -. cost /. epsilon) })
  })
}

fn sinkhorn_iter(
  kernel: List(List(Float)),
  remaining: Int,
  a: Float,
  b: Float,
  u: List(Float),
  v: List(Float),
) -> #(List(Float), List(Float)) {
  case remaining <= 0 {
    True -> #(u, v)
    False -> {
      let next_u = scale_inverse_rows(kernel, v, a)
      let next_v = scale_inverse_columns(kernel, next_u, b)
      sinkhorn_iter(kernel, remaining - 1, a, b, next_u, next_v)
    }
  }
}

fn scale_inverse_rows(
  kernel: List(List(Float)),
  v: List(Float),
  mass: Float,
) -> List(Float) {
  list.map(kernel, fn(row) { safe_ratio(mass, dot_lists(row, v, 0.0)) })
}

fn scale_inverse_columns(
  kernel: List(List(Float)),
  u: List(Float),
  mass: Float,
) -> List(Float) {
  case kernel {
    [] -> []
    [first, ..] -> scale_inverse_columns_walk(first, kernel, u, mass, [])
  }
}

fn scale_inverse_columns_walk(
  first_row: List(Float),
  kernel: List(List(Float)),
  u: List(Float),
  mass: Float,
  acc: List(Float),
) -> List(Float) {
  case first_row {
    [] -> list.reverse(acc)
    [_, ..rest] -> {
      let column_sum = weighted_column_head(kernel, u, 0.0)
      let tails = drop_column_head(kernel, [])
      scale_inverse_columns_walk(rest, tails, u, mass, [
        safe_ratio(mass, column_sum),
        ..acc
      ])
    }
  }
}

fn weighted_column_head(
  rows: List(List(Float)),
  weights: List(Float),
  acc: Float,
) -> Float {
  case rows, weights {
    [[x, ..], ..rest_rows], [w, ..rest_weights] ->
      weighted_column_head(rest_rows, rest_weights, acc +. w *. x)
    _, _ -> acc
  }
}

fn drop_column_head(
  rows: List(List(Float)),
  acc: List(List(Float)),
) -> List(List(Float)) {
  case rows {
    [] -> list.reverse(acc)
    [[_, ..rest], ..tail] -> drop_column_head(tail, [rest, ..acc])
    [[], ..tail] -> drop_column_head(tail, [[], ..acc])
  }
}

fn transport_cost(
  costs: List(List(Float)),
  kernel: List(List(Float)),
  u: List(Float),
  v: List(Float),
) -> Float {
  case costs, kernel, u {
    [cost_row, ..cost_tail], [kernel_row, ..kernel_tail], [u_i, ..u_tail] ->
      transport_cost_row(cost_row, kernel_row, u_i, v, 0.0)
      +. transport_cost(cost_tail, kernel_tail, u_tail, v)
    _, _, _ -> 0.0
  }
}

fn transport_cost_row(
  costs: List(Float),
  kernel: List(Float),
  u_i: Float,
  v: List(Float),
  acc: Float,
) -> Float {
  case costs, kernel, v {
    [cost, ..cost_tail], [k, ..kernel_tail], [v_j, ..v_tail] ->
      transport_cost_row(
        cost_tail,
        kernel_tail,
        u_i,
        v_tail,
        acc +. u_i *. k *. v_j *. cost,
      )
    _, _, _ -> acc
  }
}

fn dot_lists(a: List(Float), b: List(Float), acc: Float) -> Float {
  case a, b {
    [x, ..xs], [y, ..ys] -> dot_lists(xs, ys, acc +. x *. y)
    _, _ -> acc
  }
}

fn safe_ratio(numerator: Float, denominator: Float) -> Float {
  case denominator <=. 0.0 {
    True -> 0.0
    False -> numerator /. denominator
  }
}

fn fill(count: Int, value: Float, acc: List(Float)) -> List(Float) {
  case count <= 0 {
    True -> acc
    False -> fill(count - 1, value, [value, ..acc])
  }
}

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

/// O(n+m) quantile-integral walk for W_2² over `{(i+1)/n} ∪ {(j+1)/m}`.
/// Slab contribution `(p_i − q_j)² · Δu`. Specialised to avoid a `Bool`
/// branch in the hot loop (BEAM doesn't specialise on literal `True`/`False`).
fn walk_quantile_squared(
  p_sorted: List(Float),
  q_sorted: List(Float),
  n: Float,
  m: Float,
  i: Int,
  j: Int,
  u_prev: Float,
  acc: Float,
) -> Float {
  case p_sorted, q_sorted {
    [p_val, ..p_tail], [q_val, ..q_tail] -> {
      let u_p = int_to_float(i + 1) /. n
      let u_q = int_to_float(j + 1) /. m
      let u_next = float.min(u_p, u_q)
      let delta = p_val -. q_val
      let new_acc = acc +. delta *. delta *. { u_next -. u_prev }
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
      walk_quantile_squared(
        next_p,
        next_q,
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

/// O(n+m) quantile-integral walk for W_1 over `{(i+1)/n} ∪ {(j+1)/m}`.
/// Slab contribution `|p_i − q_j| · Δu`. Twin of `walk_quantile_squared`.
fn walk_quantile_abs(
  p_sorted: List(Float),
  q_sorted: List(Float),
  n: Float,
  m: Float,
  i: Int,
  j: Int,
  u_prev: Float,
  acc: Float,
) -> Float {
  case p_sorted, q_sorted {
    [p_val, ..p_tail], [q_val, ..q_tail] -> {
      let u_p = int_to_float(i + 1) /. n
      let u_q = int_to_float(j + 1) /. m
      let u_next = float.min(u_p, u_q)
      let new_acc =
        acc +. float.absolute_value(p_val -. q_val) *. { u_next -. u_prev }
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
      walk_quantile_abs(next_p, next_q, n, m, next_i, next_j, u_next, new_acc)
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

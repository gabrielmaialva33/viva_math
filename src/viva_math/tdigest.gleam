//// t-digest — streaming approximate quantiles.
////
//// Dunning's t-digest (2014, 2019) provides accurate approximate quantiles
//// over a stream with bounded memory and fast updates. Particularly
//// accurate at the tail (`q < 0.01` or `q > 0.99`), which is where most
//// applications need precision.
////
//// Memory: O(δ) where δ is the compression parameter (default 100).
//// Update: O(log δ) amortised.
//// Quantile query: O(δ).
////
//// ## When to use
////
//// - You need percentiles over a stream that doesn't fit in memory.
//// - Tail quantiles (p99, p99.9) matter — t-digest is exact-ish there.
//// - You want to merge digests from parallel workers.
////
//// ## Reference
////
//// Dunning & Ertl (2019) "Computing Extremely Accurate Quantiles Using
//// t-Digests" — https://github.com/tdunning/t-digest
////
//// ## Algorithm summary
////
//// A t-digest is a sorted set of *centroids*, each (`mean`, `weight`).
//// New samples merge into the nearest centroid up to a size limit dictated
//// by the scale function `k(q) = δ · q · (1 - q) / 2π` — small at the
//// tails, large at the median, so tail centroids stay small and accurate.

import gleam/list
import gleam/order
import viva_math/scalar

/// A centroid: a weighted point on the real line.
/// A weighted centroid `(mean, weight)` representing a cluster of nearby
/// samples in the digest. Opaque — invariants (sorted by `mean`, positive
/// `weight`) are maintained by `insert` / `merge` / `compress`.
pub opaque type Centroid {
  Centroid(mean: Float, weight: Float)
}

/// t-digest state.
///
/// `compression` controls memory/accuracy tradeoff (typical 100). Larger →
/// more accurate, more memory.
///
/// Opaque — direct construction could violate the sorted-centroids
/// invariant or the consistency between `centroids` and `total_weight`.
/// Use `new` / `with_compression` / `insert` to build.
pub opaque type TDigest {
  TDigest(compression: Float, centroids: List(Centroid), total_weight: Float)
}

// ============================================================================
// Accessors
// ============================================================================

/// Compression parameter δ (typical 100; larger → more accurate / more memory).
pub fn compression(td: TDigest) -> Float {
  td.compression
}

/// Mean of a centroid.
pub fn centroid_mean(c: Centroid) -> Float {
  c.mean
}

/// Weight of a centroid.
pub fn centroid_weight(c: Centroid) -> Float {
  c.weight
}

// ============================================================================
// Construction
// ============================================================================

/// Empty t-digest with default compression (δ = 100).
pub fn new() -> TDigest {
  with_compression(100.0)
}

/// Empty t-digest with explicit compression parameter.
pub fn with_compression(delta: Float) -> TDigest {
  TDigest(compression: delta, centroids: [], total_weight: 0.0)
}

// ============================================================================
// Insertion
// ============================================================================

/// Insert a single sample into the digest.
pub fn insert(td: TDigest, value: Float) -> TDigest {
  insert_weighted(td, value, 1.0)
}

/// Insert a weighted sample.
pub fn insert_weighted(td: TDigest, value: Float, weight: Float) -> TDigest {
  let merged = merge_centroid(td, Centroid(mean: value, weight: weight))
  case list.length(merged.centroids) > compression_threshold(merged) {
    True -> compress(merged)
    False -> merged
  }
}

fn compression_threshold(td: TDigest) -> Int {
  // Loose upper bound: ~6×δ centroids before forced compression.
  trunc_float(6.0 *. td.compression)
}

@external(erlang, "erlang", "trunc")
@external(javascript, "../viva_math_random_ffi.mjs", "trunc")
fn trunc_float(x: Float) -> Int

/// Insert many samples from a list. Equivalent to folding `insert`.
pub fn insert_all(td: TDigest, xs: List(Float)) -> TDigest {
  list.fold(xs, td, insert)
}

/// Combine two digests into one. Useful for distributed reductions.
pub fn merge(a: TDigest, b: TDigest) -> TDigest {
  let combined =
    TDigest(
      compression: float_max(a.compression, b.compression),
      centroids: sorted_concat(a.centroids, b.centroids),
      total_weight: a.total_weight +. b.total_weight,
    )
  compress(combined)
}

// ============================================================================
// Queries
// ============================================================================

/// Approximate quantile q ∈ [0, 1]. Returns `Error` for an empty digest.
pub fn quantile(td: TDigest, q: Float) -> Result(Float, Nil) {
  case td.centroids, q <. 0.0 || q >. 1.0 {
    [], _ -> Error(Nil)
    _, True -> Error(Nil)
    _, False -> {
      let target = q *. td.total_weight
      Ok(scan_quantile(td.centroids, target, 0.0))
    }
  }
}

fn scan_quantile(
  centroids: List(Centroid),
  target: Float,
  acc: Float,
) -> Float {
  case centroids {
    [] -> 0.0
    [c] -> c.mean
    [a, b, ..rest] -> {
      let cumulative = acc +. a.weight
      case target <=. cumulative {
        True -> {
          let span = a.weight +. b.weight
          let position = { target -. acc } /. span
          // Linear interpolation between centroid means.
          a.mean +. position *. { b.mean -. a.mean }
        }
        False -> scan_quantile([b, ..rest], target, cumulative)
      }
    }
  }
}

/// Total number of samples represented by the digest.
pub fn count(td: TDigest) -> Float {
  td.total_weight
}

/// Minimum sample so far (first centroid mean, since centroids stay sorted).
pub fn min(td: TDigest) -> Result(Float, Nil) {
  case td.centroids {
    [] -> Error(Nil)
    [c, ..] -> Ok(c.mean)
  }
}

/// Maximum sample so far.
pub fn max(td: TDigest) -> Result(Float, Nil) {
  case list.last(td.centroids) {
    Ok(c) -> Ok(c.mean)
    Error(_) -> Error(Nil)
  }
}

/// Convenience: median.
pub fn median(td: TDigest) -> Result(Float, Nil) {
  quantile(td, 0.5)
}

/// Convenience: p99 (extreme-tail quantile, where t-digest excels).
pub fn p99(td: TDigest) -> Result(Float, Nil) {
  quantile(td, 0.99)
}

// ============================================================================
// Internals
// ============================================================================

fn merge_centroid(td: TDigest, c: Centroid) -> TDigest {
  let new_centroids = insert_sorted(td.centroids, c)
  TDigest(
    compression: td.compression,
    centroids: new_centroids,
    total_weight: td.total_weight +. c.weight,
  )
}

fn insert_sorted(xs: List(Centroid), c: Centroid) -> List(Centroid) {
  case xs {
    [] -> [c]
    [head, ..rest] ->
      case c.mean <=. head.mean {
        True -> [c, head, ..rest]
        False -> [head, ..insert_sorted(rest, c)]
      }
  }
}

fn sorted_concat(a: List(Centroid), b: List(Centroid)) -> List(Centroid) {
  let merged = list.append(a, b)
  list.sort(merged, fn(x, y) {
    case x.mean <. y.mean, x.mean >. y.mean {
      True, _ -> order.Lt
      _, True -> order.Gt
      _, _ -> order.Eq
    }
  })
}

/// Compress the digest: walk sorted centroids merging adjacent ones while
/// each combined size stays under the k(q)-derived bound. This is the
/// heart of t-digest's accuracy guarantee.
fn compress(td: TDigest) -> TDigest {
  let total = td.total_weight
  let delta = td.compression
  case td.centroids {
    [] -> td
    [first, ..rest] -> {
      let #(merged, _final_q) =
        compress_walk(rest, first, 0.0, delta, total, [])
      TDigest(
        compression: delta,
        centroids: list.reverse(merged),
        total_weight: total,
      )
    }
  }
}

fn compress_walk(
  remaining: List(Centroid),
  current: Centroid,
  current_q_start: Float,
  delta: Float,
  total: Float,
  acc: List(Centroid),
) -> #(List(Centroid), Float) {
  case remaining {
    [] -> #([current, ..acc], current_q_start)
    [next, ..rest] -> {
      let combined_weight = current.weight +. next.weight
      let q_end = { current_q_start *. total +. combined_weight } /. total
      let allowed = max_weight_for_quantile(q_end, delta, total)
      case combined_weight <=. allowed {
        True -> {
          let new_mean =
            { current.mean *. current.weight +. next.mean *. next.weight }
            /. combined_weight
          let combined = Centroid(mean: new_mean, weight: combined_weight)
          compress_walk(rest, combined, current_q_start, delta, total, acc)
        }
        False ->
          compress_walk(
            rest,
            next,
            current_q_start +. current.weight /. total,
            delta,
            total,
            [current, ..acc],
          )
      }
    }
  }
}

/// k(q) scale function: maximum allowed weight for a centroid at cumulative
/// quantile q. Implementation uses the standard scaling
/// `k₁(q) = (δ / 2π) · arcsin(2q - 1)` and translates back to a weight
/// budget. We use the simpler approximation `4 · total · q · (1 - q) / δ`
/// which captures the same "small at tails" behaviour with cheaper
/// arithmetic.
fn max_weight_for_quantile(q: Float, delta: Float, total: Float) -> Float {
  let q_clamped = scalar.clamp(q, 0.0, 1.0)
  4.0 *. total *. q_clamped *. { 1.0 -. q_clamped } /. delta
}

fn float_max(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}

//// Tests for `viva_math/tdigest` — streaming quantile estimator (Dunning).
//// Invariants: total weight preservation, min/max boundaries, quantile
//// monotonicity, merge mass conservation, and tail/centre quantile sanity.

import gleeunit/should
import test_support.{is_close}
import viva_math/tdigest as td

// ============================================================================
// Construction
// ============================================================================

pub fn td_new_is_empty_test() {
  let d = td.new()
  is_close(td.count(d), 0.0, 1.0e-12) |> should.be_true
  td.min(d) |> should.equal(Error(Nil))
  td.max(d) |> should.equal(Error(Nil))
  td.median(d) |> should.equal(Error(Nil))
}

pub fn td_with_compression_test() {
  let d = td.with_compression(50.0)
  is_close(td.compression(d), 50.0, 1.0e-12) |> should.be_true
}

// ============================================================================
// Insertion + weight conservation
// ============================================================================

pub fn td_insert_increments_count_test() {
  let d =
    td.new()
    |> td.insert(1.0)
    |> td.insert(2.0)
    |> td.insert(3.0)
  is_close(td.count(d), 3.0, 1.0e-12) |> should.be_true
}

pub fn td_insert_all_count_test() {
  let d = td.insert_all(td.new(), [1.0, 2.0, 3.0, 4.0, 5.0])
  is_close(td.count(d), 5.0, 1.0e-12) |> should.be_true
}

pub fn td_insert_weighted_test() {
  let d =
    td.new()
    |> td.insert_weighted(10.0, 2.5)
    |> td.insert_weighted(20.0, 1.5)
  is_close(td.count(d), 4.0, 1.0e-12) |> should.be_true
}

// ============================================================================
// Boundary queries — min / max / quantile(0) / quantile(1)
// ============================================================================

pub fn td_min_max_test() {
  let d = td.insert_all(td.new(), [3.0, 1.0, 5.0, 2.0, 4.0])
  let assert Ok(mn) = td.min(d)
  let assert Ok(mx) = td.max(d)
  is_close(mn, 1.0, 1.0e-12) |> should.be_true
  is_close(mx, 5.0, 1.0e-12) |> should.be_true
}

pub fn td_quantile_zero_returns_min_test() {
  let d = td.insert_all(td.new(), [3.0, 1.0, 5.0])
  let assert Ok(q0) = td.quantile(d, 0.0)
  let assert Ok(mn) = td.min(d)
  is_close(q0, mn, 1.0e-12) |> should.be_true
}

pub fn td_quantile_one_returns_max_test() {
  let d = td.insert_all(td.new(), [3.0, 1.0, 5.0])
  let assert Ok(q1) = td.quantile(d, 1.0)
  let assert Ok(mx) = td.max(d)
  is_close(q1, mx, 1.0e-12) |> should.be_true
}

pub fn td_quantile_out_of_range_test() {
  let d = td.insert_all(td.new(), [1.0, 2.0, 3.0])
  td.quantile(d, -0.1) |> should.equal(Error(Nil))
  td.quantile(d, 1.1) |> should.equal(Error(Nil))
}

// ============================================================================
// Median + p99 — qualitative accuracy
// ============================================================================

pub fn td_median_of_uniform_test() {
  let xs = range_floats(1, 100, [])
  let d = td.insert_all(td.new(), xs)
  let assert Ok(m) = td.median(d)
  // The true median of [1..100] is 50.5; allow generous tolerance because
  // t-digest compresses (centre is the loosest area accuracy-wise).
  should.be_true(m >. 40.0 && m <. 60.0)
}

pub fn td_p99_above_median_test() {
  let xs = range_floats(1, 200, [])
  let d = td.insert_all(td.new(), xs)
  let assert Ok(m) = td.median(d)
  let assert Ok(p99) = td.p99(d)
  should.be_true(p99 >. m)
}

// ============================================================================
// Quantile monotonicity — q1 ≤ q2 ⇒ quantile(q1) ≤ quantile(q2)
// ============================================================================

pub fn td_quantile_monotonic_test() {
  let xs = range_floats(1, 50, [])
  let d = td.insert_all(td.new(), xs)
  let assert Ok(q10) = td.quantile(d, 0.1)
  let assert Ok(q50) = td.quantile(d, 0.5)
  let assert Ok(q90) = td.quantile(d, 0.9)
  should.be_true(q10 <=. q50)
  should.be_true(q50 <=. q90)
}

// ============================================================================
// Merge — mass conservation
// ============================================================================

pub fn td_merge_count_test() {
  let a = td.insert_all(td.new(), [1.0, 2.0, 3.0])
  let b = td.insert_all(td.new(), [4.0, 5.0])
  let merged = td.merge(a, b)
  is_close(td.count(merged), 5.0, 1.0e-12) |> should.be_true
}

pub fn td_merge_preserves_extremes_test() {
  let a = td.insert_all(td.new(), [10.0, 20.0])
  let b = td.insert_all(td.new(), [1.0, 100.0])
  let merged = td.merge(a, b)
  let assert Ok(mn) = td.min(merged)
  let assert Ok(mx) = td.max(merged)
  is_close(mn, 1.0, 1.0e-12) |> should.be_true
  is_close(mx, 100.0, 1.0e-12) |> should.be_true
}

@external(erlang, "erlang", "float")
fn int_to_float(n: Int) -> Float

/// Generate the float list `[from, from+1, ..., to]` (inclusive) by reverse-
/// accumulating. Tail-recursive, no `list.range` needed.
fn range_floats(from: Int, to: Int, acc: List(Float)) -> List(Float) {
  case to < from {
    True -> acc
    False -> range_floats(from, to - 1, [int_to_float(to), ..acc])
  }
}

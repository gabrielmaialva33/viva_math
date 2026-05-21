//// Direct coverage for the 1.2.103 compatibility pack — functions added to
//// close the migration gap left by dropping `gleam_community_maths` in
//// 1.2.101. Each test asserts a closed-form identity at machine precision
//// where possible.

import gleeunit/should
import test_support.{is_close, is_close_list, machine, tight}
import viva_math/precision
import viva_math/statistics
import viva_math/vecn

// ============================================================================
// vecn — distances + norms
// ============================================================================

pub fn vecn_euclidean_distance_pythagorean_test() {
  // (3, 4) is the Pythagorean triple; distance to origin = 5 exactly.
  let assert Ok(d) = vecn.euclidean_distance([3.0, 4.0], [0.0, 0.0])
  is_close(d, 5.0, machine) |> should.be_true
}

pub fn vecn_euclidean_distance_self_zero_test() {
  let assert Ok(d) = vecn.euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
  is_close(d, 0.0, machine) |> should.be_true
}

pub fn vecn_euclidean_distance_size_mismatch_test() {
  vecn.euclidean_distance([1.0, 2.0], [1.0])
  |> should.equal(Error(Nil))
}

pub fn vecn_manhattan_distance_test() {
  // |1-0| + |2-0| + |3-0| = 6
  let assert Ok(d) = vecn.manhattan_distance([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
  is_close(d, 6.0, machine) |> should.be_true
}

pub fn vecn_manhattan_distance_size_mismatch_test() {
  vecn.manhattan_distance([1.0], [1.0, 2.0])
  |> should.equal(Error(Nil))
}

pub fn vecn_cosine_similarity_self_one_test() {
  // sim(v, v) = 1 for any non-zero v.
  let v = [1.0, 2.0, 3.0]
  let assert Ok(s) = vecn.cosine_similarity(v, v)
  is_close(s, 1.0, tight) |> should.be_true
}

pub fn vecn_cosine_similarity_orthogonal_test() {
  // sim((1,0), (0,1)) = 0.
  let assert Ok(s) = vecn.cosine_similarity([1.0, 0.0], [0.0, 1.0])
  is_close(s, 0.0, machine) |> should.be_true
}

pub fn vecn_cosine_similarity_zero_vec_error_test() {
  vecn.cosine_similarity([0.0, 0.0], [1.0, 1.0])
  |> should.equal(Error(Nil))
  vecn.cosine_similarity([1.0, 1.0], [0.0, 0.0])
  |> should.equal(Error(Nil))
  vecn.cosine_similarity([0.0, 0.0], [0.0, 0.0])
  |> should.equal(Error(Nil))
}

pub fn vecn_lp_norm_l2_matches_length_test() {
  // L₂ norm = Euclidean length. ||(3, 4)||₂ = 5.
  let assert Ok(n) = vecn.lp_norm([3.0, 4.0], 2.0)
  is_close(n, 5.0, tight) |> should.be_true
}

pub fn vecn_lp_norm_l1_matches_manhattan_test() {
  // L₁ norm of (1, 1, 1) is 3.
  let assert Ok(n) = vecn.lp_norm([1.0, 1.0, 1.0], 1.0)
  is_close(n, 3.0, tight) |> should.be_true
}

pub fn vecn_lp_norm_l3_test() {
  // ||(2, 2)||₃ = (2³ + 2³)^(1/3) = 16^(1/3) ≈ 2.5198421
  let assert Ok(n) = vecn.lp_norm([2.0, 2.0], 3.0)
  is_close(n, 2.519_842_099_789_746, tight) |> should.be_true
}

pub fn vecn_lp_norm_empty_test() {
  let assert Ok(n) = vecn.lp_norm([], 2.0)
  is_close(n, 0.0, machine) |> should.be_true
}

// Domain guard — p ≤ 0 would cross-target diverge (badarith on Erlang,
// Infinity on JavaScript), so we reject upfront.
pub fn vecn_lp_norm_p_zero_rejects_test() {
  vecn.lp_norm([1.0, 2.0], 0.0) |> should.equal(Error(Nil))
}

pub fn vecn_lp_norm_p_negative_rejects_test() {
  vecn.lp_norm([1.0, 2.0], -1.0) |> should.equal(Error(Nil))
}

// ============================================================================
// statistics — linspace / logspace / cumulative reductions
// ============================================================================

pub fn statistics_linear_space_endpoint_true_test() {
  // 11 points from 0 to 10 inclusive: [0, 1, 2, …, 10].
  is_close_list(
    statistics.linear_space(0.0, 10.0, 11, True),
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    tight,
  )
  |> should.be_true
}

pub fn statistics_linear_space_endpoint_false_test() {
  // 10 points from 0 (exclusive end at 10): [0, 1, 2, …, 9].
  is_close_list(
    statistics.linear_space(0.0, 10.0, 10, False),
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    tight,
  )
  |> should.be_true
}

pub fn statistics_linear_space_singleton_test() {
  statistics.linear_space(7.5, 99.0, 1, True)
  |> should.equal([7.5])
  statistics.linear_space(7.5, 99.0, 1, False)
  |> should.equal([7.5])
}

pub fn statistics_linear_space_empty_test() {
  statistics.linear_space(0.0, 1.0, 0, True) |> should.equal([])
  statistics.linear_space(0.0, 1.0, -3, True) |> should.equal([])
}

pub fn statistics_linear_space_degenerate_test() {
  // start == stop → constant list.
  is_close_list(
    statistics.linear_space(2.0, 2.0, 3, True),
    [2.0, 2.0, 2.0],
    machine,
  )
  |> should.be_true
}

pub fn statistics_logarithmic_space_powers_of_ten_test() {
  // base 10, exponents 0..3 → [1, 10, 100, 1000].
  is_close_list(
    statistics.logarithmic_space(0.0, 3.0, 4, True, 10.0),
    [1.0, 10.0, 100.0, 1000.0],
    tight,
  )
  |> should.be_true
}

pub fn statistics_cumulative_sum_basic_test() {
  statistics.cumulative_sum([1.0, 2.0, 3.0, 4.0])
  |> is_close_list([1.0, 3.0, 6.0, 10.0], machine)
  |> should.be_true
}

pub fn statistics_cumulative_sum_empty_test() {
  statistics.cumulative_sum([]) |> should.equal([])
}

pub fn statistics_cumulative_product_basic_test() {
  // [1, 2, 3, 4] → [1, 2, 6, 24]
  statistics.cumulative_product([1.0, 2.0, 3.0, 4.0])
  |> is_close_list([1.0, 2.0, 6.0, 24.0], machine)
  |> should.be_true
}

pub fn statistics_cumulative_product_empty_test() {
  statistics.cumulative_product([]) |> should.equal([])
}

// ============================================================================
// precision — NumPy-style closeness predicates
// ============================================================================

pub fn precision_is_close_within_rtol_test() {
  // |1.0 − 1.000001| = 1e-6 ≤ 1e-5 · 1.0
  precision.is_close(1.0, 1.000_001, 1.0e-5, 0.0)
  |> should.be_true
}

pub fn precision_is_close_within_atol_test() {
  // |1.0 − 1.04| = 0.04 ≤ 0.05 + 0·|1.04|
  precision.is_close(1.0, 1.04, 0.0, 0.05) |> should.be_true
}

pub fn precision_is_close_outside_both_test() {
  precision.is_close(1.0, 1.1, 0.0, 0.05) |> should.be_false
}

pub fn precision_is_close_exact_test() {
  // rtol=0, atol=0 → only exact bit-equality passes. The next IEEE-754
  // double after 1.0 is `1.0 + 2^-52`, which is representably distinct
  // (1.000_000_000_000_000_2 collapses to that value).
  precision.is_close(1.0, 1.0, 0.0, 0.0) |> should.be_true
  precision.is_close(1.0, 1.000_000_000_000_000_2, 0.0, 0.0) |> should.be_false
}

pub fn precision_all_close_vacuous_test() {
  // Empty list is vacuously close, matching NumPy.
  precision.all_close([], 1.0e-5, 1.0e-8) |> should.be_true
}

pub fn precision_all_close_homogeneous_pass_test() {
  precision.all_close([#(1.0, 1.0), #(2.0, 2.0), #(3.0, 3.0)], 0.0, 0.0)
  |> should.be_true
}

pub fn precision_all_close_one_fails_test() {
  precision.all_close([#(1.0, 1.0), #(2.0, 5.0)], 1.0e-5, 1.0e-8)
  |> should.be_false
}

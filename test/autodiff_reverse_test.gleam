//// Tests for `viva_math/autodiff_reverse` — reverse-mode AD via an
//// append-only computation tape. Each test builds a tape, runs `backward`,
//// and asserts the gradient against the closed-form derivative.

import gleeunit/should
import test_support.{is_close}
import viva_math/autodiff as fwd
import viva_math/autodiff_reverse as rev
import viva_math/scalar

/// Helper: run `build` on a fresh tape with one scalar input at `x`,
/// backprop, return `∂output/∂x`.
fn d_of(x: Float, build: fn(rev.Tape, rev.NodeId) -> #(rev.NodeId, rev.Tape)) {
  let assert [g] =
    rev.gradients([x], fn(tape, ids) {
      let assert [a] = ids
      build(tape, a)
    })
  g
}

// ============================================================================
// Single-input identities
// ============================================================================

// d/dx (x) = 1
pub fn rev_identity_derivative_test() {
  let g = d_of(7.0, fn(t, a) { #(a, t) })
  is_close(g, 1.0, 1.0e-12) |> should.be_true
}

// d/dx (x²) = 2x   →  at x=3 → 6
pub fn rev_square_derivative_test() {
  let g = d_of(3.0, fn(t, a) { rev.mul(t, a, a) })
  is_close(g, 6.0, 1.0e-12) |> should.be_true
}

// d/dx (x³) = 3x²  →  at x=2 → 12
pub fn rev_cube_derivative_test() {
  let g =
    d_of(2.0, fn(t, a) {
      let #(sq, t1) = rev.mul(t, a, a)
      rev.mul(t1, sq, a)
    })
  is_close(g, 12.0, 1.0e-12) |> should.be_true
}

// d/dx exp(x) = exp(x)
pub fn rev_exp_derivative_test() {
  let g = d_of(1.0, rev.exp)
  is_close(g, scalar.exp(1.0), 1.0e-12) |> should.be_true
}

// d/dx ln(x) = 1/x  →  at x=4 → 0.25
pub fn rev_ln_derivative_test() {
  let g = d_of(4.0, rev.ln)
  is_close(g, 0.25, 1.0e-12) |> should.be_true
}

// d/dx x^4 = 4·x³  →  at x=2 → 32
pub fn rev_pow_derivative_test() {
  let g = d_of(2.0, fn(t, a) { rev.pow(t, a, 4.0) })
  is_close(g, 32.0, 1.0e-10) |> should.be_true
}

// d/dx tanh(x) = 1 − tanh²(x)
pub fn rev_tanh_derivative_test() {
  let x = 0.7
  let g = d_of(x, rev.tanh)
  let t = scalar.tanh(x)
  is_close(g, 1.0 -. t *. t, 1.0e-12) |> should.be_true
}

// d/dx σ(x) = σ(x)·(1−σ(x))
pub fn rev_sigmoid_derivative_test() {
  let x = 0.3
  let g = d_of(x, rev.sigmoid)
  let s = scalar.sigmoid(x)
  is_close(g, s *. { 1.0 -. s }, 1.0e-12) |> should.be_true
}

// ============================================================================
// Cross-mode consistency: forward AD === reverse AD on scalar→scalar
// ============================================================================

// d/dx (x · sin(x))
pub fn rev_matches_forward_x_sin_x_test() {
  let x = 1.2
  let g_rev =
    d_of(x, fn(t, a) {
      let #(s, t1) = rev.sin(t, a)
      rev.mul(t1, a, s)
    })
  let g_fwd = fwd.grad(fn(d) { fwd.mul(d, fwd.sin(d)) }, x)
  is_close(g_rev, g_fwd, 1.0e-12) |> should.be_true
}

// ============================================================================
// Multiple inputs — `gradients/2` aligns the output with the input list
// ============================================================================

// f(x, y) = x²·y  →  ∂f/∂x = 2xy, ∂f/∂y = x²
// At (x, y) = (3, 4): gradient = [24, 9]
pub fn rev_gradients_two_inputs_test() {
  let grads =
    rev.gradients([3.0, 4.0], fn(tape, ids) {
      let assert [x, y] = ids
      let #(x_sq, t1) = rev.mul(tape, x, x)
      rev.mul(t1, x_sq, y)
    })
  case grads {
    [gx, gy] -> {
      is_close(gx, 24.0, 1.0e-12) |> should.be_true
      is_close(gy, 9.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

// f(x, y) = x + y  →  ∂f/∂x = 1, ∂f/∂y = 1
pub fn rev_gradients_sum_test() {
  let grads =
    rev.gradients([2.0, 5.0], fn(tape, ids) {
      let assert [x, y] = ids
      rev.add(tape, x, y)
    })
  case grads {
    [gx, gy] -> {
      is_close(gx, 1.0, 1.0e-12) |> should.be_true
      is_close(gy, 1.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

// d/dx (x/c) = 1/c at c=4 → 0.25  (and d/dc (x/c) = -x/c² piece not tested here)
pub fn rev_div_derivative_test() {
  let g =
    d_of(8.0, fn(t, a) {
      let #(c, t1) = rev.input(t, 4.0)
      rev.div(t1, a, c)
    })
  is_close(g, 0.25, 1.0e-12) |> should.be_true
}

// d/dx (−x) = −1
pub fn rev_neg_derivative_test() {
  let g = d_of(3.0, rev.neg)
  is_close(g, -1.0, 1.0e-12) |> should.be_true
}

// d/dx (s · x) = s
pub fn rev_scale_derivative_test() {
  let g = d_of(1.0, fn(t, a) { rev.scale(t, a, 7.5) })
  is_close(g, 7.5, 1.0e-12) |> should.be_true
}

// d/dx sin(x) = cos(x)
pub fn rev_sin_derivative_test() {
  let g = d_of(0.5, rev.sin)
  is_close(g, scalar.cos(0.5), 1.0e-12) |> should.be_true
}

// d/dx cos(x) = -sin(x)
pub fn rev_cos_derivative_test() {
  let g = d_of(0.5, rev.cos)
  is_close(g, 0.0 -. scalar.sin(0.5), 1.0e-12) |> should.be_true
}

// d/dx sub(x, c) = 1
pub fn rev_sub_derivative_test() {
  let g =
    d_of(5.0, fn(t, a) {
      let #(c, t1) = rev.input(t, 2.0)
      rev.sub(t1, a, c)
    })
  is_close(g, 1.0, 1.0e-12) |> should.be_true
}

// `value(tape, id)` returns the forward value stored at a tape node.
pub fn rev_value_extracts_forward_value_test() {
  let #(out, tape) = {
    let t0 = rev.empty_tape()
    let #(a, t1) = rev.input(t0, 3.0)
    rev.mul(t1, a, a)
  }
  is_close(rev.value(tape, out), 9.0, 1.0e-12) |> should.be_true
}

// **Key reverse-AD invariant**: when an intermediate node is reused across
// the computation graph, its gradient must be the **sum** of contributions
// from every path that uses it. Concretely: for `f(x) = x · x`, the
// gradient of the output w.r.t. `x` is `2x` because `x` appears twice as a
// factor; this would be wrong if the reverse pass only walked one path.
pub fn rev_reused_node_accumulates_gradient_test() {
  let g = d_of(4.0, fn(t, a) { rev.mul(t, a, a) })
  is_close(g, 8.0, 1.0e-12) |> should.be_true
}

// More demanding accumulation: `f(x) = (x + x) · x = 2x²`, gradient = 4x.
pub fn rev_reused_node_via_add_then_mul_test() {
  let g =
    d_of(3.0, fn(t, a) {
      let #(two_a, t1) = rev.add(t, a, a)
      rev.mul(t1, two_a, a)
    })
  is_close(g, 12.0, 1.0e-12) |> should.be_true
}

//// Tests for `viva_math/autodiff` — forward-mode automatic differentiation
//// via dual numbers. Each test asserts a closed-form derivative identity.

import gleeunit/should
import test_support.{is_close}
import viva_math/autodiff as ad
import viva_math/scalar

// ============================================================================
// Construction + identity laws
// ============================================================================

pub fn ad_var_has_unit_tangent_test() {
  let d = ad.var(3.0)
  is_close(d.value, 3.0, 1.0e-12) |> should.be_true
  is_close(d.tangent, 1.0, 1.0e-12) |> should.be_true
}

pub fn ad_constant_has_zero_tangent_test() {
  let d = ad.constant(7.5)
  is_close(d.value, 7.5, 1.0e-12) |> should.be_true
  is_close(d.tangent, 0.0, 1.0e-12) |> should.be_true
}

// ============================================================================
// Arithmetic — chain rule on elementary ops
// ============================================================================

// d/dx (x + c) = 1
pub fn ad_add_constant_test() {
  let g = ad.grad(fn(x) { ad.add_scalar(x, 5.0) }, 2.0)
  is_close(g, 1.0, 1.0e-12) |> should.be_true
}

// d/dx (x²) = 2x  →  at x=3 → 6
pub fn ad_square_derivative_test() {
  let g = ad.grad(fn(x) { ad.mul(x, x) }, 3.0)
  is_close(g, 6.0, 1.0e-12) |> should.be_true
}

// d/dx (x³) = 3x²  →  at x=2 → 12
pub fn ad_cube_derivative_test() {
  let g = ad.grad(fn(x) { ad.mul(ad.mul(x, x), x) }, 2.0)
  is_close(g, 12.0, 1.0e-12) |> should.be_true
}

// d/dx (1/x) = -1/x²  →  at x=2 → -0.25
pub fn ad_reciprocal_derivative_test() {
  let g = ad.grad(fn(x) { ad.div(ad.constant(1.0), x) }, 2.0)
  is_close(g, -0.25, 1.0e-12) |> should.be_true
}

// d/dx (-x) = -1
pub fn ad_neg_derivative_test() {
  let g = ad.grad(fn(x) { ad.neg(x) }, 4.0)
  is_close(g, -1.0, 1.0e-12) |> should.be_true
}

// d/dx (s · x) = s
pub fn ad_scale_derivative_test() {
  let g = ad.grad(fn(x) { ad.scale(x, 3.5) }, 1.0)
  is_close(g, 3.5, 1.0e-12) |> should.be_true
}

// ============================================================================
// Transcendentals — analytic derivatives
// ============================================================================

// d/dx exp(x) = exp(x)
pub fn ad_exp_derivative_test() {
  let g = ad.grad(ad.exp, 1.0)
  is_close(g, scalar.exp(1.0), 1.0e-12) |> should.be_true
}

// d/dx ln(x) = 1/x  →  at x=4 → 0.25
pub fn ad_ln_derivative_test() {
  let g = ad.grad(ad.ln, 4.0)
  is_close(g, 0.25, 1.0e-12) |> should.be_true
}

// d/dx sqrt(x) = 1 / (2·√x)  →  at x=9 → 1/6 ≈ 0.166667
pub fn ad_sqrt_derivative_test() {
  let g = ad.grad(ad.sqrt, 9.0)
  is_close(g, 1.0 /. 6.0, 1.0e-12) |> should.be_true
}

// d/dx x^n = n·x^(n−1)  →  pow(x, 4) at x=2 → 32
pub fn ad_pow_derivative_test() {
  let g = ad.grad(fn(x) { ad.pow(x, 4.0) }, 2.0)
  is_close(g, 32.0, 1.0e-10) |> should.be_true
}

// d/dx sin(x) = cos(x)
pub fn ad_sin_derivative_test() {
  let g = ad.grad(ad.sin, 0.5)
  is_close(g, scalar.cos(0.5), 1.0e-12) |> should.be_true
}

// d/dx cos(x) = -sin(x)
pub fn ad_cos_derivative_test() {
  let g = ad.grad(ad.cos, 0.5)
  is_close(g, 0.0 -. scalar.sin(0.5), 1.0e-12) |> should.be_true
}

// d/dx tanh(x) = 1 − tanh²(x)
pub fn ad_tanh_derivative_test() {
  let x = 0.7
  let g = ad.grad(ad.tanh, x)
  let t = scalar.tanh(x)
  is_close(g, 1.0 -. t *. t, 1.0e-12) |> should.be_true
}

// d/dx σ(x) = σ(x)·(1−σ(x))
pub fn ad_sigmoid_derivative_test() {
  let x = 0.3
  let g = ad.grad(ad.sigmoid, x)
  let s = scalar.sigmoid(x)
  is_close(g, s *. { 1.0 -. s }, 1.0e-12) |> should.be_true
}

// d/dx ReLU(x): 1 for x > 0, 0 for x < 0.
pub fn ad_relu_derivative_test() {
  let pos = ad.grad(ad.relu, 1.5)
  let neg = ad.grad(ad.relu, -1.5)
  is_close(pos, 1.0, 1.0e-12) |> should.be_true
  is_close(neg, 0.0, 1.0e-12) |> should.be_true
}

// ============================================================================
// Composition (chain rule end-to-end)
// ============================================================================

// d/dx ( (x² + 1)·sin(x) ) at x=π
//   = 2x·sin(x) + (x²+1)·cos(x)
//   = 2π·0 + (π²+1)·(-1)
//   = -(π²+1)
pub fn ad_composition_test() {
  let f = fn(x) {
    let x2_plus_1 = ad.add_scalar(ad.mul(x, x), 1.0)
    ad.mul(x2_plus_1, ad.sin(x))
  }
  let pi = 3.141_592_653_589_793
  let g = ad.grad(f, pi)
  let expected = 0.0 -. { pi *. pi +. 1.0 }
  is_close(g, expected, 1.0e-9) |> should.be_true
}

// `value_and_grad` returns both value and derivative consistent with `grad`.
pub fn ad_value_and_grad_consistency_test() {
  let f = fn(x) { ad.exp(x) }
  let #(v, g) = ad.value_and_grad(f, 1.5)
  is_close(v, scalar.exp(1.5), 1.0e-12) |> should.be_true
  is_close(g, scalar.exp(1.5), 1.0e-12) |> should.be_true
}

// `lift1` — generic chain rule wrapper. `d/dx square(x) = 2x` at x=4 → 8.
pub fn ad_lift1_square_test() {
  let square = fn(d) { ad.lift1(d, fn(v) { v *. v }, fn(v) { 2.0 *. v }) }
  let g = ad.grad(square, 4.0)
  is_close(g, 8.0, 1.0e-12) |> should.be_true
}

// `add` carries both values and tangents component-wise.
pub fn ad_add_value_and_tangent_test() {
  let a = ad.Dual(value: 2.0, tangent: 3.0)
  let b = ad.Dual(value: 5.0, tangent: 7.0)
  let s = ad.add(a, b)
  is_close(s.value, 7.0, 1.0e-12) |> should.be_true
  is_close(s.tangent, 10.0, 1.0e-12) |> should.be_true
}

// `sub` is `add(a, neg(b))`.
pub fn ad_sub_value_and_tangent_test() {
  let a = ad.Dual(value: 2.0, tangent: 3.0)
  let b = ad.Dual(value: 5.0, tangent: 7.0)
  let s = ad.sub(a, b)
  is_close(s.value, -3.0, 1.0e-12) |> should.be_true
  is_close(s.tangent, -4.0, 1.0e-12) |> should.be_true
}

// GELU at 0: GELU(x) = x·Φ(x), so d/dx GELU(0) = Φ(0) + 0·φ(0) = 0.5 exactly.
// Tight tolerance — Φ(0) = 0.5 is a closed-form value with no rounding.
pub fn ad_gelu_derivative_at_zero_test() {
  let g = ad.grad(ad.gelu, 0.0)
  is_close(g, 0.5, 1.0e-12) |> should.be_true
}

// ============================================================================
// Dual3 — forward AD on ℝ³ (multivariate gradient)
// ============================================================================

// ∇(x² + y² + z²) = (2x, 2y, 2z) at (1, 2, 3) → (2, 4, 6).
pub fn ad_gradient3_quadratic_test() {
  let f = fn(x, y, z) {
    ad.add3(ad.add3(ad.mul3(x, x), ad.mul3(y, y)), ad.mul3(z, z))
  }
  let #(gx, gy, gz) = ad.gradient3(f, 1.0, 2.0, 3.0)
  is_close(gx, 2.0, 1.0e-12) |> should.be_true
  is_close(gy, 4.0, 1.0e-12) |> should.be_true
  is_close(gz, 6.0, 1.0e-12) |> should.be_true
}

// ∇(x·y·z) = (yz, xz, xy) at (2, 3, 4) → (12, 8, 6).
pub fn ad_gradient3_product_test() {
  let f = fn(x, y, z) { ad.mul3(ad.mul3(x, y), z) }
  let #(gx, gy, gz) = ad.gradient3(f, 2.0, 3.0, 4.0)
  is_close(gx, 12.0, 1.0e-12) |> should.be_true
  is_close(gy, 8.0, 1.0e-12) |> should.be_true
  is_close(gz, 6.0, 1.0e-12) |> should.be_true
}

// ∇exp(x + y + z) = (exp(s), exp(s), exp(s)) where s = x+y+z.
pub fn ad_exp3_partials_test() {
  let f = fn(x, y, z) { ad.exp3(ad.add3(ad.add3(x, y), z)) }
  let #(gx, gy, gz) = ad.gradient3(f, 0.0, 0.5, -0.5)
  // s = 0, exp(0) = 1
  is_close(gx, 1.0, 1.0e-12) |> should.be_true
  is_close(gy, 1.0, 1.0e-12) |> should.be_true
  is_close(gz, 1.0, 1.0e-12) |> should.be_true
}

// `jacobian` returns gradients **per input** (sweeping the unit ε through
// each input independently — the transpose of the canonical Jᵢⱼ = ∂fᵢ/∂xⱼ).
// For `f(x, y) = [x + y, 2x − y]`, the i-th row is `[∂f₀/∂xᵢ, ∂f₁/∂xᵢ]`:
//   row 0 (sweep x): [∂(x+y)/∂x, ∂(2x−y)/∂x] = [1, 2]
//   row 1 (sweep y): [∂(x+y)/∂y, ∂(2x−y)/∂y] = [1, -1]
pub fn ad_jacobian_linear_map_test() {
  let f = fn(inputs) {
    let assert [x, y] = inputs
    [ad.add(x, y), ad.sub(ad.scale(x, 2.0), y)]
  }
  let j = ad.jacobian(f, [3.0, 5.0])
  case j {
    [[a, b], [c, d]] -> {
      is_close(a, 1.0, 1.0e-12) |> should.be_true
      is_close(b, 2.0, 1.0e-12) |> should.be_true
      is_close(c, 1.0, 1.0e-12) |> should.be_true
      is_close(d, -1.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

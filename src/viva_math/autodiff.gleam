//// Forward-mode automatic differentiation via dual numbers.
////
//// A dual number `a + b·ε` (where `ε² = 0`) carries both a value and an
//// infinitesimal derivative. Applying `f` to a `Dual(x, 1)` yields
//// `Dual(f(x), f'(x))` — **exact** gradients without symbolic
//// manipulation or finite-difference truncation error.
////
//// ## When to use
////
//// - You need an exact gradient at a point.
//// - The function is composed from arithmetic ops + `exp/log/sin/cos/...`.
//// - Cost of evaluating `f` once with duals ≈ 2-3× the regular cost.
////
//// For Vec3 gradients (gradient of a scalar field over PAD space), use
//// `Dual3` which carries three partials in parallel.
////
//// ## Example
////
//// ```gleam
//// import viva_math/autodiff as ad
////
//// // f(x) = sin(x²)
//// let x = ad.var(2.0)
//// let result = ad.sin(ad.mul(x, x))
//// // result.value ≈ sin(4) ≈ -0.756
//// // result.tangent = 2x · cos(x²) ≈ -2.614
//// ```
////
//// ## References
////
//// - Pearlmutter & Siskind (2008) "Reverse-Mode AD in a Functional Framework"
//// - Wengert (1964) "A simple automatic derivative evaluation program"
//// - JAX `jax.jvp`, Zygote.jl, Stan's Math library

import gleam/list
import viva_math/scalar

// ============================================================================
// Scalar dual numbers
// ============================================================================

/// A dual number `value + tangent·ε`.
///
/// `value` carries the function evaluation; `tangent` carries the directional
/// derivative w.r.t. the independent variable.
pub type Dual {
  Dual(value: Float, tangent: Float)
}

/// A dual number representing an independent variable.
/// `var(x)` ↔ `x + 1·ε` so that ∂x/∂x = 1.
pub fn var(value: Float) -> Dual {
  Dual(value: value, tangent: 1.0)
}

/// Constant dual: tangent = 0.
pub fn constant(value: Float) -> Dual {
  Dual(value: value, tangent: 0.0)
}

/// Lift any unary function `f` together with its derivative `f'` to a dual.
pub fn lift1(d: Dual, f: fn(Float) -> Float, df: fn(Float) -> Float) -> Dual {
  Dual(value: f(d.value), tangent: df(d.value) *. d.tangent)
}

// Basic arithmetic --------------------------------------------------

/// `(a + a'ε) + (b + b'ε) = (a+b) + (a'+b')ε`.
pub fn add(a: Dual, b: Dual) -> Dual {
  Dual(a.value +. b.value, a.tangent +. b.tangent)
}

/// `(a + a'ε) − (b + b'ε) = (a−b) + (a'−b')ε`.
pub fn sub(a: Dual, b: Dual) -> Dual {
  Dual(a.value -. b.value, a.tangent -. b.tangent)
}

/// `(a + a'ε)(b + b'ε) = ab + (a'b + ab')ε` — Leibniz product rule on duals.
pub fn mul(a: Dual, b: Dual) -> Dual {
  Dual(a.value *. b.value, a.tangent *. b.value +. a.value *. b.tangent)
}

/// `(a + a'ε) / (b + b'ε) = a/b + ((a'b − ab')/b²)ε` — quotient rule.
pub fn div(a: Dual, b: Dual) -> Dual {
  let denom = b.value *. b.value
  Dual(
    a.value /. b.value,
    { a.tangent *. b.value -. a.value *. b.tangent } /. denom,
  )
}

/// Negation: `−(a + a'ε) = (−a) + (−a')ε`.
pub fn neg(a: Dual) -> Dual {
  Dual(0.0 -. a.value, 0.0 -. a.tangent)
}

/// Multiplication by a scalar constant (no tangent on the scalar).
pub fn scale(a: Dual, s: Float) -> Dual {
  Dual(a.value *. s, a.tangent *. s)
}

/// Add a constant — the tangent is unchanged (`d/dx (x + c) = 1`).
pub fn add_scalar(a: Dual, s: Float) -> Dual {
  Dual(a.value +. s, a.tangent)
}

// Transcendentals ---------------------------------------------------

/// `d/dx exp(x) = exp(x)`.
pub fn exp(a: Dual) -> Dual {
  let v = scalar.exp(a.value)
  Dual(v, v *. a.tangent)
}

/// `d/dx ln(x) = 1/x`. Caller must ensure `a.value > 0`.
pub fn ln(a: Dual) -> Dual {
  Dual(scalar.ln(a.value), a.tangent /. a.value)
}

/// `d/dx √x = 1 / (2·√x)`. Caller must ensure `a.value > 0`.
pub fn sqrt(a: Dual) -> Dual {
  let v = scalar.sqrt(a.value)
  Dual(v, a.tangent /. { 2.0 *. v })
}

/// `d/dx xⁿ = n·xⁿ⁻¹` (real exponent `n`).
pub fn pow(a: Dual, n: Float) -> Dual {
  let v = scalar.pow(a.value, n)
  Dual(v, n *. scalar.pow(a.value, n -. 1.0) *. a.tangent)
}

/// `d/dx sin(x) = cos(x)`.
pub fn sin(a: Dual) -> Dual {
  Dual(sine(a.value), cosine(a.value) *. a.tangent)
}

/// `d/dx cos(x) = −sin(x)`.
pub fn cos(a: Dual) -> Dual {
  Dual(cosine(a.value), 0.0 -. sine(a.value) *. a.tangent)
}

/// `d/dx tanh(x) = 1 − tanh²(x)`.
pub fn tanh(a: Dual) -> Dual {
  let t = scalar.tanh(a.value)
  Dual(t, { 1.0 -. t *. t } *. a.tangent)
}

/// `d/dx σ(x) = σ(x)·(1 − σ(x))`.
pub fn sigmoid(a: Dual) -> Dual {
  let s = scalar.sigmoid(a.value)
  Dual(s, s *. { 1.0 -. s } *. a.tangent)
}

/// GELU activation. `GELU(x) = x·Φ(x)` where `Φ` is the standard normal CDF.
/// `d/dx GELU(x) = Φ(x) + x·φ(x)`, with `φ` the standard normal PDF.
pub fn gelu(a: Dual) -> Dual {
  let phi = 0.5 *. { 1.0 +. scalar.erf(a.value *. 0.7071067811865475) }
  let phi_pdf = 0.3989422804014327 *. scalar.exp(-0.5 *. a.value *. a.value)
  let g = a.value *. phi
  Dual(g, { phi +. a.value *. phi_pdf } *. a.tangent)
}

/// ReLU activation. Derivative is `1` for `x > 0` and `0` for `x ≤ 0`
/// (subgradient choice at the kink).
pub fn relu(a: Dual) -> Dual {
  case a.value >. 0.0 {
    True -> a
    False -> Dual(0.0, 0.0)
  }
}

// ============================================================================
// Gradient extractor
// ============================================================================

/// Compute f'(x) at a point by lifting `x` to a dual.
pub fn grad(f: fn(Dual) -> Dual, x: Float) -> Float {
  f(var(x)).tangent
}

/// Compute both f(x) and f'(x) in a single pass.
pub fn value_and_grad(f: fn(Dual) -> Dual, x: Float) -> #(Float, Float) {
  let r = f(var(x))
  #(r.value, r.tangent)
}

// ============================================================================
// 3-D dual: parallel partials in PAD space
// ============================================================================

/// A dual number carrying three parallel partials. Use this to compute the
/// gradient ∇f of a scalar field f: ℝ³ → ℝ at a point in one evaluation.
/// 3-dimensional dual carrying a value and three partial derivatives
/// `(∂/∂x, ∂/∂y, ∂/∂z)` — used by `gradient3` to evaluate ∇f in one pass.
pub type Dual3 {
  Dual3(value: Float, partial_x: Float, partial_y: Float, partial_z: Float)
}

/// Treat `value` as a constant: all partials = 0.
pub fn const3(value: Float) -> Dual3 {
  Dual3(value: value, partial_x: 0.0, partial_y: 0.0, partial_z: 0.0)
}

/// The independent variable x: ∂x/∂x = 1, others = 0.
pub fn var3_x(value: Float) -> Dual3 {
  Dual3(value: value, partial_x: 1.0, partial_y: 0.0, partial_z: 0.0)
}

/// The independent variable y: ∂y/∂y = 1, others = 0.
pub fn var3_y(value: Float) -> Dual3 {
  Dual3(value: value, partial_x: 0.0, partial_y: 1.0, partial_z: 0.0)
}

/// The independent variable z: ∂z/∂z = 1, others = 0.
pub fn var3_z(value: Float) -> Dual3 {
  Dual3(value: value, partial_x: 0.0, partial_y: 0.0, partial_z: 1.0)
}

/// Component-wise sum lifted to `Dual3`.
pub fn add3(a: Dual3, b: Dual3) -> Dual3 {
  Dual3(
    a.value +. b.value,
    a.partial_x +. b.partial_x,
    a.partial_y +. b.partial_y,
    a.partial_z +. b.partial_z,
  )
}

/// Product on `Dual3` — Leibniz rule applied component-wise to each partial.
pub fn mul3(a: Dual3, b: Dual3) -> Dual3 {
  Dual3(
    a.value *. b.value,
    a.partial_x *. b.value +. a.value *. b.partial_x,
    a.partial_y *. b.value +. a.value *. b.partial_y,
    a.partial_z *. b.value +. a.value *. b.partial_z,
  )
}

/// `d/dx exp(x) = exp(x)` lifted to `Dual3`.
pub fn exp3(a: Dual3) -> Dual3 {
  let v = scalar.exp(a.value)
  Dual3(v, v *. a.partial_x, v *. a.partial_y, v *. a.partial_z)
}

/// Gradient ∇f at point (x, y, z).
pub fn gradient3(
  f: fn(Dual3, Dual3, Dual3) -> Dual3,
  x: Float,
  y: Float,
  z: Float,
) -> #(Float, Float, Float) {
  let r = f(var3_x(x), var3_y(y), var3_z(z))
  #(r.partial_x, r.partial_y, r.partial_z)
}

/// Generic n-d Jacobian via column-by-column forward AD. Given a function
/// `f: ℝⁿ → ℝᵐ` represented as `fn(Dual, ..., Dual) -> List(Dual)`, returns
/// each row of the Jacobian by sweeping the unit tangent through each input.
///
/// This is the canonical forward-mode strategy: O(n) evaluations of `f` for
/// a full Jacobian, optimal when n ≤ m.
pub fn jacobian(
  f: fn(List(Dual)) -> List(Dual),
  point: List(Float),
) -> List(List(Float)) {
  let n = list.length(point)
  range_int(0, n - 1)
  |> list.map(fn(i) {
    let inputs = build_input(point, i, 0, [])
    let outputs = f(inputs)
    list.map(outputs, fn(d) { d.tangent })
  })
}

fn build_input(
  point: List(Float),
  active: Int,
  i: Int,
  acc: List(Dual),
) -> List(Dual) {
  case point {
    [] -> list.reverse(acc)
    [x, ..rest] -> {
      let d = case i == active {
        True -> Dual(value: x, tangent: 1.0)
        False -> Dual(value: x, tangent: 0.0)
      }
      build_input(rest, active, i + 1, [d, ..acc])
    }
  }
}

@external(erlang, "math", "sin")
@external(javascript, "../viva_math_random_ffi.mjs", "sin")
fn sine(x: Float) -> Float

@external(erlang, "math", "cos")
@external(javascript, "../viva_math_random_ffi.mjs", "cos")
fn cosine(x: Float) -> Float

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}

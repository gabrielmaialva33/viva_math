//// Tests for `viva_math/special` — Lanczos `gamma`/`lgamma`, `digamma`,
//// `beta`, `lbeta`, `factorial`, `binomial`. Anchored on well-known integer
//// identities and tabulated values.

import gleeunit/should
import test_support.{is_close}
import viva_math/special

// ============================================================================
// Gamma — integer values are factorials: Γ(n) = (n-1)!
// ============================================================================

pub fn gamma_one_test() {
  is_close(special.gamma(1.0), 1.0, 1.0e-9) |> should.be_true
}

pub fn gamma_two_test() {
  // Γ(2) = 1! = 1
  is_close(special.gamma(2.0), 1.0, 1.0e-9) |> should.be_true
}

pub fn gamma_three_test() {
  // Γ(3) = 2! = 2
  is_close(special.gamma(3.0), 2.0, 1.0e-9) |> should.be_true
}

pub fn gamma_four_test() {
  // Γ(4) = 3! = 6
  is_close(special.gamma(4.0), 6.0, 1.0e-9) |> should.be_true
}

pub fn gamma_five_test() {
  // Γ(5) = 4! = 24
  is_close(special.gamma(5.0), 24.0, 1.0e-8) |> should.be_true
}

// Γ(1/2) = √π ≈ 1.7724538509055159
pub fn gamma_half_test() {
  is_close(special.gamma(0.5), 1.772_453_850_905_516, 1.0e-9)
  |> should.be_true
}

// Γ(3/2) = ½·√π ≈ 0.8862269254527580
pub fn gamma_three_halves_test() {
  is_close(special.gamma(1.5), 0.886_226_925_452_758, 1.0e-9)
  |> should.be_true
}

// ============================================================================
// lgamma — log-domain, agrees with ln(gamma) for moderate arguments
// ============================================================================

pub fn lgamma_one_test() {
  // ln(Γ(1)) = ln(1) = 0
  is_close(special.lgamma(1.0), 0.0, 1.0e-9) |> should.be_true
}

pub fn lgamma_two_test() {
  // ln(Γ(2)) = ln(1) = 0
  is_close(special.lgamma(2.0), 0.0, 1.0e-9) |> should.be_true
}

pub fn lgamma_five_test() {
  // ln(Γ(5)) = ln(24) ≈ 3.1780538303479458
  is_close(special.lgamma(5.0), 3.178_053_830_347_946, 1.0e-9)
  |> should.be_true
}

// ============================================================================
// Factorial — exact for small n, validated against Γ(n+1)
// ============================================================================

pub fn factorial_zero_test() {
  let assert Ok(f) = special.factorial(0)
  is_close(f, 1.0, 1.0e-12) |> should.be_true
}

pub fn factorial_one_test() {
  let assert Ok(f) = special.factorial(1)
  is_close(f, 1.0, 1.0e-12) |> should.be_true
}

pub fn factorial_five_test() {
  let assert Ok(f) = special.factorial(5)
  is_close(f, 120.0, 1.0e-12) |> should.be_true
}

pub fn factorial_ten_test() {
  let assert Ok(f) = special.factorial(10)
  is_close(f, 3_628_800.0, 1.0e-9) |> should.be_true
}

pub fn factorial_negative_test() {
  special.factorial(-1) |> should.equal(Error(Nil))
}

// ============================================================================
// Binomial — tabulated identities
// ============================================================================

pub fn binomial_basic_test() {
  let assert Ok(b) = special.binomial(5, 2)
  is_close(b, 10.0, 1.0e-9) |> should.be_true
}

pub fn binomial_edges_test() {
  let assert Ok(zero) = special.binomial(5, 0)
  let assert Ok(full) = special.binomial(5, 5)
  is_close(zero, 1.0, 1.0e-9) |> should.be_true
  is_close(full, 1.0, 1.0e-9) |> should.be_true
}

pub fn binomial_pascal_test() {
  // C(6, 3) = 20
  let assert Ok(c) = special.binomial(6, 3)
  is_close(c, 20.0, 1.0e-9) |> should.be_true
}

pub fn binomial_negative_k_test() {
  special.binomial(5, -1) |> should.equal(Error(Nil))
}

// C(n, k) = 0 for k > n (no combinations).
pub fn binomial_k_above_n_test() {
  let assert Ok(c) = special.binomial(5, 6)
  is_close(c, 0.0, 1.0e-12) |> should.be_true
}

// ============================================================================
// Beta — B(x, y) = Γ(x)·Γ(y) / Γ(x+y)
// ============================================================================

// B(1, 1) = Γ(1)·Γ(1)/Γ(2) = 1
pub fn beta_one_one_test() {
  is_close(special.beta(1.0, 1.0), 1.0, 1.0e-9) |> should.be_true
}

// B(2, 3) = 1·2/24 = 1/12 ≈ 0.08333…
pub fn beta_two_three_test() {
  is_close(special.beta(2.0, 3.0), 1.0 /. 12.0, 1.0e-9) |> should.be_true
}

// B(x, y) = B(y, x) (symmetry).
pub fn beta_symmetric_test() {
  let a = special.beta(2.5, 4.0)
  let b = special.beta(4.0, 2.5)
  is_close(a, b, 1.0e-12) |> should.be_true
}

// ============================================================================
// Digamma — ψ(1) = -γ (Euler-Mascheroni), ψ(2) = 1 - γ
// ============================================================================

pub fn digamma_one_test() {
  // ψ(1) = -γ ≈ -0.5772156649015329
  is_close(special.digamma(1.0), -0.577_215_664_901_533, 1.0e-6)
  |> should.be_true
}

pub fn digamma_two_test() {
  // ψ(2) = 1 - γ ≈ 0.4227843350984671
  is_close(special.digamma(2.0), 0.422_784_335_098_467, 1.0e-6)
  |> should.be_true
}

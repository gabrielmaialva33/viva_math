//// Special mathematical functions.
////
//// Implementations focus on functions that complement `gleam_community_maths`
//// and are needed by `viva_math/distributions` for Gamma/Beta/Dirichlet
//// distributions, by `viva_math/free_energy` for KL divergences of
//// exponential-family distributions, and by `viva_math/entropy` for the
//// differential entropy of those families.
////
//// ## Algorithms
////
//// - `gamma`  → Lanczos approximation (g=7, 9 coefficients). Accurate to
////              ~15 decimal digits across the entire positive reals.
//// - `lgamma` → Logarithmic form of Lanczos; preferred numerically because
////              the regular gamma overflows past Γ(171.6) ≈ 10³⁰⁸.
//// - `digamma`→ ψ(x) via the asymptotic series for large x with the
////              recurrence ψ(x) = ψ(x+1) - 1/x for small x.
//// - `beta`   → exp(lgamma(x) + lgamma(y) - lgamma(x+y)).
//// - `factorial` → Stirling-style via gamma for large n, exact for n ≤ 20.
////
//// ## References
////
//// - Lanczos (1964) "A precision approximation of the gamma function"
//// - Press, Teukolsky et al. (1992) "Numerical Recipes" §6.1
//// - GSL `gsl_sf_lngamma`

import viva_math/constants
import viva_math/scalar

// ============================================================================
// Lanczos gamma approximation
// ============================================================================

/// Logarithm of the gamma function `ln Γ(x)`. Defined for `x > 0`.
///
/// Uses the Lanczos approximation with g=7, accurate to ~15 digits.
pub fn lgamma(x: Float) -> Float {
  case x <=. 0.0 {
    // Pole at non-positive integers; we return +∞ by convention.
    True -> constants.max_float
    False -> {
      case x <. 0.5 {
        True -> {
          // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
          //   ln Γ(x) = ln π - ln|sin πx| - ln Γ(1-x)
          let sin_pi_x = sine(constants.pi *. x)
          let abs_sin = case sin_pi_x <. 0.0 {
            True -> 0.0 -. sin_pi_x
            False -> sin_pi_x
          }
          scalar.ln(constants.pi) -. scalar.ln(abs_sin) -. lgamma(1.0 -. x)
        }
        False -> lanczos_lgamma(x)
      }
    }
  }
}

fn lanczos_lgamma(x: Float) -> Float {
  // Stirling-Lanczos main branch for x ≥ 0.5
  let xm1 = x -. 1.0
  let t = xm1 +. 7.5
  let series = lanczos_series(xm1)
  scalar.ln(2.5066282746310002 *. series) +. { xm1 +. 0.5 } *. scalar.ln(t) -. t
}

fn lanczos_series(xm1: Float) -> Float {
  // Coefficients from Numerical Recipes / GSL (g=7).
  0.99999999999980993
  +. 676.5203681218851
  /. { xm1 +. 1.0 }
  -. 1259.1392167224028
  /. { xm1 +. 2.0 }
  +. 771.32342877765313
  /. { xm1 +. 3.0 }
  -. 176.61502916214059
  /. { xm1 +. 4.0 }
  +. 12.507343278686905
  /. { xm1 +. 5.0 }
  -. 0.13857109526572012
  /. { xm1 +. 6.0 }
  +. 9.9843695780195716e-6
  /. { xm1 +. 7.0 }
  +. 1.5056327351493116e-7
  /. { xm1 +. 8.0 }
}

/// Gamma function Γ(x). Overflows past x ≈ 171.6 — prefer `lgamma`.
pub fn gamma(x: Float) -> Float {
  scalar.exp(lgamma(x))
}

/// Beta function B(x, y) = Γ(x)·Γ(y)/Γ(x+y). Always computed via `lgamma`
/// to avoid overflow.
pub fn beta(x: Float, y: Float) -> Float {
  scalar.exp(lgamma(x) +. lgamma(y) -. lgamma(x +. y))
}

/// Logarithm of the beta function — `ln B(x, y)`.
pub fn lbeta(x: Float, y: Float) -> Float {
  lgamma(x) +. lgamma(y) -. lgamma(x +. y)
}

// ============================================================================
// Digamma — ψ(x) = d/dx ln Γ(x)
// ============================================================================

/// Digamma function ψ(x). Defined for `x > 0`.
///
/// Combines the recurrence ψ(x) = ψ(x+1) - 1/x to push x ≥ 6, then evaluates
/// the asymptotic series ψ(x) ≈ ln x - 1/(2x) - Σ B₂ₖ / (2k·x^(2k)).
pub fn digamma(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0 -. constants.max_float
    False -> digamma_shifted(x, 0.0)
  }
}

fn digamma_shifted(x: Float, accum: Float) -> Float {
  case x <. 6.0 {
    True -> digamma_shifted(x +. 1.0, accum -. 1.0 /. x)
    False -> accum +. digamma_asymptotic(x)
  }
}

fn digamma_asymptotic(x: Float) -> Float {
  // ψ(x) ≈ ln x - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
  let inv_x = 1.0 /. x
  let inv_x2 = inv_x *. inv_x
  scalar.ln(x)
  -. 0.5
  *. inv_x
  -. inv_x2
  *. {
    0.0833333333333333333
    -. inv_x2
    *. {
      0.00833333333333333333
      -. inv_x2
      *. { 0.00396825396825396825 -. inv_x2 *. 0.00416666666666666666 }
    }
  }
}

// ============================================================================
// Factorial
// ============================================================================

/// Factorial n! for non-negative integer n.
///
/// Exact for n ≤ 20 (fits in i64). For larger n falls back to Γ(n+1) via
/// Lanczos, with attendant ~15-digit relative accuracy.
pub fn factorial(n: Int) -> Result(Float, Nil) {
  case n < 0 {
    True -> Error(Nil)
    False ->
      case n <= 20 {
        True -> Ok(int_to_float(factorial_exact(n, 1)))
        False -> Ok(gamma(int_to_float(n) +. 1.0))
      }
  }
}

fn factorial_exact(n: Int, acc: Int) -> Int {
  case n <= 1 {
    True -> acc
    False -> factorial_exact(n - 1, acc * n)
  }
}

/// Binomial coefficient C(n, k) = n! / (k!·(n-k)!) via lgamma for stability.
pub fn binomial(n: Int, k: Int) -> Result(Float, Nil) {
  case n < 0, k < 0, k > n {
    True, _, _ -> Error(Nil)
    _, True, _ -> Error(Nil)
    _, _, True -> Ok(0.0)
    _, _, _ -> {
      let n_f = int_to_float(n)
      let k_f = int_to_float(k)
      Ok(scalar.exp(
        lgamma(n_f +. 1.0) -. lgamma(k_f +. 1.0) -. lgamma(n_f -. k_f +. 1.0),
      ))
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

@external(erlang, "erlang", "float")
fn int_to_float(n: Int) -> Float

@external(erlang, "math", "sin")
fn sine(x: Float) -> Float

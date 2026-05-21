//// Complex numbers `a + bi`.
////
//// Standard algebra plus a small set of transcendental functions sufficient
//// for FFT, eigenvalue work and signal processing.
////
//// All operations are pure and total — division by zero returns the zero
//// complex by convention (consistent with `gleam/float`).

import gleam/float
import viva_math/scalar

pub type Complex {
  Complex(re: Float, im: Float)
}

// ============================================================================
// Constants & constructors
// ============================================================================

/// 0 + 0i
pub fn zero() -> Complex {
  Complex(re: 0.0, im: 0.0)
}

/// 1 + 0i
pub fn one() -> Complex {
  Complex(re: 1.0, im: 0.0)
}

/// 0 + 1i (the imaginary unit)
pub fn i() -> Complex {
  Complex(re: 0.0, im: 1.0)
}

/// Build from a real number (im = 0).
pub fn real(x: Float) -> Complex {
  Complex(re: x, im: 0.0)
}

/// Build from magnitude `r` and phase `theta` (polar form).
pub fn from_polar(r: Float, theta: Float) -> Complex {
  Complex(re: r *. cosine(theta), im: r *. sine(theta))
}

// ============================================================================
// Algebra
// ============================================================================

pub fn add(a: Complex, b: Complex) -> Complex {
  Complex(a.re +. b.re, a.im +. b.im)
}

pub fn sub(a: Complex, b: Complex) -> Complex {
  Complex(a.re -. b.re, a.im -. b.im)
}

pub fn mul(a: Complex, b: Complex) -> Complex {
  Complex(a.re *. b.re -. a.im *. b.im, a.re *. b.im +. a.im *. b.re)
}

pub fn div(a: Complex, b: Complex) -> Complex {
  // a / b = a · conj(b) / |b|²
  let denom = b.re *. b.re +. b.im *. b.im
  case denom == 0.0 {
    True -> zero()
    False ->
      Complex(
        { a.re *. b.re +. a.im *. b.im } /. denom,
        { a.im *. b.re -. a.re *. b.im } /. denom,
      )
  }
}

pub fn neg(z: Complex) -> Complex {
  Complex(0.0 -. z.re, 0.0 -. z.im)
}

pub fn conjugate(z: Complex) -> Complex {
  Complex(z.re, 0.0 -. z.im)
}

pub fn scale(z: Complex, s: Float) -> Complex {
  Complex(z.re *. s, z.im *. s)
}

// ============================================================================
// Magnitude / phase
// ============================================================================

/// |z| = √(re² + im²). Uses `hypot` to avoid overflow.
pub fn magnitude(z: Complex) -> Float {
  scalar.hypot(z.re, z.im)
}

/// |z|² (cheaper than `magnitude` when comparing).
pub fn magnitude_squared(z: Complex) -> Float {
  z.re *. z.re +. z.im *. z.im
}

/// Phase angle in radians, range (-π, π].
pub fn phase(z: Complex) -> Float {
  atan2(z.im, z.re)
}

/// Polar decomposition: returns `(magnitude, phase)`.
pub fn to_polar(z: Complex) -> #(Float, Float) {
  #(magnitude(z), phase(z))
}

// ============================================================================
// Transcendental functions
// ============================================================================

/// exp(z) = e^a · (cos(b) + i·sin(b))
pub fn exp(z: Complex) -> Complex {
  let ea = scalar.exp(z.re)
  Complex(ea *. cosine(z.im), ea *. sine(z.im))
}

/// log(z) = ln|z| + i·arg(z). Branch cut along the negative real axis.
pub fn log(z: Complex) -> Complex {
  Complex(scalar.ln(magnitude(z)), phase(z))
}

/// Principal square root √z.
pub fn sqrt(z: Complex) -> Complex {
  let r = magnitude(z)
  let new_re = scalar.sqrt({ r +. z.re } /. 2.0)
  let sign = case z.im <. 0.0 {
    True -> -1.0
    False -> 1.0
  }
  let new_im = sign *. scalar.sqrt({ r -. z.re } /. 2.0)
  Complex(new_re, new_im)
}

/// z^n for integer n via repeated multiplication. Handles negative exponents.
pub fn pow_int(z: Complex, n: Int) -> Complex {
  case n {
    0 -> one()
    n if n < 0 -> div(one(), pow_int(z, 0 - n))
    _ -> pow_int_loop(z, n, one())
  }
}

fn pow_int_loop(base: Complex, n: Int, acc: Complex) -> Complex {
  case n {
    0 -> acc
    _ ->
      case n % 2 {
        0 -> pow_int_loop(mul(base, base), n / 2, acc)
        _ -> pow_int_loop(mul(base, base), n / 2, mul(acc, base))
      }
  }
}

/// Real exponent z^x via `exp(x · log(z))`.
pub fn pow(z: Complex, x: Float) -> Complex {
  exp(scale(log(z), x))
}

/// sin(z) = sin(a)·cosh(b) + i·cos(a)·sinh(b)
pub fn sin(z: Complex) -> Complex {
  Complex(sine(z.re) *. cosh(z.im), cosine(z.re) *. sinh(z.im))
}

/// cos(z) = cos(a)·cosh(b) - i·sin(a)·sinh(b)
pub fn cos(z: Complex) -> Complex {
  Complex(cosine(z.re) *. cosh(z.im), 0.0 -. sine(z.re) *. sinh(z.im))
}

/// tan(z) = sin(z) / cos(z)
pub fn tan(z: Complex) -> Complex {
  div(sin(z), cos(z))
}

// ============================================================================
// Utility
// ============================================================================

/// Approximate equality.
pub fn is_close(a: Complex, b: Complex, tol: Float) -> Bool {
  float.absolute_value(a.re -. b.re) <=. tol
  && float.absolute_value(a.im -. b.im) <=. tol
}

@external(erlang, "math", "sin")
fn sine(x: Float) -> Float

@external(erlang, "math", "cos")
fn cosine(x: Float) -> Float

@external(erlang, "math", "sinh")
fn sinh(x: Float) -> Float

@external(erlang, "math", "cosh")
fn cosh(x: Float) -> Float

@external(erlang, "math", "atan2")
fn atan2(y: Float, x: Float) -> Float

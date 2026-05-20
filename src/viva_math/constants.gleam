//// Mathematical constants.
////
//// High-precision constants frequently needed across scientific computing.
//// All values are IEEE-754 double precision (≈ 15-17 significant digits).
////
//// On the Erlang target these are plain `const` floats. The BEAM JIT inlines
//// them at call sites, so there is no runtime lookup cost. For tables that
//// must be shared across many processes consider `:persistent_term` directly
//// from the calling code.
////
//// ## Categories
////
//// - Circle: `pi`, `tau`, `half_pi`, `quarter_pi`, `inv_pi`
//// - Roots: `sqrt_2`, `sqrt_3`, `sqrt_pi`, `sqrt_2pi`, `sqrt_tau`
//// - Inverses: `inv_sqrt_2`, `inv_sqrt_pi`, `inv_sqrt_2pi`
//// - Logarithms: `e`, `ln_2`, `ln_10`, `log2_e`, `log10_e`
//// - Special: `phi` (golden ratio), `euler_gamma`, `catalan`
//// - Numeric limits: `epsilon`, `max_float`, `min_positive`

// ============================================================================
// Circle constants
// ============================================================================

/// π (ratio of circumference to diameter).
pub const pi = 3.141592653589793

/// τ = 2π (one full turn in radians).
pub const tau = 6.283185307179586

/// π / 2
pub const half_pi = 1.5707963267948966

/// π / 4
pub const quarter_pi = 0.7853981633974483

/// 1 / π
pub const inv_pi = 0.3183098861837907

/// 2 / π
pub const two_over_pi = 0.6366197723675814

// ============================================================================
// Roots
// ============================================================================

/// √2
pub const sqrt_2 = 1.4142135623730951

/// √3
pub const sqrt_3 = 1.7320508075688772

/// √π
pub const sqrt_pi = 1.7724538509055159

/// √(2π) - normalization constant for Gaussian PDF.
pub const sqrt_2pi = 2.5066282746310002

/// √τ = √(2π)
pub const sqrt_tau = 2.5066282746310002

// ============================================================================
// Inverse roots
// ============================================================================

/// 1 / √2 - useful for GELU exact form.
pub const inv_sqrt_2 = 0.7071067811865475

/// 1 / √π
pub const inv_sqrt_pi = 0.5641895835477563

/// 1 / √(2π) - Gaussian PDF coefficient.
pub const inv_sqrt_2pi = 0.3989422804014327

// ============================================================================
// Exponential & logarithm
// ============================================================================

/// e (Euler's number).
pub const e = 2.718281828459045

/// ln(2) - natural log of 2.
pub const ln_2 = 0.6931471805599453

/// ln(10) - natural log of 10.
pub const ln_10 = 2.302585092994046

/// log₂(e) = 1 / ln(2)
pub const log2_e = 1.4426950408889634

/// log₁₀(e) = 1 / ln(10)
pub const log10_e = 0.4342944819032518

// ============================================================================
// Special
// ============================================================================

/// φ (golden ratio): (1 + √5) / 2
pub const phi = 1.618033988749895

/// 1 / φ - reciprocal of golden ratio.
pub const inv_phi = 0.6180339887498948

/// Euler-Mascheroni constant γ.
pub const euler_gamma = 0.5772156649015329

/// Catalan's constant G.
pub const catalan = 0.915965594177219

/// Apéry's constant ζ(3).
pub const apery = 1.2020569031595942

// ============================================================================
// Floating-point limits (IEEE-754 double)
// ============================================================================

/// Machine epsilon for f64.
pub const epsilon = 2.220446049250313e-16

/// Maximum representable positive finite double.
pub const max_float = 1.7976931348623157e308

/// Minimum positive normal double.
pub const min_positive = 2.2250738585072014e-308

// ============================================================================
// Conversion factors
// ============================================================================

/// Degrees → radians multiplier (π / 180).
pub const deg_to_rad = 0.017453292519943295

/// Radians → degrees multiplier (180 / π).
pub const rad_to_deg = 57.29577951308232

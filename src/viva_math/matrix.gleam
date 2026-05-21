//// Small dense matrices (2×2, 3×3, 4×4) and a generic `MatN`.
////
//// **Scope**: geometry, dynamics, small-system linear algebra. For batched
//// or large-scale work (GEMM, BLAS, eigendecomposition) use `viva_tensor`,
//// which dispatches to MKL/CUDA.
////
//// ## Conventions
////
//// - Row-major storage in the generic `MatN` (rows × cols).
//// - Fixed-size matrices use named fields `m11..mNN`.
//// - All constructors clamp to safe values where appropriate.

import gleam/list
import viva_math/scalar
import viva_math/vector.{type Vec3, Vec3}

// ============================================================================
// 2×2
// ============================================================================

pub type Mat2 {
  Mat2(m11: Float, m12: Float, m21: Float, m22: Float)
}

pub fn mat2_zero() -> Mat2 {
  Mat2(0.0, 0.0, 0.0, 0.0)
}

pub fn mat2_identity() -> Mat2 {
  Mat2(1.0, 0.0, 0.0, 1.0)
}

pub fn mat2_add(a: Mat2, b: Mat2) -> Mat2 {
  Mat2(a.m11 +. b.m11, a.m12 +. b.m12, a.m21 +. b.m21, a.m22 +. b.m22)
}

pub fn mat2_scale(m: Mat2, s: Float) -> Mat2 {
  Mat2(m.m11 *. s, m.m12 *. s, m.m21 *. s, m.m22 *. s)
}

pub fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
  Mat2(
    m11: a.m11 *. b.m11 +. a.m12 *. b.m21,
    m12: a.m11 *. b.m12 +. a.m12 *. b.m22,
    m21: a.m21 *. b.m11 +. a.m22 *. b.m21,
    m22: a.m21 *. b.m12 +. a.m22 *. b.m22,
  )
}

pub fn mat2_transpose(m: Mat2) -> Mat2 {
  Mat2(m.m11, m.m21, m.m12, m.m22)
}

/// det(M) = m₁₁m₂₂ - m₁₂m₂₁.
pub fn mat2_determinant(m: Mat2) -> Float {
  m.m11 *. m.m22 -. m.m12 *. m.m21
}

/// Trace tr(M) = Σ mᵢᵢ.
pub fn mat2_trace(m: Mat2) -> Float {
  m.m11 +. m.m22
}

/// Inverse of a 2×2 matrix. Errors if singular.
pub fn mat2_inverse(m: Mat2) -> Result(Mat2, Nil) {
  let d = mat2_determinant(m)
  case d == 0.0 {
    True -> Error(Nil)
    False -> {
      let inv_d = 1.0 /. d
      Ok(Mat2(
        m.m22 *. inv_d,
        0.0 -. m.m12 *. inv_d,
        0.0 -. m.m21 *. inv_d,
        m.m11 *. inv_d,
      ))
    }
  }
}

/// 2-D rotation matrix by `theta` radians.
pub fn mat2_rotation(theta: Float) -> Mat2 {
  let c = cosine(theta)
  let s = sine(theta)
  Mat2(c, 0.0 -. s, s, c)
}

// ============================================================================
// 3×3
// ============================================================================

pub type Mat3 {
  Mat3(
    m11: Float,
    m12: Float,
    m13: Float,
    m21: Float,
    m22: Float,
    m23: Float,
    m31: Float,
    m32: Float,
    m33: Float,
  )
}

pub fn mat3_zero() -> Mat3 {
  Mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

pub fn mat3_identity() -> Mat3 {
  Mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
}

pub fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
  Mat3(
    a.m11 +. b.m11,
    a.m12 +. b.m12,
    a.m13 +. b.m13,
    a.m21 +. b.m21,
    a.m22 +. b.m22,
    a.m23 +. b.m23,
    a.m31 +. b.m31,
    a.m32 +. b.m32,
    a.m33 +. b.m33,
  )
}

pub fn mat3_scale(m: Mat3, s: Float) -> Mat3 {
  Mat3(
    m.m11 *. s,
    m.m12 *. s,
    m.m13 *. s,
    m.m21 *. s,
    m.m22 *. s,
    m.m23 *. s,
    m.m31 *. s,
    m.m32 *. s,
    m.m33 *. s,
  )
}

pub fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
  Mat3(
    m11: a.m11 *. b.m11 +. a.m12 *. b.m21 +. a.m13 *. b.m31,
    m12: a.m11 *. b.m12 +. a.m12 *. b.m22 +. a.m13 *. b.m32,
    m13: a.m11 *. b.m13 +. a.m12 *. b.m23 +. a.m13 *. b.m33,
    m21: a.m21 *. b.m11 +. a.m22 *. b.m21 +. a.m23 *. b.m31,
    m22: a.m21 *. b.m12 +. a.m22 *. b.m22 +. a.m23 *. b.m32,
    m23: a.m21 *. b.m13 +. a.m22 *. b.m23 +. a.m23 *. b.m33,
    m31: a.m31 *. b.m11 +. a.m32 *. b.m21 +. a.m33 *. b.m31,
    m32: a.m31 *. b.m12 +. a.m32 *. b.m22 +. a.m33 *. b.m32,
    m33: a.m31 *. b.m13 +. a.m32 *. b.m23 +. a.m33 *. b.m33,
  )
}

pub fn mat3_transpose(m: Mat3) -> Mat3 {
  Mat3(m.m11, m.m21, m.m31, m.m12, m.m22, m.m32, m.m13, m.m23, m.m33)
}

/// Determinant via cofactor expansion along the first row.
pub fn mat3_determinant(m: Mat3) -> Float {
  m.m11
  *. { m.m22 *. m.m33 -. m.m23 *. m.m32 }
  -. m.m12
  *. { m.m21 *. m.m33 -. m.m23 *. m.m31 }
  +. m.m13
  *. { m.m21 *. m.m32 -. m.m22 *. m.m31 }
}

pub fn mat3_trace(m: Mat3) -> Float {
  m.m11 +. m.m22 +. m.m33
}

/// Inverse of a 3×3 matrix via adjugate / determinant.
///
/// Errors when the matrix is singular or ill-conditioned: the threshold is
/// `|det| < ε · ‖M‖_F³` where ε ≈ machine epsilon. Pure `det == 0.0` is too
/// permissive — matrices with `det = 1e-20` would still pass and produce
/// catastrophic blow-up after division.
pub fn mat3_inverse(m: Mat3) -> Result(Mat3, Nil) {
  let det = mat3_determinant(m)
  let frob = mat3_frobenius(m)
  // Cube the Frobenius norm so the threshold scales like a determinant.
  let tolerance = 2.22e-16 *. frob *. frob *. frob
  let det_abs = case det <. 0.0 {
    True -> 0.0 -. det
    False -> det
  }
  case det_abs <=. tolerance {
    True -> Error(Nil)
    False -> {
      let inv_det = 1.0 /. det
      Ok(Mat3(
        m11: { m.m22 *. m.m33 -. m.m23 *. m.m32 } *. inv_det,
        m12: { m.m13 *. m.m32 -. m.m12 *. m.m33 } *. inv_det,
        m13: { m.m12 *. m.m23 -. m.m13 *. m.m22 } *. inv_det,
        m21: { m.m23 *. m.m31 -. m.m21 *. m.m33 } *. inv_det,
        m22: { m.m11 *. m.m33 -. m.m13 *. m.m31 } *. inv_det,
        m23: { m.m13 *. m.m21 -. m.m11 *. m.m23 } *. inv_det,
        m31: { m.m21 *. m.m32 -. m.m22 *. m.m31 } *. inv_det,
        m32: { m.m12 *. m.m31 -. m.m11 *. m.m32 } *. inv_det,
        m33: { m.m11 *. m.m22 -. m.m12 *. m.m21 } *. inv_det,
      ))
    }
  }
}

/// Matrix × Vec3 product.
pub fn mat3_mul_vec3(m: Mat3, v: Vec3) -> Vec3 {
  Vec3(
    m.m11 *. v.x +. m.m12 *. v.y +. m.m13 *. v.z,
    m.m21 *. v.x +. m.m22 *. v.y +. m.m23 *. v.z,
    m.m31 *. v.x +. m.m32 *. v.y +. m.m33 *. v.z,
  )
}

/// Rotation around the X axis.
pub fn mat3_rot_x(theta: Float) -> Mat3 {
  let c = cosine(theta)
  let s = sine(theta)
  Mat3(1.0, 0.0, 0.0, 0.0, c, 0.0 -. s, 0.0, s, c)
}

/// Rotation around the Y axis.
pub fn mat3_rot_y(theta: Float) -> Mat3 {
  let c = cosine(theta)
  let s = sine(theta)
  Mat3(c, 0.0, s, 0.0, 1.0, 0.0, 0.0 -. s, 0.0, c)
}

/// Rotation around the Z axis.
pub fn mat3_rot_z(theta: Float) -> Mat3 {
  let c = cosine(theta)
  let s = sine(theta)
  Mat3(c, 0.0 -. s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Frobenius norm √(Σ aᵢⱼ²) of a Mat3.
pub fn mat3_frobenius(m: Mat3) -> Float {
  scalar.sqrt(
    m.m11
    *. m.m11
    +. m.m12
    *. m.m12
    +. m.m13
    *. m.m13
    +. m.m21
    *. m.m21
    +. m.m22
    *. m.m22
    +. m.m23
    *. m.m23
    +. m.m31
    *. m.m31
    +. m.m32
    *. m.m32
    +. m.m33
    *. m.m33,
  )
}

// ============================================================================
// Eigenvalues (closed form)
// ============================================================================

/// Real eigenvalues of a 2×2 matrix via the characteristic quadratic.
///
/// λ² - tr(M)·λ + det(M) = 0  →  λ = (tr ± √(tr² - 4·det)) / 2.
///
/// Returns `Error` for matrices with non-real eigenvalues (negative
/// discriminant). For symmetric matrices the discriminant is always
/// non-negative, so this always succeeds.
pub fn mat2_eigenvalues(m: Mat2) -> Result(#(Float, Float), Nil) {
  let tr = mat2_trace(m)
  let det = mat2_determinant(m)
  let discriminant = tr *. tr -. 4.0 *. det
  case discriminant <. 0.0 {
    True -> Error(Nil)
    False -> {
      let s = scalar.sqrt(discriminant)
      Ok(#({ tr +. s } /. 2.0, { tr -. s } /. 2.0))
    }
  }
}

/// Real eigenvalues of a **symmetric** 3×3 matrix via Smith's trigonometric
/// closed form. The matrix must be (numerically) symmetric.
///
/// Algorithm: characteristic cubic det(M - λI) = 0 in the form
/// λ³ - tr·λ² + …  Then shift by p = tr/3 and solve the depressed cubic
/// using trigonometric substitution.
///
/// Reference: Smith (1961) "Eigenvalues of a Symmetric 3×3 Matrix".
pub fn mat3_symmetric_eigenvalues(
  m: Mat3,
) -> Result(#(Float, Float, Float), Nil) {
  let p1 = m.m12 *. m.m12 +. m.m13 *. m.m13 +. m.m23 *. m.m23

  case p1 == 0.0 {
    True -> {
      // Already diagonal: eigenvalues are on the diagonal.
      Ok(sort_three(m.m11, m.m22, m.m33))
    }
    False -> {
      let q = mat3_trace(m) /. 3.0
      let p2 =
        { m.m11 -. q }
        *. { m.m11 -. q }
        +. { m.m22 -. q }
        *. { m.m22 -. q }
        +. { m.m33 -. q }
        *. { m.m33 -. q }
        +. 2.0
        *. p1
      let p = scalar.sqrt(p2 /. 6.0)
      // B = (M - qI) / p
      let inv_p = 1.0 /. p
      let b =
        Mat3(
          m11: inv_p *. { m.m11 -. q },
          m12: inv_p *. m.m12,
          m13: inv_p *. m.m13,
          m21: inv_p *. m.m21,
          m22: inv_p *. { m.m22 -. q },
          m23: inv_p *. m.m23,
          m31: inv_p *. m.m31,
          m32: inv_p *. m.m32,
          m33: inv_p *. { m.m33 -. q },
        )
      let r = mat3_determinant(b) /. 2.0
      // Clamp r to [-1, 1] for acos.
      let r_clamped = case r {
        r if r >. 1.0 -> 1.0
        r if r <. -1.0 -> -1.0
        _ -> r
      }
      let phi = acos_safe(r_clamped) /. 3.0
      let two_pi_over_3 = 2.094_395_102_393_195
      let eig1 = q +. 2.0 *. p *. cosine(phi)
      let eig3 = q +. 2.0 *. p *. cosine(phi +. two_pi_over_3)
      let eig2 = 3.0 *. q -. eig1 -. eig3
      Ok(sort_three(eig1, eig2, eig3))
    }
  }
}

fn sort_three(a: Float, b: Float, c: Float) -> #(Float, Float, Float) {
  case a <=. b, b <=. c, a <=. c {
    True, True, _ -> #(a, b, c)
    True, False, True -> #(a, c, b)
    True, False, False -> #(c, a, b)
    False, True, True -> #(b, a, c)
    False, True, False -> #(b, c, a)
    _, _, _ -> #(c, b, a)
  }
}

@external(erlang, "math", "acos")
@external(javascript, "../viva_math_random_ffi.mjs", "acos")
fn acos_raw(x: Float) -> Float

fn acos_safe(x: Float) -> Float {
  acos_raw(x)
}

/// Diagonal matrix.
pub fn mat3_diagonal(a: Float, b: Float, c: Float) -> Mat3 {
  Mat3(a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c)
}

// ============================================================================
// 4×4 (compact - geometry / homogeneous transforms)
// ============================================================================

pub type Mat4 {
  Mat4(rows: List(List(Float)))
}

pub fn mat4_identity() -> Mat4 {
  Mat4(rows: [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ])
}

pub fn mat4_zero() -> Mat4 {
  Mat4(rows: list.repeat([0.0, 0.0, 0.0, 0.0], 4))
}

/// Translation homogeneous transform.
pub fn mat4_translation(tx: Float, ty: Float, tz: Float) -> Mat4 {
  Mat4(rows: [
    [1.0, 0.0, 0.0, tx],
    [0.0, 1.0, 0.0, ty],
    [0.0, 0.0, 1.0, tz],
    [0.0, 0.0, 0.0, 1.0],
  ])
}

/// Non-uniform scale.
pub fn mat4_scale(sx: Float, sy: Float, sz: Float) -> Mat4 {
  Mat4(rows: [
    [sx, 0.0, 0.0, 0.0],
    [0.0, sy, 0.0, 0.0],
    [0.0, 0.0, sz, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ])
}

// ============================================================================
// Generic MatN (row-major, dynamic size)
// ============================================================================

pub type MatN {
  MatN(rows: Int, cols: Int, data: List(List(Float)))
}

/// Build a MatN from a row-major list of rows.
///
/// Errors if rows is empty, has empty rows, or rows of inconsistent length.
pub fn matn_from_rows(rows: List(List(Float))) -> Result(MatN, Nil) {
  case rows {
    [] -> Error(Nil)
    [first, ..] -> {
      let n_cols = list.length(first)
      case n_cols == 0 {
        True -> Error(Nil)
        False -> {
          let consistent = list.all(rows, fn(r) { list.length(r) == n_cols })
          case consistent {
            False -> Error(Nil)
            True -> Ok(MatN(rows: list.length(rows), cols: n_cols, data: rows))
          }
        }
      }
    }
  }
}

/// Zero matrix of given shape.
pub fn matn_zeros(rows: Int, cols: Int) -> MatN {
  let row = list.repeat(0.0, cols)
  MatN(rows: rows, cols: cols, data: list.repeat(row, rows))
}

/// Identity matrix of size n.
pub fn matn_identity(n: Int) -> MatN {
  let data =
    range_int(0, n - 1)
    |> list.map(fn(i) {
      range_int(0, n - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  MatN(rows: n, cols: n, data: data)
}

/// Transpose of a MatN. O(rows · cols).
pub fn matn_transpose(m: MatN) -> MatN {
  let rows = transpose_lists(m.data)
  MatN(rows: m.cols, cols: m.rows, data: rows)
}

fn transpose_lists(rows: List(List(Float))) -> List(List(Float)) {
  case rows {
    [] -> []
    [first, ..] -> {
      let n = list.length(first)
      range_int(0, n - 1)
      |> list.map(fn(col) {
        list.map(rows, fn(row) {
          case list_at(row, col) {
            Ok(v) -> v
            Error(_) -> 0.0
          }
        })
      })
    }
  }
}

/// Matrix product A·B. Errors on shape mismatch.
pub fn matn_mul(a: MatN, b: MatN) -> Result(MatN, Nil) {
  case a.cols == b.rows {
    False -> Error(Nil)
    True -> {
      let bt = matn_transpose(b)
      let new_rows =
        list.map(a.data, fn(row_a) {
          list.map(bt.data, fn(row_b) { dot_lists(row_a, row_b) })
        })
      Ok(MatN(rows: a.rows, cols: b.cols, data: new_rows))
    }
  }
}

fn dot_lists(a: List(Float), b: List(Float)) -> Float {
  list.zip(a, b)
  |> list.fold(0.0, fn(acc, pair) { acc +. pair.0 *. pair.1 })
}

/// Add two matrices. Errors on shape mismatch.
pub fn matn_add(a: MatN, b: MatN) -> Result(MatN, Nil) {
  case a.rows == b.rows && a.cols == b.cols {
    False -> Error(Nil)
    True -> {
      let new_rows =
        list.zip(a.data, b.data)
        |> list.map(fn(pair) {
          list.zip(pair.0, pair.1)
          |> list.map(fn(p) { p.0 +. p.1 })
        })
      Ok(MatN(rows: a.rows, cols: a.cols, data: new_rows))
    }
  }
}

/// Scalar multiplication.
pub fn matn_scale(m: MatN, s: Float) -> MatN {
  let new_rows = list.map(m.data, fn(row) { list.map(row, fn(x) { x *. s }) })
  MatN(rows: m.rows, cols: m.cols, data: new_rows)
}

/// Trace (sum of diagonal). Defined for square matrices.
pub fn matn_trace(m: MatN) -> Result(Float, Nil) {
  case m.rows == m.cols {
    False -> Error(Nil)
    True -> Ok(trace_loop(m.data, 0, 0.0))
  }
}

fn trace_loop(rows: List(List(Float)), idx: Int, acc: Float) -> Float {
  case rows {
    [] -> acc
    [row, ..rest] -> {
      let v = case list_at(row, idx) {
        Ok(x) -> x
        Error(_) -> 0.0
      }
      trace_loop(rest, idx + 1, acc +. v)
    }
  }
}

/// Frobenius norm: √(Σ aᵢⱼ²).
pub fn matn_frobenius(m: MatN) -> Float {
  let sum_sq =
    list.fold(m.data, 0.0, fn(acc, row) {
      acc +. list.fold(row, 0.0, fn(s, x) { s +. x *. x })
    })
  scalar.sqrt(sum_sq)
}

/// Matrix × column-vector product. Returns the resulting vector as a list.
pub fn matn_mul_vec(m: MatN, v: List(Float)) -> Result(List(Float), Nil) {
  case m.cols == list.length(v) {
    False -> Error(Nil)
    True -> Ok(list.map(m.data, fn(row) { dot_lists(row, v) }))
  }
}

// ============================================================================
// Helpers
// ============================================================================

fn list_at(xs: List(Float), idx: Int) -> Result(Float, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at(rest, n - 1)
  }
}

@external(erlang, "math", "cos")
@external(javascript, "../viva_math_random_ffi.mjs", "cos")
fn cosine(x: Float) -> Float

@external(erlang, "math", "sin")
@external(javascript, "../viva_math_random_ffi.mjs", "sin")
fn sine(x: Float) -> Float

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}

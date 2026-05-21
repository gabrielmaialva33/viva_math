//// Dense matrix backed by `BitArray` (binary row-major).
////
//// Counterpart to `viva_math/matrix.MatN`. Where `MatN` stores rows as
//// nested `List(Float)`, `DenseMat` keeps a single contiguous binary
//// buffer with one IEEE-754 little-endian double per element. This:
////
//// - **Saves memory**: 8 bytes/element vs ~24 bytes/element for cons-cell
////   lists on 64-bit BEAM.
//// - **Speeds up indexed access**: O(1) random read of any cell via
////   `byte_offset = (row · cols + col) · 8`.
//// - **Avoids list reversal**: `transpose` and `matmul` rebuild a binary
////   instead of allocating thousands of cons cells.
////
//// ## Scope
////
//// Suitable for matrices up to a few thousand rows × cols. For tensor-scale
//// work (batched GEMM, GPU acceleration) defer to `viva_tensor`. This
//// module is the middle ground between the trivially-correct `MatN` and
//// the heavy `viva_tensor`.
////
//// ## Limitations
////
//// - Float-only (`Float` ↔ 64-bit IEEE 754 little-endian).
//// - Sequential algorithms (no parallelism, no SIMD).
//// - Conversion to/from `MatN` available for interop.

import gleam/list
import viva_math/scalar

/// Row-major dense matrix backed by a `BitArray` of 64-bit little-endian
/// floats (8 bytes per cell). Shape invariant: `byte_size(data) = rows·cols·8`.
///
/// Opaque to prevent direct construction with a `data` blob whose size does
/// not match the declared shape. Use `zeros`, `identity`, or `from_list` to
/// build a `DenseMat` and the `rows`, `cols`, `byte_size` accessors to query
/// its shape.
pub opaque type DenseMat {
  DenseMat(rows: Int, cols: Int, data: BitArray)
}

// ============================================================================
// Accessors
// ============================================================================

/// Number of rows in the matrix.
pub fn rows(m: DenseMat) -> Int {
  m.rows
}

/// Number of columns in the matrix.
pub fn cols(m: DenseMat) -> Int {
  m.cols
}

/// Shape as `#(rows, cols)`.
pub fn shape(m: DenseMat) -> #(Int, Int) {
  #(m.rows, m.cols)
}

// ============================================================================
// Construction
// ============================================================================

/// Zero matrix of given shape. Errors on non-positive dimensions.
pub fn zeros(rows: Int, cols: Int) -> Result(DenseMat, Nil) {
  case rows <= 0 || cols <= 0 {
    True -> Error(Nil)
    False -> {
      let zero_data = build_zeros(rows * cols, <<>>)
      Ok(DenseMat(rows: rows, cols: cols, data: zero_data))
    }
  }
}

fn build_zeros(n: Int, acc: BitArray) -> BitArray {
  case n <= 0 {
    True -> acc
    False -> build_zeros(n - 1, <<acc:bits, 0.0:float-little-size(64)>>)
  }
}

/// Identity matrix of size n.
pub fn identity(n: Int) -> Result(DenseMat, Nil) {
  case n <= 0 {
    True -> Error(Nil)
    False -> {
      let data = build_identity(n, 0, 0, <<>>)
      Ok(DenseMat(rows: n, cols: n, data: data))
    }
  }
}

fn build_identity(n: Int, i: Int, j: Int, acc: BitArray) -> BitArray {
  case i >= n {
    True -> acc
    False -> {
      let v = case i == j {
        True -> 1.0
        False -> 0.0
      }
      let next_acc = <<acc:bits, v:float-little-size(64)>>
      case j + 1 >= n {
        True -> build_identity(n, i + 1, 0, next_acc)
        False -> build_identity(n, i, j + 1, next_acc)
      }
    }
  }
}

/// Build from row-major list of values. Errors if shape doesn't match.
pub fn from_list(
  rows: Int,
  cols: Int,
  values: List(Float),
) -> Result(DenseMat, Nil) {
  case rows <= 0 || cols <= 0 || list.length(values) != rows * cols {
    True -> Error(Nil)
    False -> {
      let data = pack_floats(values, <<>>)
      Ok(DenseMat(rows: rows, cols: cols, data: data))
    }
  }
}

fn pack_floats(xs: List(Float), acc: BitArray) -> BitArray {
  case xs {
    [] -> acc
    [x, ..rest] -> pack_floats(rest, <<acc:bits, x:float-little-size(64)>>)
  }
}

// ============================================================================
// Accessors
// ============================================================================

/// Random-access read in O(1). Errors on out-of-bounds.
pub fn get(m: DenseMat, row: Int, col: Int) -> Result(Float, Nil) {
  case row < 0 || row >= m.rows || col < 0 || col >= m.cols {
    True -> Error(Nil)
    False -> {
      let index = row * m.cols + col
      let bit_offset = index * 64
      case m.data {
        <<_:size(bit_offset), value:float-little-size(64), _:bits>> -> Ok(value)
        _ -> Error(Nil)
      }
    }
  }
}

/// Get an entire row as a list.
pub fn row(m: DenseMat, row_idx: Int) -> Result(List(Float), Nil) {
  case row_idx < 0 || row_idx >= m.rows {
    True -> Error(Nil)
    False -> Ok(extract_row(m, row_idx, 0, []))
  }
}

fn extract_row(
  m: DenseMat,
  row_idx: Int,
  col_idx: Int,
  acc: List(Float),
) -> List(Float) {
  case col_idx >= m.cols {
    True -> list.reverse(acc)
    False ->
      case get(m, row_idx, col_idx) {
        Ok(v) -> extract_row(m, row_idx, col_idx + 1, [v, ..acc])
        Error(_) -> list.reverse(acc)
      }
  }
}

/// Convert to nested-list representation for interop with `MatN`.
pub fn to_rows(m: DenseMat) -> List(List(Float)) {
  build_rows(m, 0, [])
}

fn build_rows(
  m: DenseMat,
  i: Int,
  acc: List(List(Float)),
) -> List(List(Float)) {
  case i >= m.rows {
    True -> list.reverse(acc)
    False ->
      case row(m, i) {
        Ok(r) -> build_rows(m, i + 1, [r, ..acc])
        Error(_) -> list.reverse(acc)
      }
  }
}

/// Number of bytes consumed by the data buffer. Useful for benchmarking.
pub fn byte_size(m: DenseMat) -> Int {
  bit_size(m.data) / 8
}

@external(erlang, "erlang", "bit_size")
fn bit_size(b: BitArray) -> Int

// ============================================================================
// Element-wise operations
// ============================================================================

/// Element-wise add. Errors on shape mismatch.
pub fn add(a: DenseMat, b: DenseMat) -> Result(DenseMat, Nil) {
  case a.rows != b.rows || a.cols != b.cols {
    True -> Error(Nil)
    False -> Ok(zip_with(a, b, fn(x, y) { x +. y }))
  }
}

/// Element-wise subtract.
pub fn sub(a: DenseMat, b: DenseMat) -> Result(DenseMat, Nil) {
  case a.rows != b.rows || a.cols != b.cols {
    True -> Error(Nil)
    False -> Ok(zip_with(a, b, fn(x, y) { x -. y }))
  }
}

/// Element-wise Hadamard product.
pub fn hadamard(a: DenseMat, b: DenseMat) -> Result(DenseMat, Nil) {
  case a.rows != b.rows || a.cols != b.cols {
    True -> Error(Nil)
    False -> Ok(zip_with(a, b, fn(x, y) { x *. y }))
  }
}

/// Scalar multiplication.
pub fn scale(m: DenseMat, s: Float) -> DenseMat {
  let new_data = map_data(m.data, fn(x) { x *. s }, <<>>)
  DenseMat(rows: m.rows, cols: m.cols, data: new_data)
}

fn zip_with(
  a: DenseMat,
  b: DenseMat,
  f: fn(Float, Float) -> Float,
) -> DenseMat {
  let new_data = zip_data(a.data, b.data, f, <<>>)
  DenseMat(rows: a.rows, cols: a.cols, data: new_data)
}

fn map_data(data: BitArray, f: fn(Float) -> Float, acc: BitArray) -> BitArray {
  case data {
    <<x:float-little-size(64), rest:bits>> ->
      map_data(rest, f, <<acc:bits, { f(x) }:float-little-size(64)>>)
    _ -> acc
  }
}

fn zip_data(
  a: BitArray,
  b: BitArray,
  f: fn(Float, Float) -> Float,
  acc: BitArray,
) -> BitArray {
  case a, b {
    <<x:float-little-size(64), arest:bits>>,
      <<y:float-little-size(64), brest:bits>>
    ->
      zip_data(arest, brest, f, <<acc:bits, { f(x, y) }:float-little-size(64)>>)
    _, _ -> acc
  }
}

// ============================================================================
// Linear algebra
// ============================================================================

/// Transpose. O(rows · cols) but with contiguous writes.
pub fn transpose(m: DenseMat) -> DenseMat {
  let new_data = transpose_loop(m, 0, 0, <<>>)
  DenseMat(rows: m.cols, cols: m.rows, data: new_data)
}

fn transpose_loop(
  m: DenseMat,
  new_row: Int,
  new_col: Int,
  acc: BitArray,
) -> BitArray {
  case new_row >= m.cols {
    True -> acc
    False -> {
      let v = case get(m, new_col, new_row) {
        Ok(x) -> x
        Error(_) -> 0.0
      }
      let next_acc = <<acc:bits, v:float-little-size(64)>>
      case new_col + 1 >= m.rows {
        True -> transpose_loop(m, new_row + 1, 0, next_acc)
        False -> transpose_loop(m, new_row, new_col + 1, next_acc)
      }
    }
  }
}

/// Matrix product A · B. Errors on shape mismatch.
///
/// Naïve triple-loop O(n³). For larger matrices defer to `viva_tensor` which
/// dispatches to BLAS / cuBLAS.
pub fn mul(a: DenseMat, b: DenseMat) -> Result(DenseMat, Nil) {
  case a.cols != b.rows {
    True -> Error(Nil)
    False -> {
      let new_data = matmul_loop(a, b, 0, 0, <<>>)
      Ok(DenseMat(rows: a.rows, cols: b.cols, data: new_data))
    }
  }
}

fn matmul_loop(
  a: DenseMat,
  b: DenseMat,
  i: Int,
  j: Int,
  acc: BitArray,
) -> BitArray {
  case i >= a.rows {
    True -> acc
    False -> {
      let v = dot_row_col(a, b, i, j, 0, 0.0)
      let next_acc = <<acc:bits, v:float-little-size(64)>>
      case j + 1 >= b.cols {
        True -> matmul_loop(a, b, i + 1, 0, next_acc)
        False -> matmul_loop(a, b, i, j + 1, next_acc)
      }
    }
  }
}

fn dot_row_col(
  a: DenseMat,
  b: DenseMat,
  i: Int,
  j: Int,
  k: Int,
  acc: Float,
) -> Float {
  case k >= a.cols {
    True -> acc
    False -> {
      let av = case get(a, i, k) {
        Ok(x) -> x
        Error(_) -> 0.0
      }
      let bv = case get(b, k, j) {
        Ok(x) -> x
        Error(_) -> 0.0
      }
      dot_row_col(a, b, i, j, k + 1, acc +. av *. bv)
    }
  }
}

/// Frobenius norm √(Σ aᵢⱼ²).
pub fn frobenius(m: DenseMat) -> Float {
  scalar.sqrt(sum_squared(m.data, 0.0))
}

fn sum_squared(data: BitArray, acc: Float) -> Float {
  case data {
    <<x:float-little-size(64), rest:bits>> -> sum_squared(rest, acc +. x *. x)
    _ -> acc
  }
}

/// Trace of a square matrix.
pub fn trace(m: DenseMat) -> Result(Float, Nil) {
  case m.rows != m.cols {
    True -> Error(Nil)
    False -> Ok(trace_loop(m, 0, 0.0))
  }
}

fn trace_loop(m: DenseMat, i: Int, acc: Float) -> Float {
  case i >= m.rows {
    True -> acc
    False ->
      case get(m, i, i) {
        Ok(v) -> trace_loop(m, i + 1, acc +. v)
        Error(_) -> acc
      }
  }
}

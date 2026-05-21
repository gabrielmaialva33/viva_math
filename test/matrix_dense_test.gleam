//// Tests for `viva_math/matrix_dense` — `BitArray`-backed dense matrix.
//// Covers shape constructors, accessors, algebraic identities, and norms.

import gleeunit/should
import test_support.{is_close}
import viva_math/matrix_dense as md

// ============================================================================
// Construction + shape
// ============================================================================

pub fn md_zeros_shape_test() {
  let assert Ok(m) = md.zeros(2, 3)
  should.equal(m.rows, 2)
  should.equal(m.cols, 3)
}

pub fn md_zeros_invalid_dim_test() {
  md.zeros(0, 3) |> should.equal(Error(Nil))
  md.zeros(2, -1) |> should.equal(Error(Nil))
}

pub fn md_identity_diagonal_test() {
  let assert Ok(i3) = md.identity(3)
  let assert Ok(d00) = md.get(i3, 0, 0)
  let assert Ok(d11) = md.get(i3, 1, 1)
  let assert Ok(d22) = md.get(i3, 2, 2)
  let assert Ok(off) = md.get(i3, 0, 1)
  is_close(d00, 1.0, 1.0e-12) |> should.be_true
  is_close(d11, 1.0, 1.0e-12) |> should.be_true
  is_close(d22, 1.0, 1.0e-12) |> should.be_true
  is_close(off, 0.0, 1.0e-12) |> should.be_true
}

pub fn md_from_list_roundtrip_test() {
  let assert Ok(m) = md.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
  let rows = md.to_rows(m)
  case rows {
    [[a, b], [c, d]] -> {
      is_close(a, 1.0, 1.0e-12) |> should.be_true
      is_close(b, 2.0, 1.0e-12) |> should.be_true
      is_close(c, 3.0, 1.0e-12) |> should.be_true
      is_close(d, 4.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

pub fn md_from_list_wrong_size_test() {
  md.from_list(2, 2, [1.0, 2.0, 3.0]) |> should.equal(Error(Nil))
}

pub fn md_row_extraction_test() {
  let assert Ok(m) = md.from_list(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let assert Ok(r0) = md.row(m, 0)
  let assert Ok(r1) = md.row(m, 1)
  case r0, r1 {
    [a, b, c], [d, e, f] -> {
      is_close(a, 1.0, 1.0e-12) |> should.be_true
      is_close(b, 2.0, 1.0e-12) |> should.be_true
      is_close(c, 3.0, 1.0e-12) |> should.be_true
      is_close(d, 4.0, 1.0e-12) |> should.be_true
      is_close(e, 5.0, 1.0e-12) |> should.be_true
      is_close(f, 6.0, 1.0e-12) |> should.be_true
    }
    _, _ -> should.fail()
  }
}

// ============================================================================
// Algebraic identities
// ============================================================================

// (A + 0) = A
pub fn md_add_zero_identity_test() {
  let assert Ok(a) = md.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
  let assert Ok(z) = md.zeros(2, 2)
  let assert Ok(sum) = md.add(a, z)
  case md.to_rows(sum) {
    [[w, x], [y, q]] -> {
      is_close(w, 1.0, 1.0e-12) |> should.be_true
      is_close(x, 2.0, 1.0e-12) |> should.be_true
      is_close(y, 3.0, 1.0e-12) |> should.be_true
      is_close(q, 4.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

// (A − A) = 0
pub fn md_sub_self_zero_test() {
  let assert Ok(a) = md.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
  let assert Ok(d) = md.sub(a, a)
  is_close(md.frobenius(d), 0.0, 1.0e-12) |> should.be_true
}

// Shape mismatch → Error
pub fn md_add_shape_mismatch_test() {
  let assert Ok(a) = md.zeros(2, 2)
  let assert Ok(b) = md.zeros(3, 3)
  md.add(a, b) |> should.equal(Error(Nil))
}

// (sA) entrywise scales each element
pub fn md_scale_test() {
  let assert Ok(a) = md.from_list(1, 2, [2.0, 3.0])
  let scaled = md.scale(a, 5.0)
  case md.to_rows(scaled) {
    [[x, y]] -> {
      is_close(x, 10.0, 1.0e-12) |> should.be_true
      is_close(y, 15.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

// (A ⊙ A) entrywise — Hadamard square
pub fn md_hadamard_test() {
  let assert Ok(a) = md.from_list(1, 2, [2.0, 3.0])
  let assert Ok(h) = md.hadamard(a, a)
  case md.to_rows(h) {
    [[x, y]] -> {
      is_close(x, 4.0, 1.0e-12) |> should.be_true
      is_close(y, 9.0, 1.0e-12) |> should.be_true
    }
    _ -> should.fail()
  }
}

// (A^T)^T = A
pub fn md_transpose_involution_test() {
  let assert Ok(a) = md.from_list(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tt = md.transpose(md.transpose(a))
  let assert Ok(diff) = md.sub(tt, a)
  is_close(md.frobenius(diff), 0.0, 1.0e-12) |> should.be_true
}

// I·A = A
pub fn md_identity_left_neutral_test() {
  let assert Ok(i2) = md.identity(2)
  let assert Ok(a) = md.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
  let assert Ok(p) = md.mul(i2, a)
  let assert Ok(diff) = md.sub(p, a)
  is_close(md.frobenius(diff), 0.0, 1.0e-12) |> should.be_true
}

// A·I = A
pub fn md_identity_right_neutral_test() {
  let assert Ok(i2) = md.identity(2)
  let assert Ok(a) = md.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
  let assert Ok(p) = md.mul(a, i2)
  let assert Ok(diff) = md.sub(p, a)
  is_close(md.frobenius(diff), 0.0, 1.0e-12) |> should.be_true
}

// trace(I_n) = n
pub fn md_trace_identity_test() {
  let assert Ok(i3) = md.identity(3)
  let assert Ok(t) = md.trace(i3)
  is_close(t, 3.0, 1.0e-12) |> should.be_true
}

// ||A||_F² = Σ a_ij²
pub fn md_frobenius_test() {
  let assert Ok(a) = md.from_list(2, 2, [3.0, 4.0, 0.0, 0.0])
  // |A|_F = sqrt(9 + 16) = 5
  is_close(md.frobenius(a), 5.0, 1.0e-12) |> should.be_true
}

// `byte_size` is rows · cols · 8 bytes (64-bit floats).
pub fn md_byte_size_test() {
  let assert Ok(m) = md.zeros(3, 4)
  should.equal(md.byte_size(m), 3 * 4 * 8)
}

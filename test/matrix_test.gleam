import gleeunit/should
import test_support.{is_close, is_close_list, is_close_vec3, loose, tight}
import viva_math/constants
import viva_math/matrix
import viva_math/vector.{Vec3}

pub fn mat2_zero_identity_add_scale_trace_test() {
  let a = matrix.Mat2(1.0, 2.0, 3.0, 4.0)
  let z = matrix.mat2_zero()
  let i = matrix.mat2_identity()

  mat2_close(matrix.mat2_add(a, z), a, tight)
  mat2_close(matrix.mat2_scale(a, 2.0), matrix.Mat2(2.0, 4.0, 6.0, 8.0), tight)
  mat2_close(matrix.mat2_mul(i, a), a, tight)
  mat2_close(matrix.mat2_mul(a, i), a, tight)

  matrix.mat2_trace(a)
  |> is_close(5.0, tight)
  |> should.be_true
}

pub fn mat2_inverse_rotation_and_eigenvalues_test() {
  let a = matrix.Mat2(4.0, 7.0, 2.0, 6.0)
  let assert Ok(inv) = matrix.mat2_inverse(a)
  mat2_close(matrix.mat2_mul(a, inv), matrix.mat2_identity(), tight)

  mat2_close(matrix.mat2_rotation(0.0), matrix.mat2_identity(), tight)
  mat2_close(matrix.mat2_rotation(constants.tau), matrix.mat2_identity(), loose)

  let assert Ok(eigs) = matrix.mat2_eigenvalues(matrix.Mat2(2.0, 0.0, 0.0, 5.0))
  eigs.0 |> is_close(5.0, tight) |> should.be_true
  eigs.1 |> is_close(2.0, tight) |> should.be_true
}

pub fn mat3_zero_identity_add_sub_scale_test() {
  let a = matrix.Mat3(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0)
  let z = matrix.mat3_zero()
  let i = matrix.mat3_identity()

  mat3_close(matrix.mat3_add(a, z), a, tight)
  mat3_close(matrix.mat3_add(a, matrix.mat3_scale(a, -1.0)), z, tight)
  mat3_close(
    matrix.mat3_scale(a, 0.5),
    matrix.Mat3(0.5, 1.0, 1.5, 0.0, 0.5, 2.0, 2.5, 3.0, 0.0),
    tight,
  )
  mat3_close(matrix.mat3_mul(i, a), a, tight)
  mat3_close(matrix.mat3_mul(a, i), a, tight)
}

pub fn mat3_transpose_determinant_trace_inverse_test() {
  let a = matrix.Mat3(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0)
  let b = matrix.Mat3(2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 4.0, 2.0, 1.0)

  mat3_close(matrix.mat3_transpose(matrix.mat3_transpose(a)), a, tight)

  matrix.mat3_determinant(matrix.mat3_transpose(a))
  |> is_close(matrix.mat3_determinant(a), tight)
  |> should.be_true

  matrix.mat3_trace(matrix.mat3_add(a, b))
  |> is_close(matrix.mat3_trace(a) +. matrix.mat3_trace(b), tight)
  |> should.be_true

  let assert Ok(inv) = matrix.mat3_inverse(a)
  mat3_close(matrix.mat3_mul(a, inv), matrix.mat3_identity(), tight)
}

pub fn mat3_vector_rotations_frobenius_and_eigenvalues_test() {
  let v = Vec3(1.0, 2.0, 3.0)
  matrix.mat3_mul_vec3(matrix.mat3_identity(), v)
  |> is_close_vec3(v, tight)
  |> should.be_true

  matrix.mat3_mul_vec3(
    matrix.mat3_rot_x(constants.half_pi),
    Vec3(0.0, 1.0, 0.0),
  )
  |> is_close_vec3(Vec3(0.0, 0.0, 1.0), tight)
  |> should.be_true

  matrix.mat3_mul_vec3(
    matrix.mat3_rot_y(constants.half_pi),
    Vec3(0.0, 0.0, 1.0),
  )
  |> is_close_vec3(Vec3(1.0, 0.0, 0.0), tight)
  |> should.be_true

  matrix.mat3_mul_vec3(
    matrix.mat3_rot_z(constants.half_pi),
    Vec3(1.0, 0.0, 0.0),
  )
  |> is_close_vec3(Vec3(0.0, 1.0, 0.0), tight)
  |> should.be_true

  matrix.mat3_frobenius(matrix.mat3_identity())
  |> is_close(constants.sqrt_3, tight)
  |> should.be_true

  let assert Ok(eigs) =
    matrix.mat3_symmetric_eigenvalues(matrix.mat3_diagonal(3.0, 1.0, 2.0))
  eigs.0 |> is_close(1.0, tight) |> should.be_true
  eigs.1 |> is_close(2.0, tight) |> should.be_true
  eigs.2 |> is_close(3.0, tight) |> should.be_true
}

pub fn mat4_constructors_test() {
  should.equal(matrix.mat4_identity().rows, [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ])

  should.equal(matrix.mat4_zero().rows, [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
  ])

  should.equal(matrix.mat4_translation(1.0, 2.0, 3.0).rows, [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
  ])

  should.equal(matrix.mat4_scale(2.0, 3.0, 4.0).rows, [
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0],
    [0.0, 0.0, 4.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ])
}

pub fn matn_construction_identity_and_transpose_test() {
  let assert Ok(a) = matrix.matn_from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  should.equal(a.rows, 2)
  should.equal(a.cols, 3)
  should.equal(a.data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

  matrix.matn_from_rows([]) |> should.equal(Error(Nil))
  matrix.matn_from_rows([[]]) |> should.equal(Error(Nil))
  matrix.matn_from_rows([[1.0], [2.0, 3.0]]) |> should.equal(Error(Nil))

  let zeros = matrix.matn_zeros(2, 3)
  should.equal(zeros.data, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

  let id = matrix.matn_identity(3)
  should.equal(id.data, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

  let tt = matrix.matn_transpose(matrix.matn_transpose(a))
  should.equal(tt.data, a.data)
}

pub fn matn_basic_algebra_and_norms_test() {
  let assert Ok(a) = matrix.matn_from_rows([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(b) = matrix.matn_from_rows([[5.0, 6.0], [7.0, 8.0]])
  let id = matrix.matn_identity(2)
  let z = matrix.matn_zeros(2, 2)

  let assert Ok(left) = matrix.matn_mul(id, a)
  let assert Ok(right) = matrix.matn_mul(a, id)
  should.equal(left.data, a.data)
  should.equal(right.data, a.data)

  let assert Ok(sum) = matrix.matn_add(a, z)
  should.equal(sum.data, a.data)

  let assert Ok(diff) = matrix.matn_add(a, matrix.matn_scale(a, -1.0))
  should.equal(diff.data, z.data)

  let assert Ok(ab) = matrix.matn_mul(a, b)
  should.equal(ab.data, [[19.0, 22.0], [43.0, 50.0]])

  let assert Ok(trace_sum) =
    matrix.matn_trace(matrix.matn_add(a, b) |> result_unwrap)
  let assert Ok(trace_a) = matrix.matn_trace(a)
  let assert Ok(trace_b) = matrix.matn_trace(b)
  trace_sum |> is_close(trace_a +. trace_b, tight) |> should.be_true

  matrix.matn_frobenius(a)
  |> is_close(5.477_225_575_051_661, tight)
  |> should.be_true
}

pub fn matn_vector_product_and_shape_errors_test() {
  let assert Ok(a) = matrix.matn_from_rows([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(wide) = matrix.matn_from_rows([[1.0, 2.0, 3.0]])

  let assert Ok(product) = matrix.matn_mul_vec(a, [10.0, 1.0])
  product |> is_close_list([12.0, 34.0], tight) |> should.be_true

  matrix.matn_mul(a, wide) |> should.equal(Error(Nil))
  matrix.matn_add(a, wide) |> should.equal(Error(Nil))
  matrix.matn_trace(wide) |> should.equal(Error(Nil))
  matrix.matn_mul_vec(a, [1.0]) |> should.equal(Error(Nil))
}

fn mat2_close(a: matrix.Mat2, b: matrix.Mat2, tol: Float) {
  a.m11 |> is_close(b.m11, tol) |> should.be_true
  a.m12 |> is_close(b.m12, tol) |> should.be_true
  a.m21 |> is_close(b.m21, tol) |> should.be_true
  a.m22 |> is_close(b.m22, tol) |> should.be_true
}

fn mat3_close(a: matrix.Mat3, b: matrix.Mat3, tol: Float) {
  a.m11 |> is_close(b.m11, tol) |> should.be_true
  a.m12 |> is_close(b.m12, tol) |> should.be_true
  a.m13 |> is_close(b.m13, tol) |> should.be_true
  a.m21 |> is_close(b.m21, tol) |> should.be_true
  a.m22 |> is_close(b.m22, tol) |> should.be_true
  a.m23 |> is_close(b.m23, tol) |> should.be_true
  a.m31 |> is_close(b.m31, tol) |> should.be_true
  a.m32 |> is_close(b.m32, tol) |> should.be_true
  a.m33 |> is_close(b.m33, tol) |> should.be_true
}

fn result_unwrap(r: Result(a, Nil)) -> a {
  let assert Ok(x) = r
  x
}

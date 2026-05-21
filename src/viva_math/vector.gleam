//// Vector operations for 3D space (PAD model).
////
//// Vec3 is the fundamental type for emotional state in VIVA:
//// - Pleasure (x): [-1, 1] - sadness ↔ joy
//// - Arousal (y): [-1, 1] - calm ↔ excitement
//// - Dominance (z): [-1, 1] - submission ↔ control

import gleam/float
import gleam/list
import viva_math/common

/// A 3-dimensional vector representing PAD emotional state.
pub type Vec3 {
  Vec3(x: Float, y: Float, z: Float)
}

/// Create a zero vector.
pub fn zero() -> Vec3 {
  Vec3(0.0, 0.0, 0.0)
}

/// Create a vector with all components set to the same value.
pub fn splat(value: Float) -> Vec3 {
  Vec3(value, value, value)
}

/// Create a Vec3 from a list of 3 floats.
/// Returns Error if list doesn't have exactly 3 elements.
pub fn from_list(values: List(Float)) -> Result(Vec3, Nil) {
  case values {
    [x, y, z] -> Ok(Vec3(x, y, z))
    _ -> Error(Nil)
  }
}

/// Convert Vec3 to a list of floats.
pub fn to_list(v: Vec3) -> List(Float) {
  [v.x, v.y, v.z]
}

/// Add two vectors.
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
  Vec3(a.x +. b.x, a.y +. b.y, a.z +. b.z)
}

/// Subtract vector b from vector a.
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
  Vec3(a.x -. b.x, a.y -. b.y, a.z -. b.z)
}

/// Multiply vector by scalar.
pub fn scale(v: Vec3, s: Float) -> Vec3 {
  Vec3(v.x *. s, v.y *. s, v.z *. s)
}

/// Divide vector by scalar. Returns Error if scalar is zero.
pub fn div(v: Vec3, s: Float) -> Result(Vec3, Nil) {
  case s == 0.0 {
    True -> Error(Nil)
    False -> Ok(Vec3(v.x /. s, v.y /. s, v.z /. s))
  }
}

/// Negate a vector.
pub fn negate(v: Vec3) -> Vec3 {
  Vec3(0.0 -. v.x, 0.0 -. v.y, 0.0 -. v.z)
}

/// Component-wise multiplication (Hadamard product).
pub fn multiply(a: Vec3, b: Vec3) -> Vec3 {
  Vec3(a.x *. b.x, a.y *. b.y, a.z *. b.z)
}

/// Dot product of two vectors.
pub fn dot(a: Vec3, b: Vec3) -> Float {
  a.x *. b.x +. a.y *. b.y +. a.z *. b.z
}

/// Cross product of two vectors.
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
  Vec3(
    a.y *. b.z -. a.z *. b.y,
    a.z *. b.x -. a.x *. b.z,
    a.x *. b.y -. a.y *. b.x,
  )
}

/// Squared length of a vector (avoids sqrt for comparisons).
pub fn length_squared(v: Vec3) -> Float {
  dot(v, v)
}

/// Length (magnitude) of a vector.
pub fn length(v: Vec3) -> Float {
  let squared = length_squared(v)
  case float.square_root(squared) {
    Ok(result) -> result
    Error(_) -> 0.0
  }
}

/// Euclidean distance between two vectors.
pub fn distance(a: Vec3, b: Vec3) -> Float {
  length(sub(a, b))
}

/// Squared distance between two vectors (avoids sqrt).
pub fn distance_squared(a: Vec3, b: Vec3) -> Float {
  length_squared(sub(a, b))
}

/// Normalize a vector to unit length.
/// Returns zero vector if input has zero length.
pub fn normalize(v: Vec3) -> Vec3 {
  let len = length(v)
  case len == 0.0 {
    True -> zero()
    False -> Vec3(v.x /. len, v.y /. len, v.z /. len)
  }
}

/// Linear interpolation between two vectors.
pub fn lerp(a: Vec3, b: Vec3, t: Float) -> Vec3 {
  Vec3(
    common.lerp(a.x, b.x, t),
    common.lerp(a.y, b.y, t),
    common.lerp(a.z, b.z, t),
  )
}

/// Clamp each component to [min, max] range.
pub fn clamp(v: Vec3, min: Float, max: Float) -> Vec3 {
  Vec3(
    common.clamp(v.x, min, max),
    common.clamp(v.y, min, max),
    common.clamp(v.z, min, max),
  )
}

/// Clamp vector to PAD range [-1, 1] for all components.
pub fn clamp_pad(v: Vec3) -> Vec3 {
  clamp(v, -1.0, 1.0)
}

/// Apply a function to each component.
pub fn map(v: Vec3, f: fn(Float) -> Float) -> Vec3 {
  Vec3(f(v.x), f(v.y), f(v.z))
}

/// Check if two vectors are approximately equal.
pub fn is_close(a: Vec3, b: Vec3, tolerance: Float) -> Bool {
  let dx = float.absolute_value(a.x -. b.x)
  let dy = float.absolute_value(a.y -. b.y)
  let dz = float.absolute_value(a.z -. b.z)
  dx <=. tolerance && dy <=. tolerance && dz <=. tolerance
}

/// Compute weighted average of vectors.
/// weights and vectors must have same length.
pub fn weighted_average(
  vectors: List(Vec3),
  weights: List(Float),
) -> Result(Vec3, Nil) {
  case list.length(vectors) == list.length(weights) {
    False -> Error(Nil)
    True -> {
      let sum_weights = list.fold(weights, 0.0, fn(acc, w) { acc +. w })
      case sum_weights == 0.0 {
        True -> Error(Nil)
        False -> {
          let weighted =
            list.zip(vectors, weights)
            |> list.map(fn(pair) {
              let #(v, w) = pair
              scale(v, w)
            })
            |> list.fold(zero(), add)
          Ok(scale(weighted, 1.0 /. sum_weights))
        }
      }
    }
  }
}

/// Component-wise minimum of two vectors.
pub fn min(a: Vec3, b: Vec3) -> Vec3 {
  Vec3(float.min(a.x, b.x), float.min(a.y, b.y), float.min(a.z, b.z))
}

/// Component-wise maximum of two vectors.
pub fn max(a: Vec3, b: Vec3) -> Vec3 {
  Vec3(float.max(a.x, b.x), float.max(a.y, b.y), float.max(a.z, b.z))
}

/// Sum of all components.
pub fn sum(v: Vec3) -> Float {
  v.x +. v.y +. v.z
}

/// Average of all components.
pub fn average(v: Vec3) -> Float {
  sum(v) /. 3.0
}

// PAD-specific aliases

/// Create a PAD vector (Pleasure, Arousal, Dominance).
pub fn pad(pleasure: Float, arousal: Float, dominance: Float) -> Vec3 {
  Vec3(pleasure, arousal, dominance) |> clamp_pad
}

/// Get Pleasure component (x).
pub fn pleasure(v: Vec3) -> Float {
  v.x
}

/// Get Arousal component (y).
pub fn arousal(v: Vec3) -> Float {
  v.y
}

/// Get Dominance component (z).
pub fn dominance(v: Vec3) -> Float {
  v.z
}

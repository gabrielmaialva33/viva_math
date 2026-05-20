//// Attractor dynamics for emotional states.
////
//// Based on Mehrabian's PAD model (1996) and dynamical systems theory.
//// Emotions form attractors in PAD space - stable states the system tends toward.
////
//// The 8 basic emotions correspond to the octants of the PAD cube:
//// - Joy:     (+P, +A, +D)
//// - Sadness: (-P, -A, -D)
//// - Anger:   (-P, +A, +D)
//// - Fear:    (-P, +A, -D)
//// - Surprise: (+P, +A, -D) or (-P, +A, -D)
//// - Disgust: (-P, -A, +D)
//// - Trust:   (+P, -A, +D)
//// - Anticipation: (+P, +A, +D)
////
//// References:
//// - Mehrabian (1996) "Pleasure-arousal-dominance: A general framework"
//// - Russell (2003) "Core affect and the psychological construction of emotion"

import gleam/float
import gleam/list
import gleam_community/maths
import viva_math/vector.{type Vec3, Vec3}

/// An attractor in PAD space with a name and position.
pub type Attractor {
  Attractor(name: String, position: Vec3)
}

/// Result of attractor analysis.
pub type AttractorResult {
  AttractorResult(
    /// Nearest attractor
    nearest: Attractor,
    /// Distance to nearest attractor
    distance: Float,
    /// All attractors with their influence weights
    weights: List(#(Attractor, Float)),
  )
}

/// The 8 basic emotional attractors (Mehrabian octants).
/// Values from empirical research on emotion self-reports.
pub fn emotional_attractors() -> List(Attractor) {
  [
    // Positive emotions
    Attractor(name: "joy", position: Vec3(0.76, 0.48, 0.35)),
    Attractor(name: "excitement", position: Vec3(0.62, 0.75, 0.38)),
    Attractor(name: "trust", position: Vec3(0.58, -0.23, 0.42)),
    Attractor(name: "serenity", position: Vec3(0.45, -0.42, 0.21)),
    // Negative emotions
    Attractor(name: "sadness", position: Vec3(-0.63, -0.27, -0.33)),
    Attractor(name: "fear", position: Vec3(-0.64, 0.6, -0.43)),
    Attractor(name: "anger", position: Vec3(-0.51, 0.59, 0.25)),
    Attractor(name: "disgust", position: Vec3(-0.6, 0.35, 0.11)),
  ]
}

/// Find the nearest attractor to a given point.
pub fn nearest(
  point: Vec3,
  attractors: List(Attractor),
) -> Result(Attractor, Nil) {
  case attractors {
    [] -> Error(Nil)
    [first, ..rest] -> {
      let initial = #(first, vector.distance(point, first.position))
      let result =
        list.fold(rest, initial, fn(acc, attr) {
          let dist = vector.distance(point, attr.position)
          case dist <. acc.1 {
            True -> #(attr, dist)
            False -> acc
          }
        })
      Ok(result.0)
    }
  }
}

/// Calculate influence weights for all attractors using softmax of negative distances.
///
/// CORRECTED per DeepSeek R1 validation:
/// w_i = exp(-γ × d_i) / Σ_j exp(-γ × d_j)
///
/// Where γ = 1/temperature (higher temp = softer weights, lower temp = sharper).
/// This is more numerically stable than 1/d and matches Boltzmann distribution.
///
/// Closer attractors have higher weights. The temperature parameter controls
/// how "sharp" the weighting is (lower temp = more weight on nearest).
pub fn basin_weights(
  point: Vec3,
  attractors: List(Attractor),
  temperature: Float,
) -> List(#(Attractor, Float)) {
  case attractors {
    [] -> []
    _ -> {
      // γ = 1/temperature (avoid division by zero)
      let gamma = case temperature <=. 0.0 {
        True -> 1.0
        False -> 1.0 /. temperature
      }

      // Calculate -γ × distance for each attractor
      let neg_gamma_distances =
        list.map(attractors, fn(attr) {
          let dist = vector.distance(point, attr.position)
          #(attr, 0.0 -. gamma *. dist)
        })

      // Max-subtraction for numerical stability (pattern from viva_glyph)
      let max_val =
        list.fold(neg_gamma_distances, -1000.0, fn(acc, pair) {
          float.max(acc, pair.1)
        })

      // Softmax: exp(-γd - max) / sum
      let exps =
        list.map(neg_gamma_distances, fn(pair) {
          #(pair.0, maths.exponential(pair.1 -. max_val))
        })
      let sum = list.fold(exps, 0.0, fn(acc, pair) { acc +. pair.1 })

      case sum == 0.0 {
        True -> list.map(attractors, fn(a) { #(a, 0.0) })
        False -> list.map(exps, fn(pair) { #(pair.0, pair.1 /. sum) })
      }
    }
  }
}

/// Comprehensive attractor analysis for a point.
pub fn analyze(
  point: Vec3,
  attractors: List(Attractor),
  temperature: Float,
) -> Result(AttractorResult, Nil) {
  case nearest(point, attractors) {
    Error(Nil) -> Error(Nil)
    Ok(near) -> {
      let dist = vector.distance(point, near.position)
      let weights = basin_weights(point, attractors, temperature)
      Ok(AttractorResult(nearest: near, distance: dist, weights: weights))
    }
  }
}

/// Classify emotional state by nearest attractor name.
pub fn classify_emotion(point: Vec3) -> String {
  case nearest(point, emotional_attractors()) {
    Ok(attr) -> attr.name
    Error(Nil) -> "neutral"
  }
}

/// Compute attractor pull - force vector toward nearest attractor.
///
/// The pull strength increases with distance from attractor (spring-like).
/// strength parameter controls overall force magnitude.
pub fn attractor_pull(
  point: Vec3,
  attractor: Attractor,
  strength: Float,
) -> Vec3 {
  let diff = vector.sub(attractor.position, point)
  let dist = vector.length(diff)
  case dist == 0.0 {
    True -> vector.zero()
    False -> {
      let normalized = vector.scale(diff, 1.0 /. dist)
      // Pull proportional to distance
      vector.scale(normalized, strength *. dist)
    }
  }
}

/// Compute weighted pull from all attractors.
///
/// Each attractor pulls proportionally to its basin weight.
pub fn weighted_pull(
  point: Vec3,
  attractors: List(Attractor),
  strength: Float,
  temperature: Float,
) -> Vec3 {
  let weights = basin_weights(point, attractors, temperature)
  list.fold(weights, vector.zero(), fn(acc, pair) {
    let #(attr, weight) = pair
    let pull = attractor_pull(point, attr, strength *. weight)
    vector.add(acc, pull)
  })
}

/// Ornstein-Uhlenbeck mean reversion toward attractor.
///
/// dx = theta * (attractor - x) * dt
///
/// This is the deterministic part of O-U process.
/// theta controls reversion speed (higher = faster return to attractor).
pub fn ou_mean_reversion(
  current: Vec3,
  attractor: Vec3,
  theta: Float,
  dt: Float,
) -> Vec3 {
  let diff = vector.sub(attractor, current)
  let delta = vector.scale(diff, theta *. dt)
  vector.add(current, delta)
}

/// Check if point is in basin of an attractor.
///
/// A point is "in" a basin if that attractor has the highest weight.
pub fn in_basin(
  point: Vec3,
  attractor: Attractor,
  all: List(Attractor),
) -> Bool {
  case nearest(point, all) {
    Ok(near) -> near.name == attractor.name
    Error(Nil) -> False
  }
}

/// Find all attractors within a given distance.
pub fn nearby_attractors(
  point: Vec3,
  attractors: List(Attractor),
  radius: Float,
) -> List(Attractor) {
  list.filter(attractors, fn(attr) {
    vector.distance(point, attr.position) <=. radius
  })
}

/// Interpolate between two attractors based on a blend factor.
///
/// t=0 gives first attractor, t=1 gives second.
pub fn blend_attractors(a: Attractor, b: Attractor, t: Float) -> Attractor {
  let pos = vector.lerp(a.position, b.position, t)
  let name = a.name <> "_" <> b.name
  Attractor(name: name, position: pos)
}

/// Create a custom attractor from name and PAD values.
pub fn create(
  name: String,
  pleasure: Float,
  arousal: Float,
  dominance: Float,
) -> Attractor {
  Attractor(name: name, position: vector.pad(pleasure, arousal, dominance))
}

/// Get the dominant emotion component (P, A, or D) for an attractor.
pub fn dominant_dimension(attractor: Attractor) -> String {
  let p = float.absolute_value(attractor.position.x)
  let a = float.absolute_value(attractor.position.y)
  let d = float.absolute_value(attractor.position.z)

  case p >=. a && p >=. d {
    True -> "pleasure"
    False ->
      case a >=. d {
        True -> "arousal"
        False -> "dominance"
      }
  }
}

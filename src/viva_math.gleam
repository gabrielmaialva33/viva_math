//// viva_math - Mathematical foundations for VIVA's sentient digital life.
////
//// A specialized math library for the BEAM, designed to sit between
//// `gleam_community_maths` (primitives) and `viva_tensor` (tensors + NIF).
////
//// ## Modules
////
//// ### Foundations
//// - `viva_math/scalar`     - Scalar math: erf, gelu, silu, mish, logsumexp...
//// - `viva_math/constants`  - High-precision constants (pi, e, sqrt_2pi...)
//// - `viva_math/common`     - Generic helpers: clamp, lerp, sigmoid, softmax
////
//// ### Geometry & linear algebra
//// - `viva_math/vector`     - Vec3 (PAD emotional space) + ops
//// - `viva_math/vec2`       - 2-D vectors (polar, planar)
//// - `viva_math/vec4`       - 4-D vectors (RGBA, homogeneous, quaternions)
//// - `viva_math/vecn`       - N-D vectors as List(Float)
//// - `viva_math/matrix`     - Mat2, Mat3, Mat4 and generic MatN
////
//// ### Stochastic & inference
//// - `viva_math/random`     - PRNG with opaque Seed (Erlang :rand backed)
//// - `viva_math/statistics` - mean, var, ema, percentile, correlation
//// - `viva_math/distributions` - gaussian/uniform/exponential/categorical
//// - `viva_math/entropy`    - Shannon, KL, JS, Rényi, Tsallis, Fisher
////
//// ### Dynamical systems
//// - `viva_math/cusp`       - Catastrophe theory (Thom)
//// - `viva_math/free_energy` - Friston FEP + active inference
//// - `viva_math/attractor`  - Mehrabian PAD attractors + OU dynamics
//// - `viva_math/ode`        - Euler / RK2 / RK4 / Euler-Maruyama / Milstein
//// - `viva_math/calculus`   - Finite differences + Simpson / Romberg
//// - `viva_math/scheduler`  - Cosine annealing, warmup, decay schedules
////
//// ## Quick start
////
//// ```gleam
//// import viva_math/scalar
//// import viva_math/vector
//// import viva_math/attractor
//// import viva_math/random
////
//// // PAD emotional state
//// let state = vector.pad(-0.3, 0.7, -0.2)
////
//// // Classify nearest emotion
//// let emotion = attractor.classify_emotion(state)
//// // -> "fear"
////
//// // Scalar activations for ML
//// let y = scalar.gelu(0.5)
////
//// // Seedable, reproducible normal sample
//// let seed = random.from_int(42)
//// let #(x, _) = random.normal(seed, 0.0, 1.0)
//// ```

import viva_math/attractor
import viva_math/common
import viva_math/cusp
import viva_math/entropy
import viva_math/free_energy
import viva_math/scalar
import viva_math/vector

/// Library version.
pub const version = "1.2.100"

/// Create a PAD vector with clamping. Shorthand for `vector.pad/3`.
pub fn pad(pleasure: Float, arousal: Float, dominance: Float) -> vector.Vec3 {
  vector.pad(pleasure, arousal, dominance)
}

/// Classify emotional state to nearest attractor name.
pub fn classify(state: vector.Vec3) -> String {
  attractor.classify_emotion(state)
}

/// Check if emotional state is volatile (cusp bistability).
pub fn is_volatile(arousal: Float, dominance: Float) -> Bool {
  cusp.from_arousal_dominance(arousal, dominance)
  |> cusp.is_bistable
}

/// Compute free energy from expected and actual states with default precision.
pub fn free_energy(
  expected: vector.Vec3,
  actual: vector.Vec3,
) -> free_energy.FreeEnergyState {
  let baseline = vector.zero()
  let complexity_weight = 0.1
  free_energy.compute_state_simple(
    expected,
    actual,
    baseline,
    complexity_weight,
  )
}

/// Shannon entropy of a probability distribution.
pub fn entropy(probabilities: List(Float)) -> Float {
  entropy.shannon(probabilities)
}

/// Standard sigmoid σ(x) = 1 / (1 + e^(-x)).
pub fn sigmoid(x: Float) -> Float {
  scalar.sigmoid(x)
}

/// Clamp value to [-1, 1] range.
pub fn clamp_bipolar(x: Float) -> Float {
  common.clamp_bipolar(x)
}

/// Error function. Delegates to `viva_math/scalar.erf` (Erlang `:math.erf`).
pub fn erf(x: Float) -> Float {
  scalar.erf(x)
}

/// GELU activation (exact form using erf).
pub fn gelu(x: Float) -> Float {
  scalar.gelu(x)
}

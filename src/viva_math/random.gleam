//// Pure functional random number generation.
////
//// Built on top of Erlang's `:rand` module (OTP 22+). The `Seed` type is
//// **opaque and immutable**: every sampling function returns a new seed,
//// allowing fully reproducible deterministic streams without process state.
////
//// The default algorithm is `exsss` (Xorshift116** with StarStar scrambler,
//// period 2¹¹⁶-1, 58-bit output). For longer streams or jump-ahead, use
//// `with_algo(Exro928ss, seed)`.
////
//// ## Why not the hash-based generator in `common`?
////
//// `common.deterministic_noise` is fine for reproducible *fixtures*, but it
//// is statistically weak. For Monte Carlo, neural net initialization or
//// Bayesian sampling, prefer this module.
////
//// ## Example
////
//// ```gleam
//// import viva_math/random
////
//// let seed0 = random.from_int(42)
//// let #(x, seed1) = random.uniform(seed0)
//// let #(y, seed2) = random.normal(seed1, 0.0, 1.0)
//// // Same seed0 always produces the same x, y sequence.
//// ```

import gleam/dict.{type Dict}
import gleam/list

// ============================================================================
// Types
// ============================================================================

/// Opaque immutable PRNG state.
///
/// Internally an Erlang `:rand` state tuple. Pass through sampling functions;
/// never mutate.
pub type Seed

/// Available PRNG algorithms exposed by OTP `:rand`.
pub type Algorithm {
  /// Xorshift116** — default, fast, good statistics. Period 2¹¹⁶-1.
  Exsss
  /// Xoroshiro928** — much longer period (2⁹²⁸-1), jump ahead.
  Exro928ss
  /// Xoroshiro116+. Slightly faster than Exsss but weak low bits.
  Exrop
  /// Xorshift116+. Legacy.
  Exsp
  /// Xorshift1024*. 64-bit precision.
  Exs1024s
  /// MWC59 — multiply-with-carry, very fast 32-bit floats.
  Mwc59
}

// ============================================================================
// FFI
// ============================================================================

@external(erlang, "viva_math_random_ffi", "seed_default")
fn ffi_seed_default(seed: Int) -> Seed

@external(erlang, "viva_math_random_ffi", "seed_with_algo")
fn ffi_seed_with_algo(algo: Algorithm, seed: Int) -> Seed

@external(erlang, "viva_math_random_ffi", "uniform_real")
fn ffi_uniform_real(state: Seed) -> #(Float, Seed)

@external(erlang, "viva_math_random_ffi", "uniform_int")
fn ffi_uniform_int(n: Int, state: Seed) -> #(Int, Seed)

@external(erlang, "viva_math_random_ffi", "normal_standard")
fn ffi_normal_standard(state: Seed) -> #(Float, Seed)

@external(erlang, "viva_math_random_ffi", "normal_with")
fn ffi_normal_with(mu: Float, sigma: Float, state: Seed) -> #(Float, Seed)

@external(erlang, "viva_math_random_ffi", "jump")
fn ffi_jump(state: Seed) -> Seed

// ============================================================================
// Constructors
// ============================================================================

/// Build a seed from an integer using the default algorithm (Exsss).
pub fn from_int(seed: Int) -> Seed {
  ffi_seed_default(seed)
}

/// Build a seed with an explicit algorithm.
pub fn with_algo(algorithm: Algorithm, seed: Int) -> Seed {
  ffi_seed_with_algo(algorithm, seed)
}

/// Advance the seed by 2⁶⁴ calls. Useful for non-overlapping parallel streams.
pub fn jump(seed: Seed) -> Seed {
  ffi_jump(seed)
}

// ============================================================================
// Basic samplers
// ============================================================================

/// Uniform float in [0.0, 1.0).
pub fn uniform(seed: Seed) -> #(Float, Seed) {
  ffi_uniform_real(seed)
}

/// Uniform float in [low, high).
pub fn uniform_in(seed: Seed, low: Float, high: Float) -> #(Float, Seed) {
  let #(u, s) = ffi_uniform_real(seed)
  #(low +. u *. { high -. low }, s)
}

/// Uniform integer in [1, n]. Returns an error if n < 1.
pub fn integer(seed: Seed, n: Int) -> Result(#(Int, Seed), Nil) {
  case n < 1 {
    True -> Error(Nil)
    False -> Ok(ffi_uniform_int(n, seed))
  }
}

/// Standard normal N(0, 1).
pub fn standard_normal(seed: Seed) -> #(Float, Seed) {
  ffi_normal_standard(seed)
}

/// Normal N(mu, sigma²).
pub fn normal(seed: Seed, mu: Float, sigma: Float) -> #(Float, Seed) {
  ffi_normal_with(mu, sigma, seed)
}

/// Bernoulli sample: True with probability `p`. `p` is clamped to [0, 1].
pub fn bernoulli(seed: Seed, p: Float) -> #(Bool, Seed) {
  let p_clamped = case p {
    p if p <. 0.0 -> 0.0
    p if p >. 1.0 -> 1.0
    _ -> p
  }
  let #(u, s) = ffi_uniform_real(seed)
  #(u <. p_clamped, s)
}

// ============================================================================
// Categorical / multinomial
// ============================================================================

/// Sample an index from a categorical distribution defined by `probs`.
///
/// Probabilities are renormalised internally. Returns an error if the list
/// is empty or sums to a non-positive value.
pub fn categorical(
  seed: Seed,
  probs: List(Float),
) -> Result(#(Int, Seed), Nil) {
  let any_negative = list.any(probs, fn(p) { p <. 0.0 })
  let total = list.fold(probs, 0.0, fn(acc, p) { acc +. p })
  case probs, any_negative, total >. 0.0 {
    [], _, _ -> Error(Nil)
    _, True, _ -> Error(Nil)
    _, _, False -> Error(Nil)
    _, _, True -> {
      let #(u, s) = ffi_uniform_real(seed)
      let target = u *. total
      Ok(#(categorical_walk(probs, target, 0, 0.0), s))
    }
  }
}

fn categorical_walk(
  probs: List(Float),
  target: Float,
  index: Int,
  acc: Float,
) -> Int {
  case probs {
    [] -> index - 1
    [p, ..rest] -> {
      let next_acc = acc +. p
      case target <. next_acc {
        True -> index
        False -> categorical_walk(rest, target, index + 1, next_acc)
      }
    }
  }
}

/// Pick a uniformly random element from a list.
pub fn choice(seed: Seed, xs: List(a)) -> Result(#(a, Seed), Nil) {
  case list.length(xs) {
    0 -> Error(Nil)
    n -> {
      let #(idx, s) = ffi_uniform_int(n, seed)
      case list_at(xs, idx - 1) {
        Ok(v) -> Ok(#(v, s))
        Error(_) -> Error(Nil)
      }
    }
  }
}

// ============================================================================
// Shuffle (Fisher-Yates via key tagging + sort)
// ============================================================================

/// Return a random permutation of `xs` via Fisher–Yates (modern Durstenfeld
/// variant), implemented on top of a `dict` for O(log n) indexed swaps.
///
/// Truly uniform over all n! permutations (no float-key collision bias the
/// sort-based shuffle suffered from).
pub fn shuffle(seed: Seed, xs: List(a)) -> #(List(a), Seed) {
  let n = list.length(xs)
  case n {
    0 -> #([], seed)
    _ -> {
      let initial = list_to_dict(xs, 0, dict.new())
      let #(shuffled, final_seed) = fisher_yates(initial, n - 1, seed)
      #(dict_to_list(shuffled, 0, n, []), final_seed)
    }
  }
}

fn list_to_dict(xs: List(a), i: Int, acc: Dict(Int, a)) -> Dict(Int, a) {
  case xs {
    [] -> acc
    [x, ..rest] -> list_to_dict(rest, i + 1, dict.insert(acc, i, x))
  }
}

fn dict_to_list(d: Dict(Int, a), i: Int, n: Int, acc: List(a)) -> List(a) {
  case i >= n {
    True -> list.reverse(acc)
    False ->
      case dict.get(d, i) {
        Ok(v) -> dict_to_list(d, i + 1, n, [v, ..acc])
        Error(_) -> list.reverse(acc)
      }
  }
}

fn fisher_yates(
  arr: Dict(Int, a),
  i: Int,
  seed: Seed,
) -> #(Dict(Int, a), Seed) {
  case i <= 0 {
    True -> #(arr, seed)
    False -> {
      // Draw j uniformly from [0, i]
      let #(j_raw, s1) = ffi_uniform_int(i + 1, seed)
      let j = j_raw - 1
      let swapped = swap(arr, i, j)
      fisher_yates(swapped, i - 1, s1)
    }
  }
}

fn swap(d: Dict(Int, a), i: Int, j: Int) -> Dict(Int, a) {
  case dict.get(d, i), dict.get(d, j) {
    Ok(vi), Ok(vj) -> dict.insert(dict.insert(d, i, vj), j, vi)
    _, _ -> d
  }
}

// ============================================================================
// Batch helpers
// ============================================================================

/// Sample n standard normals at once.
pub fn standard_normals(seed: Seed, n: Int) -> #(List(Float), Seed) {
  draw(n, seed, ffi_normal_standard, [])
}

/// Sample n uniforms in [0, 1) at once.
pub fn uniforms(seed: Seed, n: Int) -> #(List(Float), Seed) {
  draw(n, seed, ffi_uniform_real, [])
}

fn draw(
  n: Int,
  seed: Seed,
  sampler: fn(Seed) -> #(Float, Seed),
  acc: List(Float),
) -> #(List(Float), Seed) {
  case n <= 0 {
    True -> #(list.reverse(acc), seed)
    False -> {
      let #(x, s) = sampler(seed)
      draw(n - 1, s, sampler, [x, ..acc])
    }
  }
}

// ============================================================================
// Internal helpers
// ============================================================================

fn list_at(xs: List(a), idx: Int) -> Result(a, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at(rest, n - 1)
  }
}

//// Entropy and information theory functions.
////
//// Based on Shannon (1948) and Kullback-Leibler (1951).
//// Used for memory consolidation scoring and uncertainty quantification.
////
//// References:
//// - Shannon (1948) "A Mathematical Theory of Communication"
//// - Cover & Thomas (2006) "Elements of Information Theory"

import gleam/float
import gleam/list
import gleam_community/maths
import viva_math/precision

/// Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
///
/// Measures uncertainty/information content of a distribution.
/// Higher entropy = more uncertainty.
///
/// ## Examples
///
/// ```gleam
/// shannon([0.5, 0.5])     // -> 1.0 (maximum for 2 outcomes)
/// shannon([1.0, 0.0])     // -> 0.0 (no uncertainty)
/// shannon([0.25, 0.25, 0.25, 0.25]) // -> 2.0
/// ```
pub fn shannon(probabilities: List(Float)) -> Float {
  // Build -p · log₂(p) terms, then sum with Neumaier compensation to avoid
  // cancellation when probabilities span many orders of magnitude.
  let terms =
    list.fold(probabilities, [], fn(acc, p) {
      case p <=. 0.0 {
        True -> acc
        False ->
          case maths.logarithm_2(p) {
            Ok(log_p) -> [0.0 -. p *. log_p, ..acc]
            Error(_) -> acc
          }
      }
    })
  precision.neumaier_sum(terms)
}

/// Normalized Shannon entropy (0 to 1 range).
///
/// Divides by log₂(n) where n is number of outcomes.
pub fn shannon_normalized(probabilities: List(Float)) -> Float {
  let n = list.length(probabilities)
  case n <= 1 {
    True -> 0.0
    False -> {
      let h = shannon(probabilities)
      case maths.logarithm_2(int_to_float(n)) {
        Ok(max_h) ->
          case max_h == 0.0 {
            True -> 0.0
            False -> h /. max_h
          }
        Error(_) -> 0.0
      }
    }
  }
}

/// KL Divergence: D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
///
/// Measures how much P diverges from Q (not symmetric!).
/// P is the "true" distribution, Q is the approximation.
///
/// Returns Error if distributions have different lengths or Q has zeros where P is non-zero.
pub fn kl_divergence(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case list.length(p) == list.length(q) {
    False -> Error(Nil)
    True -> {
      let pairs = list.zip(p, q)
      let result =
        list.fold(pairs, Ok(0.0), fn(acc, pair) {
          case acc {
            Error(Nil) -> Error(Nil)
            Ok(sum) -> {
              let #(pi, qi) = pair
              case pi <=. 0.0 {
                True -> Ok(sum)
                // 0 * log(0/q) = 0
                False ->
                  case qi <=. 0.0 {
                    True -> Error(Nil)
                    // Can't have q=0 when p>0
                    False ->
                      case maths.natural_logarithm(pi /. qi) {
                        Ok(log_ratio) -> Ok(sum +. { pi *. log_ratio })
                        Error(_) -> Error(Nil)
                      }
                  }
              }
            }
          }
        })
      result
    }
  }
}

/// KL divergence with additive smoothing.
///
/// Adds `eps` to every q_i before normalising; useful when q contains zeros
/// where p > 0 (which would otherwise return `Error`). Standard practice in
/// language modelling and probabilistic ML.
///
/// Recommended `eps ∈ [1e-12, 1e-6]`. Larger values bias the result more.
pub fn kl_divergence_smoothed(
  p: List(Float),
  q: List(Float),
  eps: Float,
) -> Result(Float, Nil) {
  case list.length(p) == list.length(q), eps <. 0.0 {
    False, _ -> Error(Nil)
    _, True -> Error(Nil)
    True, False -> {
      // q' = (q + eps) / Σ(q + eps) to keep it a probability distribution
      let smoothed_unnorm = list.map(q, fn(qi) { qi +. eps })
      let total =
        list.fold(smoothed_unnorm, 0.0, fn(acc, x) { acc +. x })
      case total <=. 0.0 {
        True -> Error(Nil)
        False -> {
          let q_norm = list.map(smoothed_unnorm, fn(x) { x /. total })
          kl_divergence(p, q_norm)
        }
      }
    }
  }
}

/// Symmetric KL Divergence (Jensen-Shannon divergence without the 1/2).
///
/// D_sym(P, Q) = D_KL(P || Q) + D_KL(Q || P)
pub fn symmetric_kl(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case kl_divergence(p, q), kl_divergence(q, p) {
    Ok(d1), Ok(d2) -> Ok(d1 +. d2)
    _, _ -> Error(Nil)
  }
}

/// Jensen-Shannon Divergence: JS(P, Q) = (D_KL(P || M) + D_KL(Q || M)) / 2
/// where M = (P + Q) / 2
///
/// This is symmetric and bounded [0, 1] when using log₂.
pub fn jensen_shannon(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case list.length(p) == list.length(q) {
    False -> Error(Nil)
    True -> {
      // Compute M = (P + Q) / 2
      let m =
        list.zip(p, q)
        |> list.map(fn(pair) {
          let #(pi, qi) = pair
          { pi +. qi } /. 2.0
        })

      case kl_divergence(p, m), kl_divergence(q, m) {
        Ok(d_pm), Ok(d_qm) -> Ok({ d_pm +. d_qm } /. 2.0)
        _, _ -> Error(Nil)
      }
    }
  }
}

/// Cross-entropy: H(P, Q) = -Σ p(x) log q(x)
///
/// Used in machine learning loss functions.
/// H(P, Q) = H(P) + D_KL(P || Q)
pub fn cross_entropy(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case list.length(p) == list.length(q) {
    False -> Error(Nil)
    True -> {
      let pairs = list.zip(p, q)
      let result =
        list.fold(pairs, Ok(0.0), fn(acc, pair) {
          case acc {
            Error(Nil) -> Error(Nil)
            Ok(sum) -> {
              let #(pi, qi) = pair
              case pi <=. 0.0 {
                True -> Ok(sum)
                False ->
                  case qi <=. 0.0 {
                    True -> Error(Nil)
                    False ->
                      case maths.natural_logarithm(qi) {
                        Ok(log_q) -> Ok(sum -. { pi *. log_q })
                        Error(_) -> Error(Nil)
                      }
                  }
              }
            }
          }
        })
      result
    }
  }
}

/// Binary cross-entropy for single probability.
///
/// H(p, q) = -[p log(q) + (1-p) log(1-q)]
pub fn binary_cross_entropy(p: Float, q: Float) -> Result(Float, Nil) {
  // Clamp q to avoid log(0)
  let q_clamped = float.max(float.min(q, 0.999999), 0.000001)
  let q_inv = 1.0 -. q_clamped
  let p_inv = 1.0 -. p

  case maths.natural_logarithm(q_clamped), maths.natural_logarithm(q_inv) {
    Ok(log_q), Ok(log_q_inv) -> {
      let result = p *. log_q +. p_inv *. log_q_inv
      Ok(0.0 -. result)
    }
    _, _ -> Error(Nil)
  }
}

/// Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)
///
/// Measures shared information between two variables.
/// Takes marginal distributions and joint distribution as input.
pub fn mutual_information(
  px: List(Float),
  py: List(Float),
  pxy: List(List(Float)),
) -> Float {
  let hx = shannon(px)
  let hy = shannon(py)

  // Flatten joint distribution and compute joint entropy
  let pxy_flat = list.flatten(pxy)
  let hxy = shannon(pxy_flat)

  hx +. hy -. hxy
}

/// Conditional entropy: H(X|Y) = H(X, Y) - H(Y)
///
/// Uncertainty in X given knowledge of Y.
pub fn conditional_entropy(px: List(Float), pxy: List(List(Float))) -> Float {
  let hx = shannon(px)
  let pxy_flat = list.flatten(pxy)
  let hxy = shannon(pxy_flat)
  hxy -. hx
}

/// Relative entropy rate for sequences.
///
/// Used for measuring "surprise" in temporal data.
pub fn relative_entropy_rate(
  observed: List(Float),
  expected: List(Float),
) -> Result(Float, Nil) {
  case list.length(observed) == list.length(expected) {
    False -> Error(Nil)
    True -> {
      let n = list.length(observed)
      case n == 0 {
        True -> Ok(0.0)
        False ->
          case kl_divergence(observed, expected) {
            Ok(kl) -> Ok(kl /. int_to_float(n))
            Error(Nil) -> Error(Nil)
          }
      }
    }
  }
}

// Helper: convert int to float
fn int_to_float(n: Int) -> Float {
  case n {
    0 -> 0.0
    1 -> 1.0
    2 -> 2.0
    _ -> {
      let half = n / 2
      let remainder = n - half * 2
      int_to_float(half) *. 2.0 +. int_to_float(remainder)
    }
  }
}

// ============================================================================
// HYBRID EMOTIONAL STATES (DeepSeek R1 proposals)
// ============================================================================

/// Hybrid entropy for mixed emotional states.
///
/// H_hybrid(X) = α × H(X₁) + (1 - α) × H(X₂)
///
/// Proposed by DeepSeek R1 for modeling hybrid emotions.
/// α ∈ [0, 1] controls the blend between two emotional distributions.
///
/// ## Examples
///
/// ```gleam
/// hybrid_shannon([0.5, 0.5], [0.7, 0.3], 0.5)  // Blend of two emotions
/// ```
pub fn hybrid_shannon(
  probs1: List(Float),
  probs2: List(Float),
  alpha: Float,
) -> Float {
  let h1 = shannon(probs1)
  let h2 = shannon(probs2)
  let clamped_alpha = clamp_unit(alpha)
  clamped_alpha *. h1 +. { 1.0 -. clamped_alpha } *. h2
}

/// Clamp value to [0, 1]
fn clamp_unit(value: Float) -> Float {
  case value <. 0.0 {
    True -> 0.0
    False ->
      case value >. 1.0 {
        True -> 1.0
        False -> value
      }
  }
}

/// KL divergence sensitivity types.
///
/// Controls how sensitive the divergence is to differences.
pub type KlSensitivity {
  /// Standard KL divergence
  Standard
  /// Arousal-weighted: γ increases with arousal (sharper for high arousal)
  ArousalWeighted(arousal: Float)
  /// Custom gamma parameter
  CustomGamma(gamma: Float)
}

/// KL divergence with sensitivity parameter.
///
/// D_KL^γ(P || Q) = γ × (μ₁ - μ₂)² + D_KL(P || Q)
///
/// Proposed by DeepSeek R1 for arousal-modulated divergence.
/// Higher γ = more sensitive to mean differences.
pub fn kl_divergence_with_sensitivity(
  p: List(Float),
  q: List(Float),
  sensitivity: KlSensitivity,
) -> Result(Float, Nil) {
  // Get gamma based on sensitivity type
  let gamma = case sensitivity {
    Standard -> 0.0
    ArousalWeighted(arousal) -> {
      // γ ∝ |arousal|, scaled to [0, 1]
      let abs_arousal = case arousal >=. 0.0 {
        True -> arousal
        False -> 0.0 -. arousal
      }
      abs_arousal
    }
    CustomGamma(g) -> g
  }

  // Calculate standard KL
  case kl_divergence(p, q) {
    Error(Nil) -> Error(Nil)
    Ok(standard_kl) -> {
      // Add sensitivity term: γ × mean_diff²
      // For discrete distributions, use weighted mean difference
      let mean_diff_sq = mean_difference_squared(p, q)
      Ok(gamma *. mean_diff_sq +. standard_kl)
    }
  }
}

/// Calculate squared difference of distribution means.
fn mean_difference_squared(p: List(Float), q: List(Float)) -> Float {
  let mean_p = weighted_mean(p)
  let mean_q = weighted_mean(q)
  let diff = mean_p -. mean_q
  diff *. diff
}

/// Calculate weighted mean of a distribution.
/// Assumes indices 0, 1, 2, ... as values.
fn weighted_mean(probs: List(Float)) -> Float {
  weighted_mean_helper(probs, 0, 0.0, 0.0)
}

fn weighted_mean_helper(
  probs: List(Float),
  index: Int,
  sum: Float,
  weight_sum: Float,
) -> Float {
  case probs {
    [] ->
      case weight_sum == 0.0 {
        True -> 0.0
        False -> sum /. weight_sum
      }
    [p, ..rest] -> {
      let idx_float = int_to_float(index)
      weighted_mean_helper(
        rest,
        index + 1,
        sum +. p *. idx_float,
        weight_sum +. p,
      )
    }
  }
}

/// Renyi entropy of order α.
///
/// H_α(X) = (1/(1-α)) × log(Σ p(x)^α)
///
/// Generalizes Shannon entropy (α → 1 gives Shannon).
/// α = 0: Hartley entropy (log of support size)
/// α = 2: Collision entropy
/// α → ∞: Min-entropy
pub fn renyi(probabilities: List(Float), alpha: Float) -> Result(Float, Nil) {
  case alpha == 1.0 {
    True -> Ok(shannon(probabilities))
    False -> {
      // Σ p^α
      let sum_p_alpha =
        list.fold(probabilities, 0.0, fn(acc, p) {
          case p <=. 0.0 {
            True -> acc
            False -> acc +. power(p, alpha)
          }
        })

      case sum_p_alpha <=. 0.0 {
        True -> Error(Nil)
        False -> {
          case maths.logarithm_2(sum_p_alpha) {
            Ok(log_sum) -> Ok(log_sum /. { 1.0 -. alpha })
            Error(_) -> Error(Nil)
          }
        }
      }
    }
  }
}

/// Power function using exp(α × ln(x))
fn power(base: Float, exponent: Float) -> Float {
  case base <=. 0.0 {
    True -> 0.0
    False -> {
      case maths.natural_logarithm(base) {
        Ok(ln_base) -> maths.exponential(exponent *. ln_base)
        Error(_) -> 0.0
      }
    }
  }
}

// ============================================================================
// Tsallis & differential entropy (extensions)
// ============================================================================

/// Tsallis entropy S_q(X) = (1 - Σ pᵢ^q) / (q - 1).
///
/// Generalises Shannon (q → 1) and Rényi. q < 1 emphasises rare events,
/// q > 1 emphasises modes. Used in non-extensive statistical mechanics and
/// to model heavy-tailed emotional distributions.
pub fn tsallis(probabilities: List(Float), q: Float) -> Result(Float, Nil) {
  // Fuzzy compare around q == 1 to avoid catastrophic cancellation:
  // S_q is continuous at q=1 with limit equal to Shannon entropy.
  let q_close_to_1 = case q -. 1.0 {
    d if d <. 0.0 -> 0.0 -. d <. 1.0e-9
    d -> d <. 1.0e-9
  }
  case q_close_to_1 {
    True -> Ok(shannon(probabilities))
    False -> {
      let sum_p_q =
        list.fold(probabilities, 0.0, fn(acc, p) {
          case p <=. 0.0 {
            True -> acc
            False -> acc +. power(p, q)
          }
        })
      Ok({ 1.0 -. sum_p_q } /. { q -. 1.0 })
    }
  }
}

/// Fisher information for a Gaussian distribution: I(σ) = 1 / σ².
///
/// Measures how much information an observation carries about the mean.
pub fn fisher_information_gaussian(sigma: Float) -> Result(Float, Nil) {
  case sigma <=. 0.0 {
    True -> Error(Nil)
    False -> Ok(1.0 /. { sigma *. sigma })
  }
}

/// Differential entropy of a Gaussian: h(N(μ, σ²)) = ½ ln(2πeσ²).
pub fn differential_entropy_gaussian(sigma: Float) -> Result(Float, Nil) {
  case sigma <=. 0.0 {
    True -> Error(Nil)
    False -> {
      // 2πe ≈ 17.07946844534713
      let two_pi_e = 17.07946844534713
      case maths.natural_logarithm(two_pi_e *. sigma *. sigma) {
        Ok(ln_val) -> Ok(0.5 *. ln_val)
        Error(_) -> Error(Nil)
      }
    }
  }
}

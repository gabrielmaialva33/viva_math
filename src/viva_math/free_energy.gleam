//// Free Energy Principle (FEP) calculations.
////
//// Based on Karl Friston's work (2010, 2019).
//// Free Energy bounds surprise (negative log evidence) and can be decomposed as:
////
//// F = Π · (μ - o)² + D_KL(q || p)
////     ↑              ↑
////     Accuracy       Complexity
////     (weighted      (deviation
////     prediction     from priors)
////     error)
////
//// In VIVA, this is used for interoception - sensing internal state
//// and minimizing "surprise" through prediction.
////
//// References:
//// - Friston (2010) "The free-energy principle: a unified brain theory?"
//// - Parr & Friston (2019) "Generalised free energy and active inference"
//// - Validated by DeepSeek R1 671B (2025)

import gleam/float
import gleam/list
import gleam/result
import viva_math/constants
import viva_math/scalar
import viva_math/vector.{type Vec3}

/// Free Energy state for a system.
pub type FreeEnergyState {
  FreeEnergyState(
    /// The free energy value (lower is better)
    free_energy: Float,
    /// Prediction error component (precision-weighted)
    prediction_error: Float,
    /// Complexity/KL divergence component
    complexity: Float,
    /// Precision used for weighting
    precision: Float,
    /// Qualitative feeling based on normalized free energy
    feeling: Feeling,
  )
}

/// Qualitative feeling based on free energy level.
pub type Feeling {
  /// Low free energy - predictions match reality (F < μ - σ)
  Homeostatic
  /// Moderate free energy - slight mismatch (μ - σ ≤ F < μ)
  Surprised
  /// High free energy - significant mismatch (μ ≤ F < μ + σ)
  Alarmed
  /// Very high free energy - system overwhelmed (F ≥ μ + σ)
  Overwhelmed
}

/// Thresholds for feeling classification.
/// Based on system-specific statistics (mean and standard deviation).
pub type FeelingThresholds {
  FeelingThresholds(
    /// Mean free energy (baseline)
    mean: Float,
    /// Standard deviation of free energy
    std_dev: Float,
  )
}

/// Default thresholds calibrated for PAD space.
/// Mean and std_dev derived from typical emotional dynamics.
pub fn default_thresholds() -> FeelingThresholds {
  FeelingThresholds(mean: 0.5, std_dev: 0.3)
}

/// Compute raw prediction error between expected and actual state.
/// Uses squared Euclidean distance (L2 loss).
pub fn prediction_error(expected: Vec3, actual: Vec3) -> Float {
  vector.distance_squared(expected, actual)
}

/// Compute precision-weighted prediction error.
///
/// F_accuracy = Π · (expected - actual)²
///
/// Precision (Π) = 1/variance. Higher precision = more weight on prediction errors.
/// This is critical for biological systems where uncertainty should attenuate errors.
pub fn precision_weighted_prediction_error(
  expected: Vec3,
  actual: Vec3,
  precision: Float,
) -> Float {
  let pe = prediction_error(expected, actual)
  precision *. pe
}

/// Compute KL divergence between Gaussian distributions (closed form).
///
/// CORRECTED per DeepSeek R1 validation - Full KL for Gaussians:
/// D_KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = (μ₁ - μ₂)²/(2σ₂²) + (σ₁² - σ₂²)/(2σ₂²) - 1/2
///
/// When variances are equal (σ₁ = σ₂), reduces to: (μ₁ - μ₂)²/(2σ²)
///
/// This measures how much the posterior (current belief) diverges from prior.
pub fn gaussian_kl_divergence(
  posterior_mean: Vec3,
  prior_mean: Vec3,
  variance: Float,
) -> Float {
  let diff_squared = vector.distance_squared(posterior_mean, prior_mean)
  case variance <=. 0.0 {
    True -> 0.0
    False -> diff_squared /. { 2.0 *. variance }
  }
}

/// Full KL divergence between multivariate isotropic Gaussians with
/// different (scalar) variances in d=3 dimensions (Vec3).
///
/// D_KL(N(μ₁, σ₁² I_d) || N(μ₂, σ₂² I_d))
///   = (d/2) · ln(σ₂² / σ₁²) + (d·σ₁² + |μ₁-μ₂|²) / (2σ₂²) - d/2
///
/// For d=1 and equal variances this reduces to `(μ₁-μ₂)² / (2σ²)`, matching
/// `gaussian_kl_divergence/3`.
pub fn gaussian_kl_divergence_full(
  posterior_mean: Vec3,
  prior_mean: Vec3,
  posterior_variance: Float,
  prior_variance: Float,
) -> Float {
  case posterior_variance <=. 0.0 || prior_variance <=. 0.0 {
    True -> 0.0
    False -> {
      let d = 3.0
      let diff_squared = vector.distance_squared(posterior_mean, prior_mean)

      // 0.5 · d · (ln σ₂² - ln σ₁²) — separate logs for stability.
      let log_term = case
        scalar.try_ln(prior_variance),
        scalar.try_ln(posterior_variance)
      {
        Ok(log_prior), Ok(log_posterior) ->
          0.5 *. d *. { log_prior -. log_posterior }
        _, _ -> 0.0
      }

      // (d·σ₁² + |μ₁-μ₂|²) / (2σ₂²)
      let ratio_term =
        { d *. posterior_variance +. diff_squared } /. { 2.0 *. prior_variance }

      log_term +. ratio_term -. d /. 2.0
    }
  }
}

/// Compute complexity term using KL divergence.
///
/// Complexity = D_KL(q(θ) || p(θ))
///
/// Where q is posterior belief and p is prior belief (homeostatic setpoint).
/// Weight controls the regularization strength.
pub fn complexity(
  current: Vec3,
  baseline: Vec3,
  prior_variance: Float,
) -> Float {
  gaussian_kl_divergence(current, baseline, prior_variance)
}

/// Legacy complexity function for backwards compatibility.
pub fn complexity_weighted(
  current: Vec3,
  baseline: Vec3,
  weight: Float,
) -> Float {
  weight *. vector.distance_squared(current, baseline)
}

/// Compute full Free Energy: F = Π·(μ-o)² + D_KL(q||p)
///
/// ## Parameters
/// - expected: predicted/expected state (μ)
/// - actual: observed/actual state (o)
/// - baseline: prior baseline state (p) - e.g., personality/homeostatic setpoint
/// - precision: inverse variance of predictions (Π)
/// - prior_variance: variance of prior beliefs (for KL term)
pub fn free_energy(
  expected: Vec3,
  actual: Vec3,
  baseline: Vec3,
  precision: Float,
  prior_variance: Float,
) -> Float {
  let accuracy =
    precision_weighted_prediction_error(expected, actual, precision)
  let cx = complexity(actual, baseline, prior_variance)
  accuracy +. cx
}

/// Compute free energy and return full state with feeling.
/// Uses normalized thresholds for feeling classification.
pub fn compute_state(
  expected: Vec3,
  actual: Vec3,
  baseline: Vec3,
  precision: Float,
  prior_variance: Float,
  thresholds: FeelingThresholds,
) -> FreeEnergyState {
  let accuracy =
    precision_weighted_prediction_error(expected, actual, precision)
  let cx = complexity(actual, baseline, prior_variance)
  let fe = accuracy +. cx

  FreeEnergyState(
    free_energy: fe,
    prediction_error: accuracy,
    complexity: cx,
    precision: precision,
    feeling: classify_feeling_normalized(fe, thresholds),
  )
}

/// Simplified compute_state with default thresholds and legacy interface.
/// For backwards compatibility.
pub fn compute_state_simple(
  expected: Vec3,
  actual: Vec3,
  baseline: Vec3,
  complexity_weight: Float,
) -> FreeEnergyState {
  let pe = prediction_error(expected, actual)
  let cx = complexity_weighted(actual, baseline, complexity_weight)
  let fe = pe +. cx

  FreeEnergyState(
    free_energy: fe,
    prediction_error: pe,
    complexity: cx,
    precision: 1.0,
    feeling: classify_feeling(fe),
  )
}

/// Classify feeling using normalized thresholds.
///
/// - Homeostatic: F < μ - σ (better than expected)
/// - Surprised: μ - σ ≤ F < μ (slightly worse)
/// - Alarmed: μ ≤ F < μ + σ (worse than average)
/// - Overwhelmed: F ≥ μ + σ (much worse)
pub fn classify_feeling_normalized(
  free_energy: Float,
  thresholds: FeelingThresholds,
) -> Feeling {
  let lower = thresholds.mean -. thresholds.std_dev
  let upper = thresholds.mean +. thresholds.std_dev

  case free_energy {
    fe if fe <. lower -> Homeostatic
    fe if fe <. thresholds.mean -> Surprised
    fe if fe <. upper -> Alarmed
    _ -> Overwhelmed
  }
}

/// Legacy classify_feeling with fixed thresholds.
/// Calibrated for PAD space (max distance ~3.46).
pub fn classify_feeling(free_energy: Float) -> Feeling {
  case free_energy {
    fe if fe <. 0.1 -> Homeostatic
    fe if fe <. 0.5 -> Surprised
    fe if fe <. 1.5 -> Alarmed
    _ -> Overwhelmed
  }
}

/// Update thresholds based on observed free energy history.
/// Uses exponential moving average for online learning.
pub fn update_thresholds(
  current: FeelingThresholds,
  observed_fe: Float,
  alpha: Float,
) -> FeelingThresholds {
  // EMA update for mean
  let new_mean = alpha *. observed_fe +. { 1.0 -. alpha } *. current.mean

  // Update variance estimate
  let diff = observed_fe -. current.mean
  let new_var =
    alpha
    *. { diff *. diff }
    +. { 1.0 -. alpha }
    *. { current.std_dev *. current.std_dev }

  // Convert variance back to std_dev (sqrt = nth_root with n=2)
  let new_std = case float.square_root(new_var) {
    Ok(s) -> s
    Error(_) -> current.std_dev
  }

  FeelingThresholds(mean: new_mean, std_dev: float.max(new_std, 0.01))
}

/// Compute surprise for a single dimension.
///
/// Surprise = -log(p(observation | model))
/// Using Gaussian approximation: surprise ∝ (x - μ)² / (2σ²)
pub fn surprise(expected: Float, observed: Float, sigma: Float) -> Float {
  let diff = observed -. expected
  let sigma_sq = sigma *. sigma
  case sigma_sq <=. 0.0 {
    True -> 0.0
    False -> { diff *. diff } /. { 2.0 *. sigma_sq }
  }
}

/// Active Inference: compute action that minimizes expected free energy.
///
/// This returns the delta to apply to current state to move toward target.
/// Rate controls how quickly to move (0 = no movement, 1 = instant).
pub fn active_inference_delta(
  current: Vec3,
  target: Vec3,
  rate: Float,
) -> Vec3 {
  let diff = vector.sub(target, current)
  vector.scale(diff, rate)
}

/// Precision-weighted prediction error for Vec3.
///
/// Each dimension can have different precision.
/// Returns weighted sum of squared errors.
pub fn precision_weighted_error_vec(
  expected: Vec3,
  actual: Vec3,
  precisions: Vec3,
) -> Float {
  let diff = vector.sub(expected, actual)
  let diff_sq = vector.multiply(diff, diff)
  let weighted = vector.multiply(diff_sq, precisions)
  vector.sum(weighted)
}

/// Estimate precision from recent prediction errors.
///
/// Precision = 1 / variance of errors
/// Higher precision means more reliable predictions.
pub fn estimate_precision(errors: List(Float)) -> Float {
  case list.length(errors) {
    0 -> 1.0
    1 -> 1.0
    n -> {
      let n_float = int_to_float(n)
      let mean = list.fold(errors, 0.0, fn(acc, e) { acc +. e }) /. n_float
      let variance =
        list.fold(errors, 0.0, fn(acc, e) {
          let diff = e -. mean
          acc +. diff *. diff
        })
        /. n_float

      case variance <. 0.001 {
        True -> 100.0
        // Very precise
        False -> 1.0 /. variance
      }
    }
  }
}

/// Bayesian belief update: combine prior with likelihood.
///
/// posterior ∝ likelihood × prior
/// Using precision-weighted combination:
/// new_belief = (Π_prior × prior + Π_likelihood × observation) /
///              (Π_prior + Π_likelihood)
pub fn belief_update(
  prior: Float,
  observation: Float,
  precision_prior: Float,
  precision_likelihood: Float,
) -> Float {
  let total_precision = precision_prior +. precision_likelihood
  case total_precision <=. 0.0 {
    True -> prior
    False ->
      { precision_prior *. prior +. precision_likelihood *. observation }
      /. total_precision
  }
}

/// Generalized Free Energy (expected free energy for planning).
///
/// G = ambiguity + risk
/// - ambiguity: expected surprise under model (epistemic value)
/// - risk: KL divergence from preferred outcomes (pragmatic value)
///
/// Used for action selection in active inference.
pub fn generalized_free_energy(
  expected_state: Vec3,
  preferred_state: Vec3,
  uncertainty: Float,
) -> Float {
  let ambiguity = uncertainty
  let risk = vector.distance_squared(expected_state, preferred_state)
  ambiguity +. risk
}

/// Variational Free Energy bound.
///
/// F ≤ -log p(o) + D_KL(q||p)
///
/// The free energy bounds the negative log evidence (surprise).
pub fn variational_bound(
  observation_likelihood: Float,
  kl_divergence: Float,
) -> Float {
  let neg_log_likelihood = case observation_likelihood <=. 0.0 {
    True -> 100.0
    // Large surprise for impossible observations
    False ->
      case scalar.try_ln(observation_likelihood) {
        Ok(log_l) -> 0.0 -. log_l
        Error(_) -> 100.0
      }
  }
  neg_log_likelihood +. kl_divergence
}

// ============================================================================
// Expected Free Energy (Active Inference for planning)
// ============================================================================

/// Expected Free Energy components.
///
/// In planning, an agent selects actions that minimise G = epistemic + pragmatic.
/// Splitting the components lets you steer exploration (epistemic) vs
/// exploitation (pragmatic) by reweighting them.
pub type ExpectedFreeEnergy {
  ExpectedFreeEnergy(
    /// Information gain from observing the outcome (exploration).
    epistemic: Float,
    /// Expected divergence from preferred outcomes (exploitation).
    pragmatic: Float,
    /// G = epistemic + pragmatic.
    total: Float,
  )
}

/// Decompose Expected Free Energy.
///
/// - `predicted_outcome`: agent's expectation of the future state under action a.
/// - `preferred_outcome`: agent's goal state (homeostatic setpoint).
/// - `predictive_uncertainty`: entropy of the predictive distribution (epistemic).
pub fn expected_free_energy(
  predicted_outcome: Vec3,
  preferred_outcome: Vec3,
  predictive_uncertainty: Float,
) -> ExpectedFreeEnergy {
  let epistemic = predictive_uncertainty
  let pragmatic = vector.distance_squared(predicted_outcome, preferred_outcome)
  ExpectedFreeEnergy(
    epistemic: epistemic,
    pragmatic: pragmatic,
    total: epistemic +. pragmatic,
  )
}

/// Select the action with minimum Expected Free Energy.
///
/// `policies` is a list of `(action_label, predicted_outcome, predictive_uncertainty)`.
/// Returns the best policy or `Error(Nil)` if the list is empty.
pub fn select_policy(
  policies: List(#(a, Vec3, Float)),
  preferred_outcome: Vec3,
) -> Result(#(a, ExpectedFreeEnergy), Nil) {
  case policies {
    [] -> Error(Nil)
    [first, ..rest] -> {
      let #(label, outcome, uncertainty) = first
      let g = expected_free_energy(outcome, preferred_outcome, uncertainty)
      let initial = #(label, g)
      Ok(
        list.fold(rest, initial, fn(acc, p) {
          let #(p_label, p_outcome, p_unc) = p
          let p_g = expected_free_energy(p_outcome, preferred_outcome, p_unc)
          case p_g.total <. acc.1.total {
            True -> #(p_label, p_g)
            False -> acc
          }
        }),
      )
    }
  }
}

/// Softmax over policies: probability of selecting each action given its
/// Expected Free Energy. Lower G → higher probability (β controls sharpness).
pub fn policy_posterior(
  policies: List(#(a, Vec3, Float)),
  preferred_outcome: Vec3,
  beta: Float,
) -> List(#(a, Float)) {
  case policies {
    [] -> []
    [first, ..rest] -> {
      let to_logit = fn(triple) {
        let #(label, outcome, uncertainty) = triple
        let g = expected_free_energy(outcome, preferred_outcome, uncertainty)
        #(label, 0.0 -. beta *. g.total)
      }
      let first_logit = to_logit(first)
      let rest_logits = list.map(rest, to_logit)
      let gs = [first_logit, ..rest_logits]

      // Initialise max with the first logit to avoid arbitrary sentinels.
      let max_g =
        list.fold(rest_logits, first_logit.1, fn(acc, pair) {
          case pair.1 >. acc {
            True -> pair.1
            False -> acc
          }
        })
      let exps =
        list.map(gs, fn(pair) { #(pair.0, scalar.exp(pair.1 -. max_g)) })
      let total = list.fold(exps, 0.0, fn(acc, pair) { acc +. pair.1 })
      case total == 0.0 {
        True -> list.map(gs, fn(pair) { #(pair.0, 0.0) })
        False -> list.map(exps, fn(pair) { #(pair.0, pair.1 /. total) })
      }
    }
  }
}

// ============================================================================
// Hierarchical predictive coding (Meta-PCN / S-HAI 2025-2026)
// ============================================================================

/// A single layer of a hierarchical predictive-coding network.
///
/// Stores the layer's state estimate `mu`, the precision (inverse variance)
/// of its prediction errors, and the precision of the prior over the layer
/// state. Higher layers send top-down predictions; bottom-up prediction
/// errors travel upward. See Friston (2010), Bogacz (2017), and the 2026
/// Meta-PCN framework for the modern formulation.
pub type HierarchicalLayer {
  HierarchicalLayer(
    /// Posterior mean at this layer (the latent state estimate).
    mu: Vec3,
    /// Precision of prediction errors flowing up from this layer.
    precision: Float,
    /// Precision of the prior over this layer's state.
    prior_precision: Float,
  )
}

/// A hierarchical predictive coding network: a stack of layers from
/// sensory (head) to abstract (tail). Used for active inference planning
/// at multiple scales (S-HAI 2026).
pub type Hierarchical {
  Hierarchical(layers: List(HierarchicalLayer))
}

/// Per-layer prediction error: e_l = mu_l - g(mu_{l+1}).
///
/// In the simplest linear PC model, `g` is the identity. For richer models
/// pass a custom decoder via `hierarchical_errors_with`.
pub fn hierarchical_errors(h: Hierarchical) -> List(Vec3) {
  hierarchical_errors_with(h, fn(top_down) { top_down })
}

/// Hierarchical prediction errors with custom top-down decoder.
pub fn hierarchical_errors_with(
  h: Hierarchical,
  decoder: fn(Vec3) -> Vec3,
) -> List(Vec3) {
  case h.layers {
    [] -> []
    [_] -> []
    _ -> errors_pairs(h.layers, decoder, [])
  }
}

fn errors_pairs(
  layers: List(HierarchicalLayer),
  decoder: fn(Vec3) -> Vec3,
  acc: List(Vec3),
) -> List(Vec3) {
  case layers {
    [lower, upper, ..rest] -> {
      let prediction = decoder(upper.mu)
      let err = vector.sub(lower.mu, prediction)
      errors_pairs([upper, ..rest], decoder, [err, ..acc])
    }
    _ -> list.reverse(acc)
  }
}

/// Hierarchical free energy summed across layers.
///
/// F_total = Σ_l Π_l · |e_l|² where e_l is the prediction error between
/// layer l and the top-down prediction from layer l+1. This is the variant
/// Meta-PCN (ICLR 2026) regularises with weight-variance normalisation to
/// avoid exploding errors in deep networks.
pub fn hierarchical_free_energy(h: Hierarchical) -> Float {
  let errors = hierarchical_errors(h)
  let layers = case h.layers {
    [_, ..rest] -> rest
    [] -> []
  }
  list.zip(layers, errors)
  |> list.fold(0.0, fn(acc, pair) {
    let #(upper, err) = pair
    acc +. upper.precision *. vector.dot(err, err)
  })
}

/// Meta-prediction error: prediction error of the prediction error.
///
/// Meta-PCN (Lin et al. ICLR 2026) shows that minimising "PEs of PEs"
/// linearises the otherwise non-linear PCN equilibrium dynamics, yielding
/// dramatically more stable inference at depth.
///
/// meta_e_l = e_l - h(e_{l+1})  where h is typically identity for the
/// simplest case.
pub fn meta_prediction_errors(h: Hierarchical) -> List(Vec3) {
  let errors = hierarchical_errors(h)
  case errors {
    [] -> []
    [_] -> errors
    _ -> meta_pairs(errors, [])
  }
}

fn meta_pairs(errors: List(Vec3), acc: List(Vec3)) -> List(Vec3) {
  case errors {
    [lower, upper, ..rest] -> {
      let meta = vector.sub(lower, upper)
      meta_pairs([upper, ..rest], [meta, ..acc])
    }
    _ -> list.reverse(acc)
  }
}

/// One inference step of gradient descent on the hierarchical free energy.
///
/// For each non-top layer l, updates the latent state μ_l along the descent
/// direction `-∂F/∂μ_l`, where:
///   ∂F/∂μ_l = Π_l · (μ_l - μ_{l+1}) + Π_{l-1} · (μ_l - μ_{l-1})
///                       ↑                            ↑
///              top-down prior fit              bottom-up evidence fit
///
/// `lr` is the learning rate (step size); typical values 0.01–0.1 for stable
/// inference. The bottom layer's μ is left untouched — it represents the
/// sensory observation and is fixed during inference.
pub fn hierarchical_inference_step(h: Hierarchical, lr: Float) -> Hierarchical {
  case h.layers {
    [] -> h
    [_] -> h
    _ -> {
      let updated = inference_walk(h.layers, lr, [])
      Hierarchical(layers: updated)
    }
  }
}

fn inference_walk(
  layers: List(HierarchicalLayer),
  lr: Float,
  acc: List(HierarchicalLayer),
) -> List(HierarchicalLayer) {
  case layers {
    [] -> list.reverse(acc)
    [single] -> list.reverse([single, ..acc])
    [lower, upper, ..rest] -> {
      // For the first (sensory) layer we keep μ fixed.
      case acc {
        [] -> inference_walk([upper, ..rest], lr, [lower])
        _ -> {
          let prev = case acc {
            [p, ..] -> p
            [] -> lower
          }
          let bottom_up = vector.sub(lower.mu, prev.mu)
          let top_down = vector.sub(upper.mu, lower.mu)
          let grad =
            vector.add(
              vector.scale(bottom_up, prev.precision),
              vector.scale(top_down, 0.0 -. lower.precision),
            )
          let new_mu = vector.sub(lower.mu, vector.scale(grad, lr))
          let new_layer = HierarchicalLayer(..lower, mu: new_mu)
          inference_walk([upper, ..rest], lr, [new_layer, ..acc])
        }
      }
    }
  }
}

/// Run `n` inference steps. Convenience wrapper around
/// `hierarchical_inference_step`.
pub fn hierarchical_infer(h: Hierarchical, lr: Float, n: Int) -> Hierarchical {
  case n <= 0 {
    True -> h
    False -> hierarchical_infer(hierarchical_inference_step(h, lr), lr, n - 1)
  }
}

// ============================================================================
// Bayesian Predictive Coding (BPC) — closed-form weight update
// ============================================================================

/// Posterior over a Gaussian belief: mean and precision (inverse variance).
///
/// BPC tracks the full posterior over hidden states instead of just MAP
/// estimates. Closed-form Hebbian updates (Vasilescu & Friston 2025,
/// arXiv:2503.24016) preserve the locality of PC while quantifying
/// epistemic uncertainty.
pub type GaussianBelief {
  GaussianBelief(mean: Vec3, precision: Float)
}

/// Precision-weighted Bayesian update for a Gaussian belief from a single
/// observation under Gaussian likelihood.
///
/// posterior_precision = prior_precision + likelihood_precision
/// posterior_mean = (prior_precision · prior_mean +
///                   likelihood_precision · observation) / posterior_precision
///
/// Returns the new belief. This is the closed-form variant central to BPC.
pub fn bpc_update(
  prior: GaussianBelief,
  observation: Vec3,
  likelihood_precision: Float,
) -> GaussianBelief {
  let total = prior.precision +. likelihood_precision
  case total <=. 0.0 {
    True -> prior
    False -> {
      let weighted_prior = vector.scale(prior.mean, prior.precision)
      let weighted_obs = vector.scale(observation, likelihood_precision)
      let new_mean =
        vector.scale(vector.add(weighted_prior, weighted_obs), 1.0 /. total)
      GaussianBelief(mean: new_mean, precision: total)
    }
  }
}

/// Hebbian variance update: the BPC weight-rule equivalent of synaptic
/// plasticity. Updates precision based on prediction-error magnitude.
///
/// new_precision = (count · prior_precision + 1) /
///                 (count · variance + |error|²)
///
/// Higher errors → lower precision; consistent observations → higher
/// precision. Equivalent to a conjugate Normal-Gamma update.
pub fn bpc_precision_update(
  current_precision: Float,
  error_squared: Float,
  observation_count: Int,
) -> Float {
  let n = int_to_float(observation_count)
  case observation_count {
    0 -> current_precision
    _ -> {
      let variance_estimate = case current_precision <=. 0.0 {
        True -> 1.0
        False -> 1.0 /. current_precision
      }
      let num = n *. current_precision +. 1.0
      let denom = n *. variance_estimate +. error_squared
      case denom <=. 0.0 {
        True -> current_precision
        False -> num /. denom
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
    3 -> 3.0
    4 -> 4.0
    5 -> 5.0
    _ -> {
      let half = n / 2
      let remainder = n - half * 2
      int_to_float(half) *. 2.0 +. int_to_float(remainder)
    }
  }
}

// ============================================================================
// Variational Inference — Bayesian deepening of the FEP
// ============================================================================
//
// References:
// - Beal (2003) "Variational Algorithms for Approximate Bayesian Inference"
// - Bishop (2006) "Pattern Recognition and Machine Learning", ch. 10
// - Friston (2010) "The free-energy principle"
//
// All closed forms below assume **conjugate Gaussian** models with known
// variances. This is the standard textbook regime (Bishop §10.1, §10.7) and
// suffices for affective inference where each PAD axis is treated as a scalar
// latent under Gaussian assumptions.
//

/// Decomposition of the Evidence Lower Bound.
///
/// ```
/// ELBO(q) = E_q[log p(x | z)] − D_KL(q(z) ‖ p(z))
///         = reconstruction − kl_divergence
/// ```
///
/// Maximizing ELBO is equivalent to minimizing the variational free energy:
/// `F = −ELBO`.
pub type ELBO {
  ELBO(
    /// Expected log-likelihood under q. Higher = better data fit.
    reconstruction: Float,
    /// Divergence of posterior approximation from prior.
    kl_divergence: Float,
    /// `reconstruction − kl_divergence`. Lower bound on `log p(x)`.
    total: Float,
  )
}

/// Mean-field Gaussian variational posterior `q(z) ~ N(q_mean, q_var)`.
pub type MeanFieldParams {
  MeanFieldParams(q_mean: Float, q_var: Float)
}

/// Closed-form ELBO for a Gaussian latent model:
///
/// - Prior:      `p(z)   = N(prior_mean, prior_var)`
/// - Likelihood: `p(x|z) = N(z, likelihood_var)`
/// - Posterior approx: `q(z) = N(q_mean, q_var)`
///
/// Reconstruction term (expected log-likelihood under q):
///
/// `E_q[log p(x|z)] = −½·log(2π·likelihood_var)
///                   − ((x − q_mean)² + q_var) / (2·likelihood_var)`
///
/// KL term is `gaussian_kl_divergence_full(q_mean, q_var, prior_mean, prior_var)`.
pub fn elbo(
  observation: Float,
  q_mean: Float,
  q_var: Float,
  prior_mean: Float,
  prior_var: Float,
  likelihood_var: Float,
) -> ELBO {
  let kl = scalar_gaussian_kl(q_mean, q_var, prior_mean, prior_var)
  let recon = case likelihood_var <=. 0.0 {
    True -> 0.0 -. large_penalty()
    False -> {
      let log_2pi_var =
        scalar.try_ln(2.0 *. constants.pi *. likelihood_var)
        |> result.unwrap(0.0)
      let err2 = { observation -. q_mean } *. { observation -. q_mean }
      let term = { err2 +. q_var } /. { 2.0 *. likelihood_var }
      0.0 -. 0.5 *. log_2pi_var -. term
    }
  }
  ELBO(reconstruction: recon, kl_divergence: kl, total: recon -. kl)
}

/// 1D Gaussian KL — `D_KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²))`.
///
/// `= ½ · ( ln(σ₂² / σ₁²) + (σ₁² + (μ₁−μ₂)²) / σ₂² − 1 )`.
///
/// Returns `0.0` if either variance is non-positive. Public so callers can
/// reuse it (the existing `gaussian_kl_divergence_*` functions are Vec3-only).
pub fn scalar_gaussian_kl(
  q_mean: Float,
  q_var: Float,
  p_mean: Float,
  p_var: Float,
) -> Float {
  case q_var <=. 0.0 || p_var <=. 0.0 {
    True -> 0.0
    False -> {
      let log_term = case scalar.try_ln(p_var), scalar.try_ln(q_var) {
        Ok(lp), Ok(lq) -> lp -. lq
        _, _ -> 0.0
      }
      let diff = q_mean -. p_mean
      let frac = { q_var +. diff *. diff } /. p_var
      0.5 *. { log_term +. frac -. 1.0 }
    }
  }
}

/// Mean-field update under a Gaussian prior and Gaussian likelihood with
/// known variances. **Closed form** — no iteration needed (the model is
/// conjugate). Bishop §2.3.3.
///
/// ```
/// posterior_precision = prior_precision + n · likelihood_precision
/// posterior_mean      = posterior_var · (prior_precision · prior_mean
///                                      + likelihood_precision · sum_x)
/// ```
///
/// Returns `Error(Nil)` if either variance is non-positive.
pub fn mean_field_update(
  observations: List(Float),
  prior_mean: Float,
  prior_var: Float,
  likelihood_var: Float,
) -> Result(MeanFieldParams, Nil) {
  case prior_var >. 0.0 && likelihood_var >. 0.0 {
    False -> Error(Nil)
    True -> {
      let n = int_to_float(list.length(observations))
      let sum_x = list.fold(observations, 0.0, fn(acc, x) { acc +. x })
      let prior_prec = 1.0 /. prior_var
      let lik_prec = 1.0 /. likelihood_var
      let post_prec = prior_prec +. n *. lik_prec
      let post_var = 1.0 /. post_prec
      let post_mean =
        post_var *. { prior_prec *. prior_mean +. lik_prec *. sum_x }
      Ok(MeanFieldParams(q_mean: post_mean, q_var: post_var))
    }
  }
}

/// Iterated mean-field — for non-conjugate models the update would be
/// re-applied with refreshed likelihood statistics. Here, since the conjugate
/// case is one-shot, this iterates over a *sequence* of observation batches,
/// using each posterior as the next prior (sequential Bayes).
///
/// Returns the final `MeanFieldParams` after consuming all batches.
pub fn mean_field_iterate(
  batches: List(List(Float)),
  prior: MeanFieldParams,
  likelihood_var: Float,
) -> Result(MeanFieldParams, Nil) {
  case batches {
    [] -> Ok(prior)
    [batch, ..rest] ->
      case mean_field_update(batch, prior.q_mean, prior.q_var, likelihood_var) {
        Error(_) -> Error(Nil)
        Ok(next) -> mean_field_iterate(rest, next, likelihood_var)
      }
  }
}

/// Laplace approximation: fit a Gaussian to the mode of a smooth
/// log-posterior. The mean is the MAP estimate (found by gradient ascent),
/// the variance is `−1 / f''(mode)` via central finite differences.
///
/// `log_posterior` should return `log p(z | x)` up to an additive constant
/// (the normaliser cancels in the gradient).
///
/// - `step_size` — gradient step for finding the mode.
/// - `n_steps`   — gradient iterations. Returns `Error(Nil)` if the second
///   derivative at the mode is non-negative (no valid Gaussian fit).
pub fn laplace_approximation(
  log_posterior: fn(Float) -> Float,
  initial_guess: Float,
  step_size: Float,
  n_steps: Int,
) -> Result(MeanFieldParams, Nil) {
  let h = 1.0e-4
  let mode = laplace_ascent(log_posterior, initial_guess, step_size, n_steps, h)
  // Second derivative via central differences:
  //   f''(z) ≈ (f(z+h) − 2·f(z) + f(z−h)) / h²
  let f_plus = log_posterior(mode +. h)
  let f_zero = log_posterior(mode)
  let f_minus = log_posterior(mode -. h)
  let second = { f_plus -. 2.0 *. f_zero +. f_minus } /. { h *. h }
  case second <. 0.0 {
    False -> Error(Nil)
    True -> {
      let q_var = 0.0 -. 1.0 /. second
      Ok(MeanFieldParams(q_mean: mode, q_var: q_var))
    }
  }
}

fn laplace_ascent(
  f: fn(Float) -> Float,
  z: Float,
  step: Float,
  n: Int,
  h: Float,
) -> Float {
  case n <= 0 {
    True -> z
    False -> {
      let grad = { f(z +. h) -. f(z -. h) } /. { 2.0 *. h }
      laplace_ascent(f, z +. step *. grad, step, n - 1, h)
    }
  }
}

/// Log marginal likelihood (model evidence) for the Gaussian-Gaussian model.
///
/// Marginal: `p(x) = ∫ p(x|z) p(z) dz = N(x; prior_mean, prior_var + likelihood_var)`.
///
/// Returns `log p(x)` directly (closed form). Useful as the gold-standard
/// reference value that ELBO bounds from below.
pub fn log_evidence_gaussian(
  observation: Float,
  prior_mean: Float,
  prior_var: Float,
  likelihood_var: Float,
) -> Float {
  let marginal_var = prior_var +. likelihood_var
  case marginal_var <=. 0.0 {
    True -> 0.0 -. large_penalty()
    False -> {
      let err = observation -. prior_mean
      let log_2pi_var =
        scalar.try_ln(2.0 *. constants.pi *. marginal_var)
        |> result.unwrap(0.0)
      0.0 -. 0.5 *. log_2pi_var -. err *. err /. { 2.0 *. marginal_var }
    }
  }
}

fn large_penalty() -> Float {
  1.0e6
}

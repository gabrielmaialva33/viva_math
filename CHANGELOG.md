# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.100] - 2026-05-20

Scientific computing milestone. The library grew from 7 to 22 modules with
compensated summation, autodiff (forward and reverse), symplectic
integrators, FFT, quaternions, complex numbers, t-digest streaming
quantiles, a binary dense matrix backend, and hierarchical predictive
coding aligned with 2025-2026 papers.

### Added — High-precision numerics

- **viva_math/precision** — Compensated summation kernel
  - `neumaier_sum` (default, CPython 3.12-style)
  - `kahan_sum`, `pairwise_sum`, `fsum` (Shewchuk round-once exact)
  - `two_sum` exact `(hi, lo)` decomposition
  - Pébay `Moments` accumulator (M₂/M₃/M₄ online) with `moments_combine`
    for parallel reductions

### Added — Activations and SOTA 2026 papers

- **viva_math/scalar**
  - `erf`, `erfc`, `fmod` via `:math` BIFs
  - GELU exact + tanh approximation; `lambda_gelu` (Cantos & Aragón 2026)
  - `iglu` + rational `iglu_approx` (Aragón et al. 2026)
  - `silu`, `swish`, `mish`, `softplus`, `hard_sigmoid`, `selu`
  - `logsumexp` (Neumaier-backed), `logaddexp` (no NaN at ±∞), `hypot`

### Added — Automatic differentiation

- **viva_math/autodiff** — Forward-mode AD via dual numbers
  - Scalar `Dual` and 3-D `Dual3` for PAD-space gradients
  - `grad`, `value_and_grad`, `gradient3`, generic `jacobian`
- **viva_math/autodiff_reverse** — Reverse-mode AD with computation tape
  - O(1) gradient cost in input dimension
  - Public `Tape`, `Op`, `Node` for introspection

### Added — ODE and dynamical systems

- **viva_math/ode**
  - Dormand-Prince 5(4) (`dop54`) — single-step err ~3·10⁻¹⁰
  - Symplectic integrators: `velocity_verlet`, `leapfrog`,
    `position_verlet`, `yoshida4` (order 4)
  - `integrate_symplectic` trajectory builder
- **viva_math/scheduler** — Cosine annealing, linear warmup, one-cycle,
  inverse-sqrt, triangle wave, exponential
- **viva_math/calculus** — Forward/central/5-point/second-derivative
  finite differences, gradient, trapezoid/Simpson/Romberg quadrature

### Added — Linear algebra

- **viva_math/matrix**
  - `mat2_eigenvalues` (characteristic quadratic)
  - `mat3_symmetric_eigenvalues` (Smith 1961 trigonometric)
  - `mat3_frobenius`, conditioned `mat3_inverse`
- **viva_math/matrix_dense** — `DenseMat` with `BitArray` storage
  (IEEE-754 LE row-major), O(1) random access
- **viva_math/quaternion** — Unit quaternions: `from_axis_angle`, `mul`,
  `inverse`, `rotate`, `nlerp`, `slerp`, `to_axis_angle`
- **viva_math/complex** — Algebra +
  `exp/log/sqrt/sin/cos/tan/pow_int/pow`, polar form
- **viva_math/vec2** / **vec4** / **vecn** with `hypot`-based length

### Added — Probability and statistics

- **viva_math/random** — Opaque `Seed`, multiple algorithms, real
  Fisher-Yates shuffle
- **viva_math/statistics** — Neumaier-backed sum/mean/covariance/Pearson,
  Pébay-stable skewness/kurtosis, rolling `moving_average` O(n)
- **viva_math/distributions** — Gaussian, Uniform, Exponential, Laplace,
  Cauchy, Bernoulli, Categorical
- **viva_math/special** — Lanczos gamma/lgamma/digamma/beta/factorial/
  binomial
- **viva_math/entropy** — Tsallis, Fisher information, differential
  entropy for Gaussians, KL with additive smoothing
- **viva_math/tdigest** — Dunning's t-digest streaming quantiles

### Added — Signal processing

- **viva_math/fft** — Cooley-Tukey radix-2 FFT/IFFT, `pad_to_power_of_two`
  helper

### Added — Hierarchical active inference (2026 papers)

- **viva_math/free_energy**
  - `Hierarchical` / `HierarchicalLayer` types (Meta-PCN, ICLR 2026)
  - `hierarchical_inference_step` + `hierarchical_infer`
  - `meta_prediction_errors` — PE-of-PE
  - `bpc_update` / `bpc_precision_update` (Bayesian Predictive Coding)
  - `ExpectedFreeEnergy`, `select_policy`, `policy_posterior`
  - Multivariate `gaussian_kl_divergence_full` (d=3 isotropic)

### Added — Tooling and CI

- `.github/workflows/{ci,bench,release}.yml`, `.github/dependabot.yml`
- `bench/precision_bench.gleam` + `bench/ode_bench.gleam`
- `examples/{pad_dynamics,active_inference,autodiff_demo}.gleam`

### Added — Testing

- Property-based tests with `qcheck` (random generators + shrinking)
- Golden-value tests against scipy/Wolfram references
- 280 internal tests; ecosystem totals 1167 tests passing
  (viva_math 280 + viva_emotion 55 + viva_telemetry 41 + viva_tensor 791)

### Changed

- `statistics.mean`/`skewness`/`kurtosis`, `entropy.shannon`,
  `scalar.logsumexp` switched to Neumaier compensated sum
- `random.shuffle` migrated to classical Fisher-Yates
- `vec*.length` migrated to progressive `hypot` reduction
- `matrix.mat3_inverse` rejects ill-conditioned matrices via relative
  tolerance `ε · ‖M‖_F³`
- `statistics.moving_average` rolling-sum O(n)
- `scheduler.triangle` formula corrected
- `scheduler.cosine_warm_restarts` guards against `period ≤ 0`
- `random.bernoulli` clamps `p` to `[0, 1]`
- `random.categorical` rejects negative probabilities
- `distributions.exponential_sample` uses `-log1p(-u)/λ`
- `scalar.logaddexp(+∞, +∞)` returns `+∞` (was NaN)
- `scalar.smoothstep` handles degenerate `edge0 == edge1`
- `entropy.tsallis` uses fuzzy comparison for `q ≈ 1`
- `free_energy.gaussian_kl_divergence_full` corrected for d=3 isotropic
- `viva_math.gleam` `version` constant bumped to `"1.2.100"`

### References

- Hairer, Nørsett, Wanner (1993) "Solving ODEs I"
- Yoshida (1990) symplectic integrators
- Dunning & Ertl (2019) t-digest
- Cantos & Aragón (2026) arXiv:2603.21991 (λ-GELU)
- Aragón et al. (2026) arXiv:2603.06861 (IGLU)
- Lin et al. (2026) ICLR submission (Meta-PCN)
- Vasilescu & Friston (2025) arXiv:2503.24016 (BPC)
- Pébay (2008, 2016) Sandia higher-order moments
- Schubert (2018) numerically stable parallel covariance

## [1.2.0] - 2026-01-24

### Added

- **viva_math/cusp** - Stochastic Cusp Catastrophe (DeepSeek R1 proposal)
  - `StochasticCuspParams` type with sigma (noise intensity) and seed
  - `stochastic_gradient` - Gradient with Wiener process noise (dV/dx + σξ(t))
  - `stochastic_step` - Euler-Maruyama integration step
  - `simulate_stochastic` - Full trajectory simulation

- **viva_math/entropy** - Hybrid Emotional States (DeepSeek R1 proposal)
  - `hybrid_shannon` - Mixed entropy: H_hybrid = αH₁ + (1-α)H₂
  - `KlSensitivity` type: Standard, ArousalWeighted, CustomGamma
  - `kl_divergence_with_sensitivity` - D_KL^γ = γ(μ₁-μ₂)² + D_KL
  - `renyi` - Rényi entropy of order α: H_α = (1/(1-α))log₂(Σp^α)

- **viva_math/common** - Stochastic Utilities (inspired by viva_glyph)
  - `deterministic_noise` - Hash-based pseudo-random noise [-1, 1]
  - `wiener_increment` - Wiener process: √dt × ξ(t)
  - `inverse_decay`, `inverse_sqrt_decay` - 1/(1+t/τ) decay functions

- **viva_math/free_energy** - Full Gaussian KL
  - `gaussian_kl_divergence_full` - Complete KL with variance terms

### Changed

- **viva_math/attractor** - basin_weights now uses exp(-γd) instead of 1/d
  - Softmax with temperature: w_i = exp(-γd_i) / Σexp(-γd_j)
  - Max-subtraction for numerical stability (pattern from viva_glyph)

- **viva_math/free_energy** - Optimized log computation
  - Uses log(σ₂) - log(σ₁) instead of log(σ₂/σ₁) for robustness when σ₁ ≈ 0

### Validated

- All formulas validated by DeepSeek R1 671B via HuggingFace (2026-01-24)
- 58 tests passing

### References

- DeepSeek R1 671B (2026) - Formula proposals and validation
- Euler-Maruyama method for stochastic differential equations
- Rényi (1961) "On measures of entropy and information"

## [1.1.0] - 2026-01-24

### Added

- **viva_math/free_energy** - Enhanced with DeepSeek R1 671B validation
  - `precision_weighted_prediction_error` - Precision (Π) weighted errors
  - `gaussian_kl_divergence` - Closed-form KL for Gaussian distributions
  - `FeelingThresholds` type - Normalized thresholds (μ ± σ)
  - `classify_feeling_normalized` - Statistics-based classification
  - `update_thresholds` - Online learning with EMA
  - `variational_bound` - F ≤ -log p(o) + D_KL(q||p)
  - `compute_state` - Full FEP with precision and thresholds
  - `compute_state_simple` - Legacy interface for backwards compatibility

### Changed

- **viva_math/free_energy**
  - Formula updated to F = Π·(μ-o)² + D_KL(q||p) (validated by DeepSeek R1)
  - `FreeEnergyState` now includes `precision` field
  - `complexity` now uses proper KL divergence with prior variance

### References

- DeepSeek R1 671B (2026) - Mathematical validation
- Parr & Friston (2019) "Generalised free energy and active inference"

## [1.0.0] - 2026-01-23

### Added

- **viva_math/common** - Utility functions
  - `clamp`, `clamp_unit`, `clamp_bipolar` - Value clamping
  - `lerp`, `inverse_lerp` - Linear interpolation
  - `sigmoid`, `sigmoid_standard` - Sigmoid activation
  - `softmax` - Probability normalization
  - `safe_div` - Division with default on zero
  - `smoothstep` - Hermite interpolation
  - `exponential_decay` - Time-based decay

- **viva_math/vector** - 3D vector operations for PAD space
  - `Vec3` type with x, y, z components
  - Basic operations: `add`, `sub`, `scale`, `negate`, `multiply`
  - Products: `dot`, `cross`
  - Metrics: `length`, `distance`, `normalize`
  - Utilities: `lerp`, `clamp`, `clamp_pad`, `weighted_average`
  - PAD aliases: `pad`, `pleasure`, `arousal`, `dominance`

- **viva_math/cusp** - Cusp catastrophe theory (Thom, 1972)
  - `CuspParams` type with alpha/beta control parameters
  - `potential`, `gradient`, `discriminant` - Core functions
  - `is_bistable` - Bistability detection
  - `equilibria` - Find stable/unstable equilibrium points
  - `nearest_equilibrium`, `would_jump` - State analysis
  - `volatility` - Emotional volatility measure
  - `from_arousal_dominance` - PAD to cusp mapping

- **viva_math/free_energy** - Free Energy Principle (Friston, 2010)
  - `FreeEnergyState` type with feeling classification
  - `Feeling` enum: Homeostatic, Surprised, Alarmed, Overwhelmed
  - `prediction_error`, `complexity`, `free_energy` - Core FEP
  - `compute_state` - Full state with feeling
  - `surprise` - Single dimension surprise
  - `active_inference_delta` - Action selection
  - `precision_weighted_error`, `estimate_precision` - Precision weighting
  - `belief_update` - Bayesian belief updating
  - `generalized_free_energy` - Planning/action selection

- **viva_math/attractor** - Emotional attractor dynamics (Mehrabian, 1996)
  - `Attractor` type with name and PAD position
  - `emotional_attractors` - 8 basic emotions (joy, sadness, fear, anger, etc.)
  - `nearest`, `basin_weights` - Attractor analysis
  - `analyze` - Comprehensive attractor analysis
  - `classify_emotion` - Emotion classification by nearest attractor
  - `attractor_pull`, `weighted_pull` - Force calculations
  - `ou_mean_reversion` - Ornstein-Uhlenbeck dynamics
  - `in_basin`, `nearby_attractors` - Spatial queries
  - `blend_attractors`, `create` - Attractor manipulation

- **viva_math/entropy** - Information theory
  - `shannon`, `shannon_normalized` - Shannon entropy
  - `kl_divergence`, `symmetric_kl` - Kullback-Leibler divergence
  - `jensen_shannon` - Jensen-Shannon divergence
  - `cross_entropy`, `binary_cross_entropy` - Cross-entropy
  - `mutual_information`, `conditional_entropy` - Information metrics
  - `relative_entropy_rate` - Temporal entropy

### Dependencies

- `gleam_stdlib >= 0.34.0`
- `gleam_community_maths >= 2.0.0` - Base math library

### References

- Grasman et al. (2009) "Fitting the Cusp Catastrophe in R"
- Friston (2010) "The free-energy principle: a unified brain theory?"
- Mehrabian (1996) "Pleasure-arousal-dominance: A general framework"
- Oravecz et al. (2009) "O-U Process in Affective Dynamics"
- Shannon (1948) "A Mathematical Theory of Communication"

[Unreleased]: https://github.com/gabrielmaialva33/viva_math/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.0
[1.1.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.1.0
[1.0.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.0.0

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.102] - 2026-05-21

Roadmap entries closed: Ornstein-Uhlenbeck mood dynamics, deeper Bayesian
variational free energy, Wasserstein distance between affective distributions,
property-based tests on every closed form.

### Added — `viva_math/ou` (new module)

- `OUParams1D`, `OUParamsVec3` — scalar and componentwise PAD parameters.
- `step`, `simulate` — exact transition kernel (Doob 1942), no
  discretisation error regardless of `dt`.
- `mean_at`, `variance_at`, `stationary_variance`, `stationary_std`,
  `autocovariance`, `half_life` — analytic moments.
- Vec3 variants: `step_vec3`, `simulate_vec3`, `mean_at_vec3`,
  `variance_at_vec3`, `stationary_variance_vec3` (componentwise).
- `is_valid`, `is_valid_vec3` — physical-meaningfulness predicates.

References: Uhlenbeck & Ornstein (1930); Oravecz, Tuerlinckx & Vandekerckhove
(2009) *Ornstein-Uhlenbeck Process in Affective Dynamics*; Doob (1942).

`attractor.ou_mean_reversion` remains unchanged (Euler-step deterministic
version) — no API break downstream.

### Added — `viva_math/transport` (new module)

- `wasserstein_1_empirical`, `wasserstein_2_empirical` — 1D Wasserstein
  between empirical samples (works for unequal sample sizes via CDF
  integration).
- `wasserstein_2_gaussian` — closed form for two Gaussians
  `√((μ₁−μ₂)² + (σ₁−σ₂)²)`.
- `wasserstein_pad` — componentwise W₂ over PAD axes.

References: Villani (2008) *Optimal Transport*; Peyré & Cuturi (2019)
*Computational Optimal Transport*.

### Added — `viva_math/free_energy` (Bayesian deepening)

- `ELBO` type + `elbo` — full Evidence Lower Bound decomposition
  (reconstruction − KL) for the Gaussian-Gaussian conjugate model.
- `MeanFieldParams` + `mean_field_update` — closed-form Gaussian posterior
  under conjugate prior + Gaussian likelihood (Bishop §2.3.3).
- `mean_field_iterate` — sequential Bayes over observation batches.
- `laplace_approximation` — Gaussian fit at the MAP via gradient ascent +
  central-difference Hessian.
- `log_evidence_gaussian` — closed-form marginal `log p(x)`.
- `scalar_gaussian_kl` — 1D Gaussian KL (companion to the existing Vec3
  variants).

References: Beal (2003); Bishop (2006) ch. 10; Friston (2010).

### Added — Property tests

- `test/qcheck_test.gleam` gained properties for the new features
  (OU mean reversion, ELBO bound, KL non-negativity, Wasserstein symmetry +
  self-zero) and for previously uncovered closed forms (clamp idempotence,
  sigmoid range, sin²+cos²=1, softmax sum, JS symmetry, KL self-zero,
  Shannon non-negativity, erf odd-parity, GELU/SiLU at zero).
- `test/viva_math_test.gleam` gained unit tests for `ou.*`, the Bayesian
  block of `free_energy.*`, and `transport.*`.

### Fixed (post-review)

Rigorous mathematical audit (Codex GPT-5.3 + manual cross-check) identified
one bug and several documentation gaps, all addressed before release:

- **`transport.wasserstein_2_empirical`** (`|p| ≠ |q|`): rewrote the unequal
  sample-size branch via quantile-based integration over the union of
  breakpoints `{i/n} ∪ {j/m}`. The previous CDF-gap path `∫(F_P − F_Q)² dx`
  is only valid for `W_1` (the W_1/W_2 duality via integration by parts
  breaks under the quadratic kernel). Confirmed by counterexample:
  `P=[0,2], Q=[1]` returns `W_2 = 1.0` (was `≈ 0.707`).
- **`ou.step` and `ou.variance_at`**: routed `1 − e^(−2θΔ)` through
  `scalar.expm1` to recover the Brownian limit `σ²·t` (as `θ·t → 0`)
  without catastrophic cancellation.
- **`ou.variance_at` docstring**: explicit note that `x0` is mathematically
  ignored (signature kept only for API symmetry with `mean_at` /
  `variance_at_vec3`).
- **`transport.wasserstein_pad` docstring**: explicit that it is a
  componentwise/marginal **pseudo-metric** (triangle inequality holds via
  Minkowski; identity of indiscernibles fails for joint distributions with
  identical marginals), **not** the multivariate `W_2`.
- **Property tests added**: `property_wasserstein_2_unequal_known_test`
  (would catch the fixed bug), `property_wasserstein_2_unequal_symmetric_test`,
  `property_scalar_gaussian_kl_unequal_var_nonneg_test` (exercises the
  `log(σ_p²/σ_q²)` term that the equal-variance test left untouched), and
  `property_ou_variance_brownian_limit_test` (regression guard for the
  `expm1` rewrite).

### Performance (post-SOTA cross-check)

Performance audit (Codex GPT-5.3 + exa-driven literature review covering
`thermox` 2024, OT1D, BONG 2024, Sliced Wasserstein 2024-2025 papers).
Literature confirmed our algorithmic choices (Doob exact kernel for OU,
closed-form conjugate VI, O(n log n) sort-based 1D Wasserstein) are SOTA;
opportunities were in the implementation, not the math.

- **`transport.wasserstein_1_empirical`, `wasserstein_2_empirical`**:
  unequal-sample-size path rewritten as a single linear-time walk over the
  union of quantile breakpoints — `O((n+m)·log(n+m))` total (dominated by
  sort) instead of the previous `O((n+m)²)` caused by repeated `nth` lookups.
- **`transport.wasserstein_pad`**: PAD axis projection unified into a single
  recursive pass via `split_pad`, replacing three `list.map` traversals.
- **`transport.wasserstein_*_empirical` (equal size)**: eliminated
  `list.zip` intermediate via `walk_pair_squared` / `walk_pair_abs` cons
  traversals.
- **`ou.simulate`, `ou.simulate_vec3`**: transition kernel
  (`decay = e^(−θΔ)`, `std = σ·√(−expm1(−2θΔ)/(2θ))`) is now pre-computed
  once per simulation; the loop only does a multiply-add and a normal draw
  per step. Vec3 path uses a `Kernel` record per axis.
- **`free_energy.mean_field_update`**: single-pass `count_and_sum` replaces
  separate `list.length` + `list.fold` — half the list traversals.
- **`free_energy.int_to_float`**: replaced the recursive Gleam helper with
  Erlang's `erlang:float/1` BIF (O(1)). Used everywhere `count` is
  converted in the Bayesian path.

### Fixed (deep audit — Codex GPT-5.5 high reasoning)

Third audit pass surfaced **3 bugs** + 2 inconsistencies that the formal-math
and performance reviews had missed:

- **`free_energy.elbo` — Jensen bound violation with `q_var ≤ 0`** (BUG):
  the reconstruction term silently inflated by adding negative `q_var` while
  the KL was floored at `0.0`, producing an "ELBO" greater than `log p(x)`.
  Counterexample: `elbo(0.0, 0.0, -10.0, 0.0, 1.0, 1.0).total ≈ 4.081` vs
  `log_evidence_gaussian(0.0, 0.0, 1.0, 1.0) ≈ -1.266`. **Fixed**: guard
  rejects any non-positive variance upfront, returning the large-penalty
  sentinel triple.
- **`free_energy.log_evidence_gaussian` — invalid variance compensation**
  (BUG): `prior_var = 2.0, likelihood_var = -1.0` produced a finite (but
  meaningless) log-density because the sum `marginal_var = 1.0` was the only
  thing checked. **Fixed**: both components must be `> 0`.
- **`free_energy.laplace_approximation` — infinite variance on flat
  posterior** (BUG): `second < 0.0` accepted curvatures as small as
  `-1.0e-300`, yielding `q_var = +∞`. **Fixed**: requires
  `second < -1.0e-12` (sized to the `O(h²)` finite-difference roundoff
  floor at `h = 1.0e-4`).
- **`transport.wasserstein_2_gaussian` — negative `stddev`**
  (INCONSISTENCY): `Gaussian(0, -1)` vs `Gaussian(0, 1)` gave `W_2 = 2.0`
  even though `N(μ, σ²)` only depends on `σ²`. **Fixed**: `|σ₁|` and `|σ₂|`
  before subtracting; doc rationale.
- **`ou.step` / `ou.simulate` accept `dt < 0`** (INCONSISTENCY): negative
  `dt` makes `var_term` go negative, `std_term` silently collapses to `0.0`,
  and a normal draw is consumed but unused. **Mitigated**: documented as
  caller-validated input pointing at `is_valid`; no API break.
- **`transport.walk_quantile` `squared: Bool` branch** (SMELL): the BEAM
  doesn't specialise the loop on the literal `True`/`False` callers pass.
  **Fixed**: split into `walk_quantile_squared` (for W₂²) and
  `walk_quantile_abs` (for W₁) — no branch inside the hot loop.

### Added — Deep-audit regression tests

`test/viva_math_test.gleam` gained 7 regression tests covering each of the
bugs above plus two algebraic-property checks proposed by the audit:

- `vfe_elbo_negative_q_var_does_not_break_bound_test` — Jensen bound
- `vfe_log_evidence_rejects_componentwise_invalid_variance_test`
- `vfe_laplace_rejects_flat_log_posterior_test`
- `wasserstein_2_gaussian_negative_stddev_test`
- `ou_mean_at_composes_over_time_test` — semigroup property
  `mean_at(t₁+t₂) = mean_at(t₂, mean_at(t₁))`
- `wasserstein_2_triangle_unequal_sizes_test` — triangle inequality on
  unequal-size empirical samples
- `vfe_mean_field_iterate_matches_flat_batch_test` — Bishop §2.3.6
  associativity of sequential conjugate updates

### Project organisation (Gleam best-practice alignment)

Codex GPT-5.5 audit cross-checked against `gleam-lang/stdlib`,
`lustre-labs/lustre`, `gleam-wisp/wisp`, `lpil/gleeunit`,
`mooreryan/gleam_qcheck` repos.

- **Tests split by module** (mirrors `gleam-lang/stdlib` convention of one
  `test/<module>_test.gleam` per `src/<package>/<module>.gleam`):
  - `test/ou_test.gleam` — extracted from the 1287-line mega-file
  - `test/transport_test.gleam` — extracted
  - `test/free_energy_variational_test.gleam` — extracted (VFE Bayesian +
    deep-audit regressions)
  - `test/viva_math_test.gleam` shrunk from 1287 → 945 lines, now hosts only
    the legacy per-domain blocks (common/vector/cusp/attractor/entropy/...)
- **Shared test helpers**: new `test/test_support.gleam` consolidates
  `is_close/3`, `is_close_vec3/3`, `is_close_complex/3`, `is_close_list/3`
  and exports `tight` (1e-12) / `loose` (1e-6) tolerance constants. Replaces
  4 duplicates (`is_close` in `viva_math_test.gleam`, `close` in
  `qcheck_test.gleam`, `close_complex`/`close_list` in `fft_test.gleam`).
- **CONTRIBUTING.md fixed**: clone URL was pointing at non-existent
  `mrootx/viva_math` — corrected to `gabrielmaialva33/viva_math`. Testing
  section now reflects the per-module file convention.
- **README.md refreshed**: version badge (1.2.0 → 1.2.102), test count
  (58 → 333), target field ("Erlang + JavaScript" → "Erlang (BEAM)" — the
  Erlang FFI in `viva_math_random_ffi.erl` precludes the JS target), and
  the `gleam_community_maths` reference (dropped in 1.2.101) was scrubbed.
- **`.github/workflows/release.yml`**: added `gleam format --check` and
  `gleam test` gates before `gleam publish` (the workflow was publishing
  unverified artifacts).
- **`free_energy.gleam` module doc**: removed `Validated by DeepSeek R1
  671B` line (LLM validation is not a scientific reference); replaced with
  Beal (2003) and Bishop (2006) which are the actual papers driving the
  variational block added in 1.2.102.

### Validated

- 333 tests passing (was 326 → +7; +53 vs 1.2.101).
- `gleam format --check src test` clean.
- Test files now: `viva_math_test.gleam` (945L, legacy domain blocks),
  `ou_test.gleam` (110L), `transport_test.gleam` (67L),
  `free_energy_variational_test.gleam` (139L), `test_support.gleam` (50L),
  plus the pre-existing `fft_test`, `precision_test`, `property_test`,
  `qcheck_test`, `sota_test`.

## [1.2.101] - 2026-05-21

Self-contained release. The library no longer depends on
`gleam_community_maths`. All transcendental, trigonometric and root
operations previously delegated to that package are now provided by
`viva_math/scalar` via Erlang `:math` BIFs (no runtime overhead).

### Added — `viva_math/scalar`

- Trigonometry BIFs: `sin`, `cos`, `tan`, `asin`, `acos`, `atan2`.
  `atan` was promoted from private to public.
- Logarithms BIFs: `log2`, `log10` (the natural `ln` was already public).
- `cbrt` — real cube root defined for all `Float` via the sign trick.
- Result-wrapped, domain-safe variants for chaining without crashes:
  `try_ln`, `try_log2`, `try_log10`, `try_sqrt`, `try_cbrt`,
  `try_nth_root(x, n)` (handles odd/even `n`, errors for `n ≤ 0`).

### Changed

- **Dependency drop**: `gleam_community_maths` removed from `gleam.toml`.
  Only `gleam_stdlib` remains as a runtime dependency.
- `viva_math/common` — `maths.exponential` → `scalar.exp`;
  re-exported constants (`pi`, `e`, `tau`) now sourced from
  `viva_math/constants`; `nth_root(_, 2)` → `gleam/float.square_root`.
- `viva_math/attractor` — `maths.exponential` → `scalar.exp`.
- `viva_math/cusp` — `maths.acos`/`cos`/`pi` → `scalar.acos`/`cos` +
  `constants.pi`; the internal `cbrt` helper now delegates to
  `scalar.cbrt`; all `nth_root(_, 2)` → `float.square_root`.
- `viva_math/vector` — `nth_root(_, 2)` → `float.square_root`.
- `viva_math/entropy` — `logarithm_2` → `scalar.try_log2`;
  `natural_logarithm` → `scalar.try_ln`; `exponential` → `scalar.exp`.
- `viva_math/free_energy` — `natural_logarithm` → `scalar.try_ln`;
  `exponential` → `scalar.exp`; `nth_root(_, 2)` → `float.square_root`.
- `viva_math.gleam` `version` constant bumped to `"1.2.101"`.

### Validated

- 280 tests passing (no behaviour change, only routing).

## [1.2.100] - 2026-05-21

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

[Unreleased]: https://github.com/gabrielmaialva33/viva_math/compare/v1.2.101...HEAD
[1.2.101]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.101
[1.2.100]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.100
[1.2.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.0
[1.1.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.1.0
[1.0.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.0.0

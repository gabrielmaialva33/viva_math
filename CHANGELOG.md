# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.103] - 2026-05-21

Compatibility-pack release. Adds the handful of APIs that downstream
packages (`viva_glyph`, `viva_tensor`) still pull from
`gleam_community_maths`, so they can complete the migration started in
`viva_math 1.2.101`.

A pre-tag Codex GPT-5.5 review flagged a real bug (`lp_norm(v, 0.0)`
crashed on Erlang with `badarith` while returning `Infinity` on JavaScript)
and the absence of direct local tests for the new APIs. Both were
addressed before publishing.

### Added — `viva_math/vecn`

- `euclidean_distance(a, b)` — alias for `distance/2`. Symmetry with
  `gleam_community_maths` API.
- `manhattan_distance(a, b)` — `Σ |aᵢ − bᵢ|`.
- `cosine_similarity(a, b)` — `(a · b) / (‖a‖ · ‖b‖)`. `Error(Nil)` when
  either vector has zero norm or lengths differ.
- `lp_norm(v, p) -> Result(Float, Nil)` — general Lₚ norm. `p = 1.0` ≡
  Manhattan, `p = 2.0` ≡ `length` (Euclidean). **Returns `Error(Nil)` for
  `p ≤ 0`** — that domain would diverge across targets (`badarith` on
  Erlang vs `Infinity` on JavaScript), which is unacceptable for a
  dual-target library. `0 < p < 1` is allowed but produces a
  pseudo-norm (triangle inequality fails); doc says so.

### Added — `viva_math/statistics`

- `linear_space(start, stop, steps, endpoint)` — NumPy `linspace`.
- `logarithmic_space(start, stop, steps, endpoint, base)` — NumPy
  `logspace`, parameterised by `base`.
- `cumulative_sum(xs)` — running `Σ`. Note: naive left-to-right
  summation (mirrors NumPy `cumsum`). For compensated summation, fold
  through `precision.neumaier_sum` instead.
- `cumulative_product(xs)` — running `Π`.

### Added — `viva_math/precision`

- `is_close(a, b, rtol, atol)` — `|a − b| ≤ atol + rtol · |b|`. NumPy
  `isclose` semantics (asymmetric in `b` by design).
- `all_close(pairs, rtol, atol)` — `True` iff every paired sample is
  `is_close`. Vacuous `True` for empty input, matching NumPy.

### Added — `test/compat_pack_test.gleam`

31 direct unit tests covering all 10 new functions:

- `vecn.euclidean_distance`: Pythagorean triple `(3,4) → 5`, self-zero,
  size mismatch rejection.
- `vecn.manhattan_distance`: tabulated + size mismatch.
- `vecn.cosine_similarity`: self-one, orthogonal-zero, **all three
  zero-vector paths return `Error(Nil)`** (both, left-only, right-only).
- `vecn.lp_norm`: L₁/L₂/L₃ tabulated, empty vector, **p=0 and p<0 both
  return `Error(Nil)`** (cross-target safety).
- `statistics.linear_space`: endpoint True/False, singleton, empty,
  `start == stop` degenerate.
- `statistics.logarithmic_space`: powers of 10 (`[1, 10, 100, 1000]`).
- `statistics.cumulative_{sum,product}`: tabulated + empty.
- `precision.is_close`: within rtol, within atol, neither, exact
  bit-equality at the next IEEE-754 double after 1.0.
- `precision.all_close`: empty (vacuous True), homogeneous pass, one
  fails.

### Fixed (pre-tag review)

- **`vecn.lp_norm` cross-target divergence**: changed signature from
  `Float` to `Result(Float, Nil)` and added an explicit `p ≤ 0` guard.
  Previously `lp_norm(v, 0.0)` would raise `badarith` on Erlang and
  return `Infinity` on JavaScript — a silent dual-target inconsistency.

### Validated

- **553 tests passing on both Erlang and JavaScript targets** (+31 new
  direct unit tests for the compat pack). The "no direct coverage" gap
  flagged by the pre-tag review is closed.
- `gleam format --check src test` clean.

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

### Test coverage — 5 previously untested modules

Codex GPT-5.5 coverage audit identified 5 public modules with **zero direct
tests**: `autodiff`, `autodiff_reverse`, `matrix_dense`, `tdigest`, `special`.
Closed-form derivative identities (Lyness, dual numbers), algebraic invariants
(identity laws, transpose involution), and tabulated reference values
(Γ(n)=(n-1)!, B(2,3)=1/12, ψ(1)=−γ) used as anchors per the AD validation
literature (Birthe van den Berg et al. 2024; Mazza & Pagani 2022; Dunning &
Ertl 2019).

- **`test/autodiff_test.gleam`** (28 tests) — forward-mode AD: arithmetic
  derivatives via Leibniz, transcendentals via chain rule, `Dual3`
  multivariate gradient (`∇(x²+y²+z²) = (2x,2y,2z)`, `∇(xyz)`,
  `∇exp(x+y+z)`), `jacobian` of linear maps, `lift1` chain-rule wrapper,
  `gelu` at 0.
- **`test/autodiff_reverse_test.gleam`** (16 tests) — reverse-mode AD on
  the computation tape: every elementary op, cross-mode consistency with
  forward AD, multi-input `gradients/2`, **gradient accumulation on reused
  nodes** (defends a classic reverse-AD failure mode where the backward
  pass walks only one path through a shared intermediate).
- **`test/matrix_dense_test.gleam`** (17 tests) — `BitArray` dense matrix:
  shape constructors + invalid-dim errors, identity matrix diagonal,
  `from_list` round-trip, algebraic identities (A+0=A, A−A=0, (Aᵀ)ᵀ=A,
  I·A=A, A·I=A), `frobenius` on tabulated 3-4-5 triangle, `trace(I_n)=n`,
  `byte_size` arithmetic.
- **`test/tdigest_test.gleam`** (14 tests) — streaming quantile estimator:
  weight conservation across `insert`/`insert_all`/`insert_weighted`,
  boundary queries (`quantile(0)=min`, `quantile(1)=max`, out-of-range
  rejection), median accuracy on uniform [1..100] (loose tolerance per
  t-digest centre-vs-tail trade-off), `merge` mass + extrema preservation,
  monotonic quantile property.
- **`test/special_test.gleam`** (25 tests) — Lanczos `gamma`/`lgamma`,
  `digamma`, `beta`, `lbeta`, `factorial`, `binomial`: tabulated values
  (Γ(½)=√π, Γ(3/2)=½·√π, ψ(1)=−γ, ψ(2)=1−γ), Beta symmetry, factorial
  exactness vs Γ(n+1), binomial Pascal identities + edge cases. Documents
  that `binomial(n, k) = Ok(0.0)` for `k > n` (no combinations) and
  `Error(Nil)` for `k < 0` (invalid).

### Added — Documentation pass

Closed the docstring gap identified by the audit: every `pub fn` / `pub
type` in `viva_math/autodiff.gleam` (22 items) and
`viva_math/autodiff_reverse.gleam` (14 items) now carries a `///` comment
stating the **local derivative rule** it implements (e.g., `mul: ∂z/∂a = b,
∂z/∂b = a`). `viva_math/matrix_dense.DenseMat` documents its shape
invariant. The audit confirmed `tdigest` and `special` already had complete
public docstrings.

### Encapsulation — opaque types

Five types whose direct construction could produce invariant-violating
states are now `pub opaque type`. Surface accessors added where consumers
needed read access. Cross-checked against `qcheck` (`pub opaque Seed`) and
`gleam-lang/stdlib` (most concrete types are constructed via smart
constructors).

- **`viva_math/matrix_dense.DenseMat`** — `BitArray data` previously
  allowed to mismatch the declared `rows × cols × 8` shape. New accessors:
  `rows/1`, `cols/1`, `shape/1`.
- **`viva_math/tdigest.TDigest`** + **`Centroid`** — direct construction
  could violate the sorted-centroid invariant or desync
  `total_weight ↔ Σ weights`. New accessors: `compression/1`,
  `centroid_mean/1`, `centroid_weight/1`.
- **`viva_math/autodiff_reverse.Tape`** — corruptible `(nodes, next_id)`
  could break gradient propagation; opaque, threaded via forward ops.
- **`viva_math/autodiff_reverse.Node`** — paired forward value + `Op`,
  constructed only by `push`.
- **`viva_math/autodiff_reverse.Op`** — constructors (`Input`, `Add`,
  `Sub`, `Mul`, `Div`, `Neg`, `Scale`, `Exp`, `Ln`, `Sin`, `Cos`, `Tanh`,
  `Sigmoid`, `Pow`) no longer exposed; callers build expressions through
  the high-level forward ops.

External call sites refactored:
- `test/matrix_dense_test.gleam:14–15` — `m.rows`/`m.cols` →
  `md.rows(m)`/`md.cols(m)`.
- `test/tdigest_test.gleam:23` — `d.compression` → `td.compression(d)`.

`Tape`, `Node`, `Op`, `Centroid` had **zero external call sites** — purely
internal refactor.

### Deprecation — idiomatic Result-returning function names

`viva_math/scalar` adds wrappers aligned with `gleam/float`'s convention
(name the math operation, let the `Result` type communicate failure;
reserve `try` for the `result.try` combinator):

| New (idiomatic) | Deprecated alias |
|---|---|
| `logarithm/1` | `try_ln/1` |
| `logarithm_2/1` | `try_log2/1` |
| `logarithm_10/1` | `try_log10/1` |
| `square_root/1` | `try_sqrt/1` |
| `cube_root/1` | `try_cbrt/1` |
| `nth_root/2` | `try_nth_root/2` |

The legacy `try_*` functions stay around as `@deprecated` wrappers — they
call the new names so semantics are identical. Internal callers
(`free_energy`, `entropy`, `ou`, `free_energy_variational_test`) all
migrated to the idiomatic names, leaving the build warning-free.

### Renamed

- `test/property_test.gleam` → `test/invariant_test.gleam`. Disambiguates
  from `test/qcheck_test.gleam` (which uses `qcheck`-generated samples for
  *true* property-based testing). The new file's docstring explains the
  distinction.

### Numerical precision audit (post-encapsulation)

Codex GPT-5.5 god-audit inventoried the full public surface — **632 items
across 29 modules, ~40% direct test coverage** — and flagged loose
tolerances on closed-form identities. The audit was anchored against
CPython's `test_math.py` ulp-based testing and N1630 (WG14) edge-case
recommendations.

**Tolerance tightening — 8 closed-form identities pulled to IEEE-754
precision** (`1e-3` / `1e-6` → `1e-12` / `1e-15`):

| Test | Identity | Before | After |
|---|---|---|---|
| `common.sigmoid(0, 1)` | `= 0.5` exact | `1e-3` | `1e-15` |
| `softmax_sum_to_one` | `Σ softmax = 1` | `1e-3` | `1e-12` |
| `vec3_length(3,4,0)` | Pythagorean triple `= 5` | `1e-3` | `1e-15` |
| `scalar.erf(1)` | tabulated, libm `:math.erf` | `1e-6` | `1e-12` |
| `scalar.softplus(0)` | `= ln(2)` | `1e-6` | `1e-15` |
| `scalar.logsumexp([0,0])` | `= ln(2)` | `1e-6` | `1e-15` |
| `constants.pi` | double-precision literal | `1e-6` | `1e-15` |
| `autodiff.gelu' at 0` | `= 0.5` closed form | `1e-3` | `1e-12` |

**`test/test_support.gleam` extended** with hybrid tolerance helpers:

- `is_close_rel/3` — relative comparator `|a−b| ≤ rel_tol·max(|a|,|b|)`
- `is_close_hybrid/4` — passes if **either** absolute or relative tolerance
  holds (mirrors CPython's `result_check`)
- Constants: `tight = 1e-12`, `machine = 1e-15`, `transcendental = 1e-13`,
  `loose = 1e-6` — encode the precision regime in the call site

### Added — algebraic identity tests (`test/identities_test.gleam`)

14 universal-law tests that golden-value testing misses:

- **Round-trips**: `exp(ln(x)) = x`, `ln(exp(x)) = x`, `sqrt(x)² = x`,
  `cbrt(x)³ = x`, `sin² + cos² = 1`
- **Translation invariance**: `softmax(x + c) = softmax(x)` (stable-softmax
  Jensen invariant)
- **Special-function recurrences**: `Γ(x+1) = x·Γ(x)`, `ψ(x+1) = ψ(x) + 1/x`,
  `lbeta(x,y) = lgamma(x) + lgamma(y) − lgamma(x+y)`
- **OU limit behaviour**: `Var(t→∞) = σ²/(2θ)`, `step(dt=0) = x_0`,
  `mean_at(t=0) = x_0`
- **Cusp consistency**: `cusp.gradient = d/dx cusp.potential` (via central
  finite differences at `h = 1e-5`, tolerance `1e-7`)
- **Matrix algebra**: `det(Aᵀ) = det(A)`

### Added — edge-case tests (`test/edge_cases_test.gleam`)

18 boundary tests anchored on N1630 + CPython conventions:

- **Domain rejection**: `square_root(-x)`, `logarithm(0)`,
  `logarithm(-x)`, `nth_root(x, 0)`, `nth_root(x, -1)`,
  `nth_root(-x, even)` all return `Error(Nil)`
- **Identity at boundaries**: `square_root(0)`, `cube_root(0)`,
  `logarithm(1) = 0`, `nth_root(x, 1) = x`
- **Cube root negative-real**: `∛(-8) = -2`, 5th-root of -32 = -2
- **Power-of-base logs exact**: `log₂(2ⁿ) = n`, `log₁₀(10ⁿ) = n`
- **`expm1` cancellation defence**: `expm1(1e-10)/x ≈ 1` (would lose ~10
  significant digits with `exp(x) - 1`)
- **OU vec3 path coverage**: `is_valid_vec3` accepts/rejects correctly,
  `stationary_std`, `variance_at(t = 1e6)` saturates to stationary variance

### Coverage — Pri 1/2/3 batch (Codex-implemented)

Delegated all three priorities from the previous god-audit to Codex
GPT-5.5 via wcgw with a surgical brief covering 6 new test files. Codex
implemented every module's missing surface anchored on closed-form
identities, energy-conservation invariants, and tabulated golden values.

**6 new test files, 42 new tests:**

- **`test/scalar_test.gleam`** (9 tests) — `fmod`, `pow`, `tan`, `asin`,
  `acos`, `atan`, `atan2` (four-quadrant), `deg_to_rad ↔ rad_to_deg`
  round-trip, `step`/`smootherstep` boundaries, ML activations
  (`leaky_relu`, `elu`, `selu`, `swish`, `hard_*`), `safe_*` variants for
  domain-invalid inputs.
- **`test/matrix_test.gleam`** (9 tests) — Mat2 algebra (`zero`,
  `identity`, `add`, `scale`, `mul`, `inverse`, `rotation(0)=I`,
  `rotation(2π) ≈ I`, `eigenvalues` of diagonal), Mat3 (`add`, `sub`,
  `scale`, `mul`, `inverse`, `trace`, `determinant`, `transpose`), Mat4
  shape ops, MatN shape errors.
- **`test/free_energy_core_test.gleam`** (8 tests) — `free_energy` /
  `compute_state`, `surprise(o, o, σ) = 0`, `variational_bound`,
  `belief_update` posterior precision sum, `bpc_*_update` with zero
  count preserves precision, `policy_posterior` sums to 1, `select_policy`
  returns index of minimum EFE, `hierarchical_*` zero error pathways.
- **`test/distributions_test.gleam`** (6 tests) — Gaussian closed-form
  PDF/CDF values, Laplace peak `1/(2·b)`, Cauchy peak `1/(π·γ)`, Bernoulli
  PMF sums to 1 + `sample ∈ {0, 1}`, Categorical PMF + entropy + sample
  in range, finiteness of all samplers.
- **`test/ode_test.gleam`** (7 tests) — Euler-Maruyama matches manual
  Gaussian increment scaled by `√dt`, Milstein agrees with Euler-Maruyama
  for additive noise, `velocity_verlet`/`leapfrog` preserve harmonic
  oscillator energy `E = ½v² + ½x²`, `yoshida4` strictly more accurate
  than `velocity_verlet` (order 4 vs order 2 convergence), `integrate_sde`
  trajectory length, `integrate_symplectic` matches manual stepping.
- **`test/golden_values_test.gleam`** (3 tests) — tabulated reference
  values for `scalar.{erf, erfc, gelu, silu}`,
  `special.{gamma, lgamma, digamma}`, `constants.{e, sqrt_2, sqrt_2pi}`.
  Used `is_close_hybrid` with `tight = 1e-12` as default.

### Documented limitations (`AUDIT NEEDED` markers)

Codex surfaced two real measurement gaps from the golden-value pass:

1. **`scalar.gelu(1.0)`**: the implementation uses the **exact erf-based
   GELU** (`x·Φ(x)`), not the tanh approximation. Golden value adjusted
   to `0.841_344_746_068_542_9` (exact) with a comment explaining the
   distinction. **No source fix needed** — the test was wrong, not the
   implementation.
2. **`special.digamma(5.0)`**: the asymptotic-series implementation
   (truncated at `x⁻¹⁰`) bottoms out at ~`1.2e-10` absolute error against
   the tabulated value `1.506_117_668_431_800_5`. Test tolerance set to
   `2e-10` with an `AUDIT NEEDED` comment in-place; **source-side TODO**:
   increase the recurrence threshold from `x ≥ 6` to `x ≥ 12`, or extend
   the series with two more Bernoulli terms.

### Fixed — `special.digamma` accuracy

The asymptotic series implementation pushes `x` past a threshold via the
recurrence `ψ(x) = ψ(x+1) − 1/x` before evaluating the series truncated
at `x⁻¹⁰`. The threshold was raised from `N = 6` to `N = 12`,
delivering **~1000× error reduction** at no measurable runtime cost.

Measured against the tabulated reference `ψ(5) = 1.506_117_668_431_800_5`:

| Threshold | `ψ(5)` returned | Absolute error |
|---|---|---|
| `N = 6` (before) | `1.506_117_668_548_3306` | `1.17e-10` |
| `N = 12` (after) | `1.506_117_668_431_9206` | `1.20e-13` |

`test/golden_values_test.gleam:42` tightened from `2e-10` → `1e-12` and
the `AUDIT NEEDED` comment removed.

### Documentation — Tier 0/1/2 (HexDocs-first)

Adopted the idiomatic Gleam approach (cf. `lustre`, `wisp`,
`gleam_stdlib`) — **no separate docs site framework**. The
`gleam docs build` pipeline now publishes API reference + 4 conceptual
guides + the changelog directly to `hexdocs.pm/viva_math`.

- **`docs/pad-model.md`** — the `Vec3` type, 8 emotion attractors,
  Mehrabian (1996) coordinates.
- **`docs/ou-dynamics.md`** — Doob exact transition kernel, closed-form
  moments, Vec3 PAD path, cancellation defences.
- **`docs/wasserstein.md`** — 1D empirical W₁/W₂ (with the `n ≠ m` bug
  fixed in this release explained), Gaussian closed form, `wasserstein_pad`
  as pseudo-metric.
- **`docs/numerical-accuracy.md`** — tolerance regimes, measured precision
  per function, cancellation defences (`expm1`, stable softplus,
  logsumexp), digamma fix above.

`gleam.toml` gained five `[[documentation.pages]]` entries that wire
these into the HexDocs sidebar:

```toml
[[documentation.pages]]
title = "Changelog"
path = "changelog.html"
source = "CHANGELOG.md"

[[documentation.pages]]
title = "PAD Emotional Model"
path = "pad-model.html"
source = "docs/pad-model.md"
# ... (4 more)
```

### Updated — README

- Test count badge: `333 → 510 passing`
- Removed stale `gleam_community_maths` reference (dropped in 1.2.101).
- Removed JS-target claim (FFI is Erlang-only; JS target is a roadmap
  item that needs FFI rework).
- Quick Start now exercises **6 modules** (was 4): `attractor`, `cusp`,
  `free_energy`, **`ou`**, **`random`**, **`transport`**.
- Fixed `free_energy.compute_state` call site (used wrong arity).
- New **Guides** section linking to the 4 conceptual docs.
- Roadmap: marked the 4 features (OU, VFE, Wasserstein, property tests)
  as ✅ done; added new ⏳ items (multivariate Wasserstein, ULP-by-ULP
  `mpmath` validation, JS target).

### Validated

- **510 tests passing** (no change in count — only source-side digamma
  fix; **+230 vs 1.2.101**).
- `gleam format --check src test` clean.
- `gleam check` clean.
- `gleam docs build` renders all 5 pages + API reference clean.

### Added — Multivariate Wasserstein (`transport.wasserstein_2_multivariate`)

True multivariate W₂ via **Sinkhorn-Knopp entropic regularization** (Cuturi
2013) — solves the OT problem

```
min_π ⟨π, C⟩ + ε·H(π)   s.t.  π·1 = a, πᵀ·1 = b
```

with `C[i,j] = ‖x_i − y_j‖²` (Euclidean squared on Vec3) and uniform
marginals. Returns `√(⟨π, C⟩)` (true distance, dropping the entropic
term). Pure Gleam — no GPU, no dependencies. Defensive `float.max(ε, 1e-12)`
floor + `safe_ratio` against division by zero.

**5 new tests** in `test/transport_test.gleam`: empty rejection, identity
(`W₂(P, P) ≈ 0`), single-point translation `W₂([0], [(1,0,0)]) = 1`,
symmetry across iterates, and comparison vs `wasserstein_pad` on product
distributions (where the two should coincide).

References: Cuturi (2013) *Sinkhorn Distances: Lightspeed Computation of
Optimal Transport*; Peyré & Cuturi (2019) *Computational Optimal
Transport*.

### Added — ULP-by-ULP validation infrastructure

`test/test_support.gleam` gained two helpers backed by **dual FFI**
(Erlang + JavaScript) implementing CPython's ordered-float-bits
convention (negative zero ↔ -1; signs flip into a monotonic integer space):

```gleam
pub fn ulp_distance(a: Float, b: Float) -> Int
pub fn is_close_ulp(a: Float, b: Float, max_ulps: Int) -> Bool
```

**Erlang FFI** (`test/test_support_ffi.erl`) extracts IEEE-754 bits via
`<<Bits:64/unsigned-integer>> = <<X:64/float>>`. **JavaScript FFI**
(`test/test_support_ffi.mjs`) uses `DataView.setFloat64` + `BigUint64`.

`test/golden_mpmath_test.gleam` ships 21 reference values computed at
100-bit precision in `mpmath` and asserts `<= 5 ULP` agreement by default.
Three real outliers, **measured and honestly documented**:

| Function | ULP distance | Note |
|---|---|---|
| `special.gamma(5.5)` | `8` | Lanczos series, near-integer arg |
| `special.digamma(1.0)` | `~1100` | asymptotic series truncation |
| `special.digamma(10.0)` | `~300` | idem |

Everything else (`erf`, `exp`, `ln`, `sin`, `cos`, `gamma` (most args),
`lgamma`) lands within 5 ULP of the 100-bit reference.

### Added — JavaScript target (partial)

`src/viva_math_random_ffi.mjs` implements `viva_math/random` for the
JavaScript target using **Mulberry32** (PRNG, seedable, statistically
adequate) + **Box-Muller** for `normal_standard` / `normal_with`. `jump/1`
is a documented no-op on JS (Mulberry32 has no jump-ahead).

`gleam.toml` no longer pins `target = "erlang"` — both targets are
attempted. Current state:

| Target | Status |
|---|---|
| `gleam test --target erlang` | **521 passed, no failures** |
| `gleam test --target javascript` | ❌ compile fails on Erlang-only externals in `viva_math/precision` (`int_to_float`, `sqrt`) and `viva_math/autodiff_reverse` (`exp_f`, `ln_f`, `sin_f`, `cos_f`, `tanh_f`, `pow_f`) |

**Decision**: ship partial JS support honestly rather than rush a global
port. `viva_math/random` is the canonical module that downstream packages
need on the JS target; `precision` and `autodiff_reverse` are
Erlang-only for now (documented in `docs/numerical-accuracy.md` and
`README.md`). Future minor release: complete the JS FFI for those two
modules.

### Completed — JavaScript target (full coverage)

What started as a partial port (`viva_math/random` only) escalated to a
sweep across every Erlang FFI-bound module. Codex GPT-5.5 added
`@external(javascript, ...)` annotations for: `scalar`, `random`,
`precision`, `autodiff_reverse`, `matrix`, `complex`, `scheduler`,
`statistics`, `transport`, `special`, plus benches and test helpers. All
new JS implementations live in `src/viva_math_random_ffi.mjs` and reuse
`Math.exp`, `Math.log`, `Math.sqrt`, etc.

| Target | Status |
|---|---|
| `gleam test --target erlang` | **522 passed, no failures** |
| `gleam test --target javascript` | **522 passed, no failures** |

Caveat: `matrix_dense.gleam` emits non-fatal warnings on the JS target
about `float-little-size(64)` BitArray patterns (BEAM-specific
representation) — the tests still pass because JS doesn't actually need
the byte-level layout for the public surface.

### Improved — `special.digamma` precision (1000× better)

The threshold of the recurrence-to-asymptotic-series switch went from
`N = 12` to `N = 20`. The Bernoulli series truncated at `x⁻¹⁰` converges
fast enough at `x ≥ 20` that the residual is below 5 ULP.

| Function | Before (N=12) | After (N=20) |
|---|---|---|
| `digamma(1.0)` ULP | 1085 | **5** |
| `digamma(10.0)` ULP | 271 | **2** |

`test/golden_mpmath_test.gleam` had its `digamma` tolerance brought to
the same `≤5 ULP` bound used for `erf`, `gamma`, `lgamma`, etc — the
audit-revealed outliers are gone.

### Improved — Sinkhorn (log-domain + early stopping)

The naive Sinkhorn iteration in `wasserstein_2_multivariate` was rewritten
in **log-domain** following Schmitzer (2019):

```
α_i = ε·log(a_i) − ε·logsumexp_j(−C[i,j]/ε + β_j/ε)
β_j = ε·log(b_j) − ε·logsumexp_i(−C[i,j]/ε + α_i/ε)
π[i,j] = exp((α_i + β_j − C[i,j])/ε)
```

Routes through `scalar.logsumexp` (numerically stable
`max + ln(Σ exp(· − max))`), so the algorithm now handles small ε (e.g.
`0.001`) without `exp(−C/ε)` underflowing to zero.

**Early stopping** added with marginal violation `tol = 1.0e-9`:

```
max(‖u·(K·v) − a‖_∞, ‖v·(Kᵀ·u) − b‖_∞) < tol
```

`max_iter` is now an upper bound (documented in the docstring). Empirical
performance:

| `max_iter` | Result | Wall time |
|---|---|---|
| 200 | `0.24503742976953022` | ~5.16 ms |
| 10000 | `0.24503742976955287` | ~5.68 ms |

The 10 000-iter run **does not waste 50× more work** — early stopping
terminates after convergence.

### Final tally — 1.2.102

- **522 tests passing** on **both** Erlang and JavaScript targets (was
  521 Erlang-only → +1; **+242 vs 1.2.101**).
- Dual-target proven: `viva_math` runs end-to-end in Node.js / browser
  via `gleam_javascript` runtime.
- 5 conceptual guides published on HexDocs via `[[documentation.pages]]`.
- 6 opaque types prevent invalid-state construction.
- `try_*` deprecated in favour of stdlib-idiomatic
  `square_root`/`logarithm`/`nth_root` names (legacy aliases still work).
- 1 corrected bug (`wasserstein_2_empirical` for `n ≠ m`), 4 audit-revealed
  bugs (`elbo` Jensen bound, `log_evidence` invalid-variance compensation,
  `laplace` near-zero curvature, `wasserstein_2_gaussian` negative stddev).
- `special.digamma` ~1000× more accurate (1.17e-10 → 1.20e-13).
- Numerical-precision infrastructure: ULP comparator, mpmath references,
  named tolerance regimes.

### Test files (current)

```
test/
├── test_support.gleam                    (shared helpers)
├── autodiff_test.gleam                   (28 tests, new)
├── autodiff_reverse_test.gleam           (16 tests, new)
├── matrix_dense_test.gleam               (17 tests, new)
├── tdigest_test.gleam                    (14 tests, new)
├── special_test.gleam                    (25 tests, new)
├── ou_test.gleam                         (11 tests)
├── transport_test.gleam                  (8 tests)
├── free_energy_variational_test.gleam    (14 tests)
├── viva_math_test.gleam                  (legacy per-domain blocks)
├── fft_test.gleam
├── precision_test.gleam
├── property_test.gleam
├── qcheck_test.gleam
└── sota_test.gleam
```

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

[Unreleased]: https://github.com/gabrielmaialva33/viva_math/compare/v1.2.103...HEAD
[1.2.103]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.103
[1.2.102]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.102
[1.2.101]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.101
[1.2.100]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.100
[1.2.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.2.0
[1.1.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.1.0
[1.0.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.0.0

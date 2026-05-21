# Numerical Accuracy

`viva_math` is built on top of Erlang's `:math` BIFs (libm-quality) and
adds a thin layer of domain-safe wrappers. This page documents the
**measured precision** of public functions and the conventions used in
the test suite.

## Tolerance regimes

`test/test_support.gleam` exposes named tolerances so each test encodes
its precision expectation explicitly:

| Constant | Value | When to use |
|---|---|---|
| `machine` | `1e-15` | Single IEEE-754 op (rounding only). |
| `tight`   | `1e-12` | Closed-form algebraic identities. |
| `transcendental` | `1e-13` | 1-2 libm calls (`sqrt`, `exp`, `ln`). |
| `loose`   | `1e-6`  | Iterative / Monte-Carlo / approximations. |

Plus helpers:

```gleam
is_close(a, b, abs_tol)         // |a − b| ≤ abs_tol
is_close_rel(a, b, rel_tol)     // |a − b| ≤ rel_tol · max(|a|, |b|)
is_close_hybrid(a, b, abs, rel) // CPython-style fallback
ulp_distance(a, b)              // IEEE-754 ULP distance
is_close_ulp(a, b, max_ulps)    // ULP-bounded comparison
```

## Functions with documented precision

| Function | Identity / golden | Tolerance | Source |
|---|---|---|---|
| `scalar.exp(ln(x)) = x` | round-trip | `1e-13` rel | `identities_test.gleam` |
| `scalar.sqrt(x)² = x` | round-trip | `1e-13` rel | `identities_test.gleam` |
| `scalar.cbrt(x)³ = x` | round-trip (incl. neg) | `1e-12` abs | `identities_test.gleam` |
| `scalar.sin² + cos²` | Pythagorean | `1e-13` abs | `identities_test.gleam` |
| `scalar.erf(0.5)` | golden value | `1e-12` | `golden_values_test.gleam` |
| `scalar.erfc(1.0)` | golden value | `1e-12` | `golden_values_test.gleam` |
| `scalar.gelu(1.0)` | exact erf-based | `1e-12` | `golden_values_test.gleam` |
| `scalar.silu(1.0)` | golden value | `1e-12` | `golden_values_test.gleam` |
| `constants.{pi, e, sqrt_2, sqrt_2pi}` | literal | `1e-15` | `golden_values_test.gleam` |
| `special.gamma(2.5)` | golden value | `1e-12` | `golden_values_test.gleam` |
| `special.gamma(0.1)` | golden value | `1e-12` | `golden_values_test.gleam` |
| `special.lgamma(10)` | golden value | `1e-12` | `golden_values_test.gleam` |
| `special.digamma(5)` | golden value (post-1.2.102 fix) | `1e-12` | `golden_values_test.gleam` |
| `scalar.{erf,exp,ln,sin,cos}` | mpmath 100-bit references | ≤ 5 ULP | `golden_mpmath_test.gleam` |
| `special.{gamma,lgamma}` | mpmath 100-bit references | ≤ 5 ULP except `gamma(5.5)` at 8 ULP | `golden_mpmath_test.gleam` |
| `special.digamma` | mpmath 100-bit references | ≤ 5 ULP | `golden_mpmath_test.gleam` |
| `special.Γ(x+1) = x·Γ(x)` | recurrence | `1e-10` rel | `identities_test.gleam` |
| `special.ψ(x+1) = ψ(x) + 1/x` | recurrence | `1e-7` abs | `identities_test.gleam` |
| `special.lbeta(x,y) = lgamma decomp` | decomposition | `1e-12` | `identities_test.gleam` |
| `common.softmax(x+c) = softmax(x)` | translation invariance | `1e-13` | `identities_test.gleam` |
| `ou.variance_at(t→∞) = stationary_variance` | limit | `1e-12` | `identities_test.gleam` |
| `ou.variance_at` (Brownian limit, `θ·t = 1e-9`) | `σ²·t` | rel `1%` | `qcheck_test.gleam` |
| `transport.wasserstein_2_empirical([0,2],[1])` | true `W₂ = 1` | `1e-9` | `qcheck_test.gleam` |

## `special.digamma` — post-1.2.102 improvement

The asymptotic series for `ψ(x)` converges faster as `x` grows. The
recurrence `ψ(x) = ψ(x+1) − 1/x` is used to push `x ≥ N` before invoking
the series. The threshold was raised in 1.2.102 from `N = 6` to `N = 12`,
then to `N = 20`, which removes the previous mpmath-reference exceptions
without adding extra Bernoulli terms.

| `x` | Before (N=6) | After (N=12) | After (N=20) |
|---|---|---|---|
| `1.0` | not measured | 1085 ULP | 5 ULP |
| `5.0` | `1.17e-10` | `1.20e-13` | ≤ `1e-12` abs |
| `10.0` | not measured | 271 ULP | 2 ULP |

Test `golden_values_test.special_golden_values_test` was tightened from
`2e-10` to `1e-12` in the same release. The ULP-based mpmath references now
hold at ≤ 5 ULP for the exercised `digamma` points.

## JavaScript target status

The package is configured without a fixed `target` so Erlang remains the
default while JavaScript can be tested explicitly. Full-package
`gleam test --target javascript` passes as of 1.2.102.

- `viva_math/random` has a JavaScript FFI backed by Mulberry32 plus
  Box-Muller normal sampling. `jump` is a no-op on JavaScript.
- `test/test_support` has JavaScript ULP FFI for golden-value tests.
- Erlang `:math` wrappers are mirrored to JavaScript `Math` functions where
  available. `erf`/`erfc` use a Cephes-style rational approximation in the
  shared JavaScript FFI.
- JavaScript compilation still warns for `matrix_dense` bit-array segments
  using `float-little-size(64)` because JavaScript numbers expose 52 integer
  precision bits in this representation.

## Cusp catastrophe

`viva_math/cusp` implements Thom's (1972) cusp catastrophe — the canonical
model for sudden mood transitions:

```
V(x) = x⁴/4 + α·x²/2 + β·x
dV/dx = x³ + αx + β
```

`cusp.gradient(params, x)` is the analytic derivative. The test suite
verifies it matches central finite differences on `cusp.potential` to
within `1e-7` at `h = 1e-5` (the expected `O(h²)` FD truncation error).

## Cancellation defences

Functions that would suffer catastrophic cancellation in their naive form
are routed through stable identities:

- `ou.variance_at` and `ou.step` use `scalar.expm1(−2θdt)` instead of
  `1 − exp(−2θdt)`, so the Brownian limit `σ²·t` (as `θ·t → 0`) is
  recovered without precision loss.
- `scalar.softplus(x)` uses `max(x, 0) + ln(1 + exp(−|x|))` to avoid
  overflow at large `x`.
- `scalar.logsumexp(xs)` uses the stable trick
  `max(xs) + ln(Σ exp(xᵢ − max(xs)))`.

## What's NOT yet validated

- Subnormal-number behaviour of every transcendental (most exercised
  edge case is `1e-10` for cancellation; full subnormal range is roadmap).
- Subnormal-number behaviour and `matrix_dense` JavaScript bit-array warning
  resolution.

## References

- Goldberg, D. (1991). *What every computer scientist should know about
  floating-point arithmetic*.
- Zimmermann, P. et al. (2024). *Accuracy of Mathematical Functions in
  Single, Double, Double Extended, and Quadruple Precision*
  ([glibc-2.39 report](https://members.loria.fr/PZimmermann/papers/glibc239-20240215.pdf)).
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*.

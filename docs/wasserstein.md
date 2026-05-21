# Wasserstein Distance

`viva_math/transport` computes Wasserstein distances between
one-dimensional empirical distributions and applies the construction
componentwise over the PAD axes.

## What it's for

Comparing **populations of affective states** — e.g., is today's distribution
of user moods statistically different from last week's? Unlike KL or
Jensen-Shannon, Wasserstein operates in the **same units as your data**
(if pleasures are in `[-1, 1]`, the distance is in pleasure-units), and it
gives a stable metric even when the supports of the two distributions
barely overlap.

## API

```gleam
import viva_math/transport
import viva_math/distributions
import viva_math/vector

let p = [0.1, 0.5, 0.7, 0.9]
let q = [0.2, 0.4, 0.6, 0.8]

// Earth-mover distance.
let assert Ok(w1) = transport.wasserstein_1_empirical(p, q)

// Quadratic kernel.
let assert Ok(w2) = transport.wasserstein_2_empirical(p, q)

// Closed form for Gaussians: √((μ₁−μ₂)² + (σ₁−σ₂)²)
let g1 = distributions.Gaussian(mean: 0.0, stddev: 1.0)
let g2 = distributions.Gaussian(mean: 2.0, stddev: 1.5)
let w2_gauss = transport.wasserstein_2_gaussian(g1, g2)

// Componentwise W₂ across PAD axes.
let p_pads = [vector.pad(0.1, 0.2, 0.3), vector.pad(-0.4, 0.5, 0.0)]
let q_pads = [vector.pad(0.0, 0.1, 0.2), vector.pad(-0.3, 0.4, 0.1)]
let assert Ok(w_pad) = transport.wasserstein_pad(p_pads, q_pads)
```

## Definitions

For 1D empirical samples sorted as `p_(1) ≤ … ≤ p_(n)` and `q_(1) ≤ … ≤ q_(m)`:

```
W_p^p(P, Q) = ∫_0^1 |F_P⁻¹(u) − F_Q⁻¹(u)|^p du
```

For equal sample sizes (`n = m`) this reduces to a sorted pairwise sum:

```
W_2(P, Q) = √((1/n) · Σ (p_(i) − q_(i))²)
```

For unequal sizes, the inverse-CDF integral is evaluated over the union
of quantile breakpoints `{i/n} ∪ {j/m}` in `O(n + m)` after sorting.

## Important: the W₁/W₂ duality is asymmetric

A common pitfall:

```
W_1(P, Q) = ∫_ℝ |F_P(x) − F_Q(x)| dx        ✓ holds (absolute-value identity)
W_2(P, Q) = √(∫_ℝ (F_P(x) − F_Q(x))² dx)   ✗ DOES NOT HOLD
```

`viva_math 1.2.102` fixed a bug in `wasserstein_2_empirical` for unequal
sample sizes that used the (wrong) CDF-based form. Counterexample:

```
P = [0, 2], Q = [1]
True W_2 = 1.0    (from ∫(F⁻¹_P − F⁻¹_Q)² du = 0.5·1 + 0.5·1 = 1)
Old CDF form = √(0.5) ≈ 0.707    ← wrong
```

The new implementation uses the inverse-CDF form, which is correct for
all `p` (including `p = 2`).

## `wasserstein_pad` — pseudo-metric on joints

`wasserstein_pad` computes:

```
D(P, Q) = √(W_2²(P_P, Q_P) + W_2²(P_A, Q_A) + W_2²(P_D, Q_D))
```

This is the Euclidean norm of the per-axis marginal distances — equivalent
to the **Sliced Wasserstein** along the canonical PAD basis. It satisfies
the triangle inequality (Minkowski) but is **not** the true multivariate
W₂: two joint distributions with identical marginals and different
correlations are tied at distance 0. Useful as a fast lower bound on the
true multivariate W₂; tight when marginals are independent.

For the true multivariate W₂, you'd need to solve an optimal-transport
assignment problem with cost `‖x − y‖²` (roadmap).

## Complexity

`O((n + m)·log(n + m))` dominated by the sort. The post-sort walk over the
breakpoint union is linear in `n + m`.

## See also

- [`viva_math/entropy`](https://hexdocs.pm/viva_math/viva_math/entropy.html)
  — KL, JS, Rényi divergences (information-theoretic distances).
- [`viva_math/distributions`](https://hexdocs.pm/viva_math/viva_math/distributions.html)
  — closed-form Gaussian/Laplace/Cauchy used by `wasserstein_2_gaussian`.

## References

- Villani, C. (2008). *Optimal Transport: Old and New*.
- Peyré, G., & Cuturi, M. (2019). *Computational Optimal Transport*.
- Bonneel, N., et al. (2015). *Sliced and Radon Wasserstein Barycenters of
  Measures*.

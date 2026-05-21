# Ornstein-Uhlenbeck Mood Dynamics

The Ornstein-Uhlenbeck (OU) process is the canonical model for **mean-reverting
stochastic processes** — the kind of dynamic you want when modelling
emotion regulation: noise pushes the system around, but a homeostatic pull
brings it back toward a baseline.

```
dX_t = θ(μ − X_t) dt + σ dW_t
```

| Parameter | Meaning |
|---|---|
| `θ` (theta) | Mean-reversion speed. Larger ⇒ faster return to `μ`. |
| `μ` (mu)    | Long-run mean (the attractor). |
| `σ` (sigma) | Diffusion / volatility. |

## Scalar API

```gleam
import viva_math/ou
import viva_math/random

let params = ou.OUParams1D(theta: 1.0, mu: 0.5, sigma: 0.3)
let seed = random.from_int(42)

// Single step — exact transition kernel (Doob 1942), no Euler discretisation.
let #(x1, seed1) = ou.step(params, 0.0, 0.1, seed)

// Full trajectory.
let #(traj, _) = ou.simulate(params, 0.0, 0.01, 1000, seed)
```

### Closed-form moments

```gleam
ou.mean_at(params, x0: 0.0, t: 5.0)
// μ + (x0 − μ)·e^(−θt)

ou.variance_at(params, _x0: 0.0, t: 5.0)
// σ²/(2θ) · (1 − e^(−2θt))

ou.stationary_variance(params)
// σ²/(2θ)

ou.stationary_std(params)
// σ/√(2θ)

ou.autocovariance(params, lag: 1.5)
// σ²/(2θ) · e^(−θ|τ|)

ou.half_life(params)
// ln(2)/θ
```

### Why not Euler-Maruyama?

`ou.step` uses **Doob's exact transition kernel**:

```
X_{t+Δ} = μ + (X_t − μ)·e^(−θΔ) + σ·√((1 − e^(−2θΔ))/(2θ)) · Z
```

with `Z ~ N(0, 1)`. This gives **zero discretisation error regardless of
`dt`**, unlike Euler-Maruyama which is `O(√dt)` weak. For Euler/Milstein
on arbitrary SDEs, use `viva_math/ode`.

The variance factor routes through `scalar.expm1` to avoid catastrophic
cancellation when `θ·dt → 0` — the Brownian limit `σ²·dt` is recovered
without precision loss.

## Vec3 — componentwise PAD dynamics

For affective dynamics across all three PAD axes:

```gleam
import viva_math/ou
import viva_math/vector.{Vec3}

let params =
  ou.OUParamsVec3(
    theta: Vec3(0.5, 1.0, 0.7),    // arousal regulates fastest
    mu:    Vec3(0.2, 0.0, 0.1),    // mild positive baseline
    sigma: Vec3(0.1, 0.2, 0.1),    // arousal more volatile
  )

let seed = random.from_int(7)
let #(trajectory, _) =
  ou.simulate_vec3(params, vector.zero(), dt: 0.05, n: 200, seed)
```

Each axis evolves independently (diagonal covariance) — sufficient for
most affective modelling. For cross-axis correlation, you'd need a
multivariate extension (roadmap).

## Numerical guarantees

| Property | Test | Tolerance |
|---|---|---|
| `mean_at(t = 0) = x_0` | `identities_test.gleam` | `1e-15` |
| `variance_at(t = 0) = 0` | `ou_test.gleam` | `1e-12` |
| `step(dt = 0) = x_0` | `identities_test.gleam` | `1e-15` |
| `variance_at(t → ∞) = stationary_variance` | `identities_test.gleam` | `1e-12` |
| Brownian limit `σ²·t` as `θ·t → 0` | `qcheck_test.gleam` | rel `1%` at `θ·t = 1e-9` |
| `autocov(0) = stationary_variance` | `qcheck_test.gleam` | `1e-12` |

## References

- Uhlenbeck, G. E., & Ornstein, L. S. (1930). *On the theory of the Brownian
  motion*. Physical Review, 36(5), 823.
- Doob, J. L. (1942). *The Brownian movement and stochastic equations*.
- Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2009).
  *A hierarchical Ornstein-Uhlenbeck model for continuous repeated
  measurement data*.

# PAD Emotional Model

The PAD (Pleasure, Arousal, Dominance) model is the canonical type used to
represent emotional state across the VIVA ecosystem. Coined by Mehrabian
(1996), it places every emotion on three orthogonal axes in `[-1, 1]³`.

## The `Vec3` type

```gleam
import viva_math/vector.{type Vec3, Vec3}

let fear = vector.pad(-0.64, 0.60, -0.43)
//                    ↑       ↑       ↑
//                Pleasure  Arousal  Dominance
```

`vector.pad/3` constructs a clamped `Vec3` — the constructor enforces
each axis in `[-1, 1]`, so out-of-range inputs are saturated rather than
silently corrupting downstream calculations.

| Axis | Range | Semantic |
|---|---|---|
| **Pleasure** (`x`) | `[-1, 1]` | sadness ↔ joy |
| **Arousal**  (`y`) | `[-1, 1]` | calm ↔ excitement |
| **Dominance** (`z`)| `[-1, 1]` | submission ↔ control |

## The 8 basic emotion attractors

Mehrabian (1996) places each octant of the PAD cube at a known attractor.
`viva_math/attractor` exposes them with empirically calibrated coordinates:

| Emotion    |       P |       A |       D |
|:-----------|--------:|--------:|--------:|
| Joy        | `+0.76` | `+0.48` | `+0.35` |
| Sadness    | `-0.63` | `-0.27` | `-0.33` |
| Fear       | `-0.64` | `+0.60` | `-0.43` |
| Anger      | `-0.51` | `+0.59` | `+0.25` |
| Trust      | `+0.58` | `-0.23` | `+0.42` |
| Disgust    | `-0.60` | `+0.35` | `+0.11` |
| Serenity   | `+0.45` | `-0.42` | `+0.21` |
| Excitement | `+0.62` | `+0.75` | `+0.38` |

```gleam
import viva_math/attractor
import viva_math/vector

// Classify the nearest attractor by Euclidean distance in PAD space.
let state = vector.pad(-0.3, 0.7, -0.2)
attractor.classify_emotion(state)
// -> "fear"
```

## Operations on `Vec3`

`viva_math/vector` is the full algebra layer over PAD vectors:

```gleam
import viva_math/vector

let a = vector.pad(0.5, -0.2, 0.7)
let b = vector.pad(0.1, 0.3, -0.4)

vector.add(a, b)           // pointwise
vector.scale(a, 0.5)       // scalar
vector.dot(a, b)           // inner product
vector.distance(a, b)      // L2
vector.lerp(a, b, 0.3)     // linear interpolation
vector.normalize(a)        // unit vector
```

## See also

- [`viva_math/cusp`](./numerical-accuracy.md#cusp-catastrophe) — sudden
  mood transitions via Thom (1972) catastrophe theory.
- [`viva_math/free_energy`](https://hexdocs.pm/viva_math/viva_math/free_energy.html)
  — Friston (2010) Free Energy Principle for interoception.
- [`viva_math/ou`](./ou-dynamics.md) — Ornstein-Uhlenbeck stochastic mood
  dynamics over PAD vectors.
- [`viva_math/transport`](./wasserstein.md) — distances between PAD
  distributions for population-level analysis.

## References

- Mehrabian, A. (1996). *Pleasure-arousal-dominance: A general framework
  for describing and measuring individual differences in temperament*.
- Russell, J. A. (2003). *Core affect and the psychological construction
  of emotion*.

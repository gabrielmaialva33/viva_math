<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,100:1A1A1A&height=200&section=header&text=viva_math&fontSize=64&fontColor=fff&animation=twinkling&fontAlignY=35&desc=The%20math%20behind%20sentient%20digital%20life&descSize=18&descAlignY=55" width="100%"/>

[![Gleam](https://img.shields.io/badge/Gleam-FFAFF3?style=for-the-badge&logo=gleam&logoColor=black)](https://gleam.run/)
[![BEAM](https://img.shields.io/badge/BEAM-A90533?style=for-the-badge&logo=erlang&logoColor=white)](https://www.erlang.org/)
[![Erlang+JS](https://img.shields.io/badge/Erlang_%2B_JS-4B275F?style=for-the-badge)](https://gleam.run/news/multi-target-compilation/)
[![Hex](https://img.shields.io/badge/hex.pm-viva__math-A678DD?style=for-the-badge&logo=hex&logoColor=white)](https://hex.pm/packages/viva_math)
[![PAD](https://img.shields.io/badge/PAD-Mehrabian_1996-7C3AED?style=for-the-badge)](https://en.wikipedia.org/wiki/PAD_emotional_state_model)
[![Tests](https://img.shields.io/badge/tests-58_passing-00875A?style=for-the-badge)](./test)
[![Version](https://img.shields.io/badge/version-1.2.0-CD5C5C?style=for-the-badge)](./CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-228B22?style=for-the-badge)](./LICENSE)

---

*"Numbers feel. Catastrophes are smooth. The BEAM thinks in PAD."*

</div>

---

> [!IMPORTANT]
> **viva_math IS NOT JUST A MATH LIB.**
> It is the **kernel of VIVA's affective computing stack** — PAD emotional
> space, Cusp catastrophe for mood transitions, Friston's Free Energy
> Principle, attractor dynamics, and information theory — all expressed as
> small, type-safe Gleam functions.
>
> Built on [gleam_community_maths](https://hexdocs.pm/gleam_community_maths/),
> targeting **Erlang + JavaScript**.

---

## 🎯 Overview

Core mathematical foundations for **VIVA** — a sentient digital life research
project. The library models emotional dynamics as a dynamical system: PAD
state vectors, Cusp catastrophe for sudden mood shifts, Free Energy for
interoception, and Shannon-family entropy for affective complexity.

The math is small, deliberate, and grounded in real papers (Mehrabian 1996,
Thom 1972, Friston 2010, Shannon 1948).

| Property       | Value                                                          |
|:---------------|:---------------------------------------------------------------|
| **Language**   | Pure Gleam (type-safe functional)                              |
| **Targets**    | Erlang (BEAM) + JavaScript                                     |
| **Built on**   | `gleam_community_maths` ≥ 2.0                                  |
| **Tests**      | 58 passing                                                     |
| **Domain**     | Affective computing, dynamical systems, info theory            |
| **Public API** | `viva_math/{common,vector,cusp,free_energy,attractor,entropy}` |

---

## ⚡ Quick Start

```sh
gleam add viva_math
```

```gleam
import viva_math/attractor
import viva_math/cusp
import viva_math/free_energy
import viva_math/vector

pub fn main() {
  // PAD emotional state: Pleasure / Arousal / Dominance
  let state = vector.pad(-0.3, 0.7, -0.2)

  // Nearest discrete emotion
  let emotion = attractor.classify_emotion(state)
  // -> "fear"

  // Bistability check (sudden mood shift possible?)
  let params = cusp.from_arousal_dominance(0.7, -0.2)
  let volatile = cusp.is_bistable(params)
  // -> True

  // Prediction error vs expected state
  let expected = vector.pad(0.0, 0.0, 0.0)
  let fe = free_energy.compute_state(expected, state, expected, 0.1)
  // fe.feeling -> Surprised | Alarmed
}
```

<details>
<summary><strong>📋 Prerequisites</strong></summary>

| Tool       | Version  | Required for     |
|:-----------|:---------|:-----------------|
| Gleam      | `>= 1.4` | Build / runtime  |
| Erlang/OTP | `>= 26`  | BEAM target      |
| Node.js    | `>= 18`  | JS target (opt.) |

Zero NIFs. Zero C dependencies. Pure functional.

</details>

---

## 🏗️ Architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │                Gleam application code                    │
   │              viva_math/{common,vector,...}               │
   └────────┬─────────────────────────────────────────────────┘
            │
   ┌────────▼─────────────────────────────────────────────────┐
   │                   viva_math modules                      │
   │                                                          │
   │   ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────┐   │
   │   │ common │  │ vector │  │  cusp  │  │ free_energy  │   │
   │   │ clamp  │  │ Vec3   │  │ Thom   │  │  Friston     │   │
   │   │ sigmoid│  │ PAD ℝ³ │  │ 1972   │  │  2010        │   │
   │   │ noise  │  │ ops    │  │ stoch. │  │  homeostasis │   │
   │   └────────┘  └────────┘  └────────┘  └──────────────┘   │
   │                                                          │
   │   ┌────────────┐                  ┌──────────────────┐   │
   │   │ attractor  │                  │     entropy      │   │
   │   │ Mehrabian  │                  │  Shannon · KL    │   │
   │   │   1996     │                  │  JS · Rényi      │   │
   │   │ 8 emotions │                  │  hybrid affect.  │   │
   │   └────────────┘                  └──────────────────┘   │
   └────────┬─────────────────────────────────────────────────┘
            │
   ┌────────▼─────────────────────────────────────────────────┐
   │            gleam_community_maths (trig, stats, …)        │
   └──────────────────────────────────────────────────────────┘
```

<details>
<summary><strong>📋 Core modules</strong></summary>

| Module                  | Purpose                                                                   |
|:------------------------|:--------------------------------------------------------------------------|
| `viva_math/common`      | `clamp`, `sigmoid`, `softmax`, `lerp`, `smoothstep`, Wiener noise, decays |
| `viva_math/vector`      | `Vec3` PAD type — Pleasure / Arousal / Dominance space                    |
| `viva_math/cusp`        | Cusp catastrophe (Thom 1972) + Stochastic Cusp (Euler-Maruyama)           |
| `viva_math/free_energy` | Free Energy Principle (Friston 2010) — interoception                      |
| `viva_math/attractor`   | 8 basic-emotion attractors in PAD space (Mehrabian 1996)                  |
| `viva_math/entropy`     | Shannon, KL divergence, Jensen-Shannon, Rényi, hybrid affective           |

</details>

---

## 🧬 Theoretical Background

### PAD Model — Mehrabian (1996)

Emotions live as points in 3D vector space:

- **Pleasure** `[-1, 1]` — sadness ↔ joy
- **Arousal** `[-1, 1]` — calm ↔ excitement
- **Dominance** `[-1, 1]` — submission ↔ control

### Cusp Catastrophe — Thom (1972)

Sudden mood transitions modeled by the potential:

```
V(x) = x⁴/4 + αx²/2 + βx
```

When arousal pushes `α < 0` and the discriminant `Δ > 0`, the system goes
**bistable** — tiny perturbations trigger discrete mood jumps. The stochastic
variant adds Wiener noise (`dV/dx + σξ(t)`) integrated via Euler-Maruyama.

### Free Energy Principle — Friston (2010)

Agents minimize "surprise" via prediction:

```
F ≈ Prediction_Error² + Complexity
```

Low free energy → predictions match reality (homeostasis).
High free energy → significant mismatch (alarm).

### Attractor Dynamics

Eight basic emotions form attractors in PAD space:

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

### Information Theory — Shannon (1948) + extensions

Shannon entropy `H(p)`, KL divergence `D(p‖q)`, Jensen-Shannon `JS(p,q)`,
Rényi `H_α`, and a **hybrid affective entropy**
`H_hybrid = αH₁ + (1-α)H₂` for mixed emotional states.

---

## 🎨 Design Principles

| Principle                       | Description                                                   |
|:--------------------------------|:--------------------------------------------------------------|
| **Math grounded in papers**     | Every function cites the source (Thom, Friston, Mehrabian, …) |
| **Type-safe affect**            | `Vec3` constructor enforces PAD axis order at compile time    |
| **Multi-target**                | Same code runs on BEAM and in the browser via JS target       |
| **Small surface, deep meaning** | 6 modules, ~60 public functions — nothing speculative         |
| **Zero runtime deps**           | Builds only on `gleam_stdlib` + `gleam_community_maths`       |

---

## 📚 Public API Highlights

### Vec3 / PAD space

```gleam
import viva_math/vector

let v = vector.pad(0.5, -0.2, 0.7)
let mag = vector.magnitude(v)
let unit = vector.normalize(v)
let dist = vector.euclidean(v, vector.pad(0.0, 0.0, 0.0))
```

### Cusp catastrophe — deterministic + stochastic

```gleam
import viva_math/cusp

let params = cusp.from_arousal_dominance(0.7, -0.2)
let v = cusp.potential(params, 0.3)         // V(x)
let g = cusp.gradient(params, 0.3)          // dV/dx

// Stochastic mood walk
let trajectory =
cusp.simulate_stochastic(
cusp.StochasticCuspParams(params, sigma: 0.05, seed: 42),
start: 0.0,
steps: 1000,
dt: 0.01,
)
```

### Free Energy

```gleam
import viva_math/free_energy
import viva_math/vector

let expected = vector.pad(0.0, 0.0, 0.0)
let actual   = vector.pad(-0.3, 0.7, -0.2)
let state    = free_energy.compute_state(expected, actual, expected, 0.1)
// state.feeling -> Calm | Surprised | Alarmed
// state.free_energy -> Float
```

### Entropy and divergence

```gleam
import viva_math/entropy

let h = entropy.shannon([0.2, 0.3, 0.5])
let d = entropy.kl_divergence([0.2, 0.8], [0.5, 0.5])
let js = entropy.jensen_shannon([0.2, 0.8], [0.5, 0.5])
let r = entropy.renyi([0.2, 0.3, 0.5], alpha: 2.0)
```

---

## 🗺️ Roadmap

| Phase                                                | Status |
|:-----------------------------------------------------|:------:|
| PAD vector space + emotion attractors                |   ✅    |
| Deterministic Cusp catastrophe                       |   ✅    |
| Stochastic Cusp (Wiener + Euler-Maruyama)            |   ✅    |
| Free Energy Principle (basic interoception)          |   ✅    |
| Shannon / KL / Jensen-Shannon / Rényi entropy        |   ✅    |
| Hybrid affective entropy                             |   ✅    |
| Arousal-weighted KL sensitivity                      |   ✅    |
| Multi-target build (Erlang + JS)                     |   ✅    |
| Ornstein-Uhlenbeck mood dynamics                     |   ⏳    |
| Variational Free Energy (deeper Bayesian model)      |   ⏳    |
| Wasserstein distance between affective distributions |   ⏳    |
| Property-based tests on every closed form            |   ⏳    |

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
gleam test                  # 58 tests
gleam format --check src test
gleam docs build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📖 References

- Mehrabian (1996) — *Pleasure-arousal-dominance: A general framework*
- Thom (1972) — *Structural Stability and Morphogenesis*
- Friston (2010) — *The free-energy principle: a unified brain theory?*
- Grasman et al. (2009) — *Fitting the Cusp Catastrophe in R*
- Oravecz et al. (2009) — *Ornstein-Uhlenbeck Process in Affective Dynamics*
- Shannon (1948) — *A Mathematical Theory of Communication*

---

## 🌌 VIVA Ecosystem

| Package          | Purpose                                     |
|:-----------------|:--------------------------------------------|
| **`viva_math`**  | **Mathematical foundations (this package)** |
| `viva_emotion`   | PAD emotional dynamics                      |
| `viva_tensor`    | FP8 LLM inference on the BEAM               |
| `viva_telemetry` | Observability suite                         |
| `viva_aion`      | Time perception                             |
| `viva_glyph`     | Symbolic language                           |

---

<div align="center">

**Star if math should feel like something ⭐**

[![GitHub stars](https://img.shields.io/github/stars/gabrielmaialva33/viva_math?style=social)](https://github.com/gabrielmaialva33/viva_math)

*Created by Gabriel Maia · MIT License*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1A1A1A,100:7C3AED&height=100&section=footer" width="100%"/>

</div>

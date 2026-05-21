# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

`viva_math` is the **mathematical kernel of the VIVA ecosystem** — pure Gleam,
BEAM-targeted, **zero NIFs**, **zero C deps**. It sits between `gleam_stdlib`
(primitives) and downstream packages like `viva_emotion`, `viva_tensor`,
`viva_aion`. Every formula here cites a paper (Mehrabian 1996, Thom 1972,
Friston 2010, Shannon 1948, …) — keep that contract.

## Build / test / dev commands

```bash
gleam deps download         # install deps
gleam build                 # compile
gleam test                  # run all 280+ tests (gleeunit + qcheck)
gleam format src test       # required before commit
gleam check                 # type check without compile artifacts
gleam docs build            # generate hexdocs locally
```

### Running a single test

`gleeunit` discovers any `*_test` function. Filter via env var:

```bash
gleam test -- --filter clamp_test          # run only this test
gleam test -- --filter cusp                # run any test matching "cusp"
```

### Benchmarks (not regular tests)

```bash
gleam run -m viva_math/bench_ode          # ODE solver accuracy/cost comparison
gleam run -m viva_math/bench_precision    # numerical precision audit
```

### Examples convention

Files in `examples/` are **not built by default** — they're tutorial scripts.
To run one, **copy it into `src/`** and then:

```bash
cp examples/pad_dynamics.gleam src/
gleam run -m pad_dynamics
```

## Architecture

Module layout under `src/viva_math/` is grouped by concern, not alphabetically.
When adding new code, place it in the right layer:

| Layer | Modules | Notes |
|-------|---------|-------|
| **Foundations** | `scalar`, `constants`, `common` | `scalar` wraps Erlang `:math` BIFs (post-1.2.101 — no `gleam_community_maths` dep). All transcendentals (`sin`, `cos`, `log2`, `cbrt`, `erf`, `gelu`, `silu`, …) live here. |
| **Geometry / LA** | `vector` (Vec3 PAD), `vec2`, `vec4`, `vecn`, `matrix`, `matrix_dense`, `quaternion`, `complex` | `Vec3` is the canonical PAD type — Pleasure/Arousal/Dominance, constructor clamps to `[-1,1]`. |
| **Stochastic / inference** | `random` (opaque `Seed`, `:rand` backed), `statistics`, `distributions`, `entropy`, `tdigest` | `random.Seed` is opaque on purpose — never expose ints to callers; thread `Seed` through functions. |
| **Dynamical systems** | `cusp`, `free_energy`, `attractor`, `ode`, `calculus`, `scheduler` | `ode` exposes Euler / RK2 / RK4 / RKF45 / DOP54 + Euler-Maruyama / Milstein. |
| **Specialized** | `special`, `precision`, `fft`, `autodiff`, `autodiff_reverse` | `precision` is the numerical-correctness audit harness. |

Top-level `src/viva_math.gleam` is a **facade** — only re-exports the most
common functions (`pad`, `classify`, `entropy`, `sigmoid`, `erf`, `gelu`,
`free_energy`, `is_volatile`) and holds the `version` constant. Don't bloat it.

### Single FFI file

`src/viva_math_random_ffi.erl` exists only to give `random.Seed` deterministic
behavior across BEAM/JS targets. If you touch it, also update the JS shim
in the same module's `@external` annotations.

## Dependency contract (post-1.2.101)

Runtime deps are exactly **`gleam_stdlib`** — nothing else. If you reach for
`gleam_community_maths` or any other math lib, **stop**: route through
`viva_math/scalar` (Erlang `:math` BIFs) or `viva_math/constants` instead.
The 1.2.101 refactor was specifically to drop that dep — do not reintroduce.

Dev deps: `gleeunit`, `qcheck` only. Property tests live in
`test/property_test.gleam` and `test/qcheck_test.gleam`.

## Versioning + release

This package uses the **VIVA ecosystem versioning scheme `X.Y.NNN`** (three-digit
patch as a build counter), not strict semver patch. Current: `1.2.101`.
Bump in three places, atomic commit:

1. `gleam.toml` → `version = "..."`
2. `src/viva_math.gleam` → `pub const version = "..."`
3. `CHANGELOG.md` → new section with today's date

Tag as `vX.Y.NNN`, push tag, then `gleam publish` to Hex.pm.

## Conventions worth knowing

- **Floating-point comparisons in tests**: use the `is_close` helper, not
  `should.equal` for `Float`. Direct equality on floats is a flake.
- **Doc comments**: `////` for module-level, `///` for items. Public functions
  should cite the source paper inline when implementing a named algorithm.
- **PAD axis order is sacred**: `pad(pleasure, arousal, dominance)`. The `Vec3`
  constructor is opaque to enforce this — don't pattern-match on raw fields,
  use the accessors.
- **Constructors must be unique within a module** (Gleam rule). Several
  `*_test.gleam` files import types from multiple modules — keep an eye on
  collisions.
- **No `let assert` in production** — only in tests. Production code uses
  explicit `case` with all branches handled.

## VIVA ecosystem context

`viva_math` is a leaf dep — many other packages depend on it. Breaking changes
ripple. Order of upgrade: `viva_math` → `viva_emotion` / `viva_telemetry` /
`viva_tensor` → `viva_aion` / `viva_glyph` → main `viva` repo. Always run the
downstream `gleam test` of `viva_emotion` and `viva_tensor` before publishing
a change to the public API here.

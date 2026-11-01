# AGENTS.md

Instructions for coding agents (Codex, Claude Code, OpenCode, Gemini CLI) working in this repository. `CLAUDE.md` only imports this file.

## What this package is

`viva_math` is the **mathematical kernel of the VIVA ecosystem** — pure Gleam,
BEAM-targeted, **zero NIFs**, **zero C deps**. It sits between `gleam_stdlib`
(primitives) and downstream packages like `viva_emotion`, `viva_tensor`,
`viva_aion`. Every formula here cites a paper (Mehrabian 1996, Thom 1972,
Friston 2010, Shannon 1948, …) — keep that contract.

- Version: see `gleam.toml` and `pub const version` in `src/viva_math.gleam`
  (VIVA `X.Y.NNN` scheme — see "Versioning + release").
- Tests: 553 (2026-11-01, Gleam 1.17.0 / OTP 28 — all passing, format clean).

## Build / test / dev commands

```bash
gleam deps download              # install deps
gleam build                      # compile
gleam check                      # type check without build artifacts
gleam test                       # full gleeunit + qcheck suite
gleam test -- --filter cusp      # only tests whose name matches "cusp"
gleam format --check src test    # CI gate — must pass before commit/PR
gleam format src test            # fix formatting
gleam docs build                 # local hexdocs
```

### Benchmarks (not regular tests)

```bash
gleam run -m viva_math/bench_ode          # ODE solver accuracy/cost comparison
gleam run -m viva_math/bench_precision    # numerical precision audit
```

### Examples convention

Files in `examples/` (`active_inference`, `autodiff_demo`, `pad_dynamics`) are
tutorial scripts and are **not built by default**. To run one, copy it into
`src/` and `gleam run -m <module_name>`; don't leave the copy behind.

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
referenced by the same module's `@external` annotations. Tests have their own
shims in `test/test_support_ffi.{erl,mjs}`.

## Dependency contract (post-1.2.101)

Runtime deps are exactly **`gleam_stdlib`** — nothing else. If you reach for
`gleam_community_maths` or any other math lib, **stop**: route through
`viva_math/scalar` (Erlang `:math` BIFs) or `viva_math/constants` instead.
The 1.2.101 refactor was specifically to drop that dep — do not reintroduce.

Dev deps: `gleeunit`, `qcheck` only. Property tests live in
`test/property_test.gleam` and `test/qcheck_test.gleam`.

## Versioning + release

This package uses the **VIVA ecosystem versioning scheme `X.Y.NNN`** (three-digit
patch as a build counter), not strict semver patch. Bump in three places, one
atomic commit:

1. `gleam.toml` → `version = "..."`
2. `src/viva_math.gleam` → `pub const version = "..."`
3. `CHANGELOG.md` → new section with today's date

Tag as `vX.Y.NNN`, push the tag, then `gleam publish` to Hex.pm
(`.github/workflows/release.yml` also runs on tags).

## Conventions worth knowing

- **Floating-point comparisons in tests**: use the `is_close` helper from
  `test/test_support.gleam`, not `should.equal` on `Float`. Direct equality on
  floats is a flake.
- **Doc comments**: `////` for module-level, `///` for items. Public functions
  implementing a named algorithm cite the source paper, standard, or trusted
  implementation inline.
- **PAD axis order is sacred**: `pad(pleasure, arousal, dominance)`. The `Vec3`
  constructor is opaque to enforce this — don't pattern-match on raw fields,
  use the accessors.
- **Constructors must be unique within a module** (Gleam rule). Several
  `*_test.gleam` files import types from multiple modules — keep an eye on
  collisions.
- **No `let assert` in production** — only in tests. Production code uses
  explicit `case` with all branches handled.
- Test functions must be named `*_test`. Add targeted tests for new public
  APIs, edge cases, numerical tolerances, and cross-target behavior when FFI
  is touched.
- Two-space indentation, `gleam format src test` before submitting changes
  (Claude Code runs it via hook; Codex must run it explicitly).

## Commit & pull request guidelines

History follows conventional prefixes: `feat(module): ...`, `fix(module): ...`,
`test(scope): ...`, `docs(scope): ...`, `refactor(scope): ...`, `ci: ...`.
Pull requests should describe the change, include mathematical references for
numerical behavior, note downstream impact on VIVA packages, and confirm
`gleam format --check src test` and `gleam test` pass. Update `CHANGELOG.md`
for user-facing API or behavior changes.

## CI

`.github/workflows/`: `ci.yml` (matrix OTP 27.0 + 28.0, Gleam 1.16.0 —
format check, build, test), `bench.yml`, `release.yml` (OTP 28 / Gleam 1.16,
publishes on tag).

## VIVA ecosystem context

`viva_math` is a leaf dep — many other packages depend on it. Breaking changes
ripple. Order of upgrade: `viva_math` → `viva_emotion` / `viva_telemetry` /
`viva_tensor` → `viva_aion` / `viva_glyph` → main `viva` repo. Current
downstream pins: `viva_emotion >= 1.2.100`, `viva_tensor >= 1.2.103`,
`viva >= 1.2.0`. Always run the downstream `gleam test` of `viva_emotion` and
`viva_tensor` before publishing a change to the public API here.

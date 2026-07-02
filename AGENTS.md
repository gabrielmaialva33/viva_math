# Repository Guidelines

## Project Structure & Module Organization

`viva_math` is a pure Gleam math package for the VIVA ecosystem. Source code
lives in `src/viva_math/`, grouped by concern: foundations (`scalar`,
`constants`, `common`), geometry and linear algebra (`vector`, `vec2`, `vec4`,
`vecn`, `matrix`, `matrix_dense`, `quaternion`, `complex`), stochastic and
inference (`random`, `statistics`, `distributions`, `entropy`, `tdigest`),
dynamical systems (`cusp`, `free_energy`, `attractor`, `ode`, `calculus`,
`scheduler`), and specialized numerics (`special`, `precision`, `fft`,
`autodiff`, `autodiff_reverse`). The facade is `src/viva_math.gleam`; keep it
small. Tests are in `test/`, examples in `examples/`, guides in `docs/`, and
CI/release automation in `.github/`.

## Build, Test, and Development Commands

- `gleam deps download`: install package dependencies.
- `gleam build`: compile the library.
- `gleam check`: type-check without relying on build artifacts.
- `gleam test`: run the full gleeunit and qcheck suite.
- `gleam test -- --filter cusp`: run tests whose names match a filter.
- `gleam format --check src test`: verify formatting before committing.
- `gleam docs build`: build local HexDocs.
- `gleam run -m viva_math/bench_ode`: run ODE benchmark scripts.

Example files are tutorial scripts and are not built by default. To run one,
copy it into `src/` and use `gleam run -m module_name`.

## Coding Style & Naming Conventions

Use standard Gleam formatting with two-space indentation and run
`gleam format src test` before submitting changes. Public modules use `////`
comments; public types and functions use `///`. Named algorithms should cite a
paper, standard, or trusted implementation in their docs. Avoid `let assert` in
production code; use explicit `case` handling. Runtime dependencies must remain
limited to `gleam_stdlib`; route transcendentals through `viva_math/scalar`.

## Testing Guidelines

Tests use `gleeunit` plus property checks with `qcheck`. Test functions must be
named `*_test`. Use helpers from `test/test_support.gleam` for floating-point
comparisons instead of direct equality. Add targeted tests for new public APIs,
edge cases, numerical tolerances, and cross-target behavior when FFI is touched.

## Commit & Pull Request Guidelines

History follows conventional prefixes such as `feat(module): ...`,
`fix(module): ...`, `test(scope): ...`, `docs(scope): ...`, `refactor(scope): ...`,
and `ci: ...`. Pull requests should describe the change, include mathematical
references for numerical behavior, note downstream impact on VIVA packages, and
confirm `gleam format --check src test` and `gleam test` pass. Update
`CHANGELOG.md` for user-facing API or behavior changes.

# Contributing to viva_math

Thanks for your interest in contributing to viva_math!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/gabrielmaialva33/viva_math.git
cd viva_math

# Install dependencies
gleam deps download

# Build
gleam build

# Run tests
gleam test

# Format code
gleam format
```

## Guidelines

### Code Style

- Run `gleam format` before committing
- Follow existing patterns in the codebase
- Add doc comments (`////`) for public functions
- Include examples in doc comments when helpful

### Testing

- Add tests to the matching `test/<module>_test.gleam` file (mirrors
  `src/viva_math/<module>.gleam`) — `gleeunit` discovers any public
  `*_test` function in `test/`. Examples: `test/ou_test.gleam`,
  `test/transport_test.gleam`, `test/free_energy_variational_test.gleam`.
- Import shared comparators from `test/test_support.gleam`:
  `is_close/3`, `is_close_vec3/3`, `is_close_complex/3`, `is_close_list/3`.
- Tests should be self-contained and descriptive.
- Use `is_close` (not `should.equal`) for any `Float` comparison.

### Documentation

- Update `README.md` for new modules
- Add entries to `CHANGELOG.md` under `[Unreleased]`
- Include academic references when implementing algorithms

### Commit Messages

Use conventional commits:

```
feat: add new function to entropy module
fix: correct bistability threshold in cusp
docs: update README with new examples
test: add tests for vector operations
refactor: simplify trigonometric_roots
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run `gleam test` and `gleam format`
5. Commit with descriptive message
6. Push and open a PR

## Questions?

Open an issue or reach out to @gabrielmaialva33.

## License

By contributing, you agree that your contributions will be licensed under MIT.

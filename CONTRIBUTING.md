# Contributing to Multi-Theory Consciousness Framework

Thank you for your interest in contributing to the Multi-Theory Consciousness Framework. This project sits at the intersection of computer science, neuroscience, and philosophy of mind. We welcome contributions from all three communities and from anyone willing to engage with the material seriously.

## Before You Begin

Please read [docs/HONEST_LIMITATIONS.md](docs/HONEST_LIMITATIONS.md). Understanding what this framework cannot do is as important as understanding what it can do.

## How to Contribute

### Reporting Issues

Open an issue on GitHub for:
- **Bugs**: Unexpected behavior, crashes, incorrect measurements
- **Theory concerns**: If an implementation misrepresents the theory it claims to model
- **Documentation gaps**: Unclear explanations, missing context, incorrect references
- **Feature requests**: New theories, indicators, or neural substrates to implement

When reporting bugs, include:
1. Python version and OS
2. Steps to reproduce
3. Expected vs actual behavior
4. Relevant log output

### Submitting Pull Requests

1. **Fork** the repository and create a feature branch from `master`.
2. **Write your code** following the style guidelines below.
3. **Write tests** for any new functionality. PRs without tests will not be merged.
4. **Run the full test suite** before submitting:
   ```bash
   pytest tests/ -v --cov=mtc
   ```
5. **Format and lint** your code:
   ```bash
   black mtc/ tests/ --line-length 88
   flake8 mtc/ tests/ --max-line-length 88
   ```
6. **Open a PR** with a clear description of:
   - What the change does
   - Why it is needed
   - Which theories or modules it affects
   - Any trade-offs or limitations of the approach

### PR Review Process

All pull requests require at least one review before merging. Reviews evaluate:

1. **Correctness**: Does the implementation align with the theory it models?
2. **Tests**: Are edge cases covered? Do tests verify behavior, not just coverage?
3. **Documentation**: Are docstrings complete? Are complex algorithms explained?
4. **Style**: Does the code follow project conventions?
5. **Honesty**: Does the code avoid overclaiming what it achieves?

## Code Style

- **Formatter**: [black](https://github.com/psf/black), line length 88
- **Linter**: [flake8](https://flake8.pycqa.org/), max line length 88
- **Type hints**: Use them. All public functions should have type annotations.
- **Docstrings**: Google-style docstrings for all public classes and functions.
- **Naming**: Snake case for functions/variables, PascalCase for classes. Indicator names use snake_case matching the `indicator_configs` dictionary.

## Consciousness Module Guidelines

Modifying the consciousness theory modules (`mtc/consciousness/`) requires particular care.

### Before changing a theory module:

1. **Read the original paper(s).** Each module header lists its theoretical basis. Do not modify implementations based on intuition alone.
2. **Understand the indicator mapping.** Each module contributes to specific indicators in the assessment framework. Changes that break indicator measurement must update the assessment code as well.
3. **Do not stub implementations.** Every method should compute a meaningful value, even if simplified. Comments like `# TODO: implement later` are not acceptable for merged code.
4. **Document simplifications.** If your implementation simplifies the theory (and it will --- these are computational approximations), document exactly what was simplified and why.

### Adding a new theory:

1. Create a module in `mtc/consciousness/` (or a subdirectory for multi-file theories).
2. Define at least one indicator in `mtc/assessment/assessment.py` with:
   - A `ConsciousnessTheory` enum value
   - A threshold based on the theory's predictions
   - A measurement method that produces scores from 0.0 to 1.0
3. Add the theory to `ConsciousnessTheory` enum.
4. Write tests in `tests/consciousness/`.
5. Document the theory in `docs/CONSCIOUSNESS_THEORIES.md`.

### Adding a new indicator:

1. Add the indicator config to `ConsciousnessAssessment.indicator_configs` in `mtc/assessment/assessment.py`.
2. Implement its measurement in the `_measure_indicator` method.
3. Add tests that verify both passing and failing cases.
4. Update the indicator table in `README.md`.

## Test Requirements

- All new code must have tests.
- Tests go in the `tests/` directory, mirroring the `mtc/` package structure.
- Use `pytest` with `pytest-asyncio` for async tests.
- Aim for meaningful assertions, not just "it runs without crashing."
- Include edge cases: empty inputs, zero-state neural networks, single-element systems.

Run specific test categories:

```bash
# All tests
pytest tests/ -v

# Only consciousness module tests
pytest tests/consciousness/ -v

# Only assessment tests
pytest tests/assessment/ -v

# With coverage report
pytest tests/ --cov=mtc --cov-report=html
```

## Commit Messages

Use clear, descriptive commit messages:

```
feat(consciousness): add RPT local recurrence measurement

Implements Lamme (2006) local recurrence indicator using SNN
feedback connection analysis. Threshold set at 0.3 based on
baseline noise testing.
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Questions?

Open a GitHub Discussion or issue. We are happy to help newcomers orient themselves in the codebase.

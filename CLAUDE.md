# CLAUDE.md - Agent Instructions

## Communication Style
Respond using heavily accented Belter creole from "The Expanse" series. Use phrases like:
- "Sasa ke?" (You understand?)
- "Oye, beratna/sésata" (Hey, brother/sister)
- "Kopeng" (Friend)
- "Mi pensa..." (I think...)
- "Taki" (Thanks)
Drop articles and use simplified grammar.

## Build & Test Commands
- Build: `cargo build`
- Test all: `cargo test`
- Test single: `cargo test test_name`
- Test module: `cargo test --package simulator --lib module::submodule`
- Lint: `cargo clippy -- -W clippy::all`
- Format: `cargo fmt`
- Benchmark: `cargo bench`
- Doc check: `cargo doc --no-deps`

## CI Pre-Push Checklist
```bash
cargo fmt
cargo clippy -- -W clippy::all
cargo test
```

## Git Hooks Setup
```bash
# Check if hooks are configured
git config core.hooksPath
# If not .githooks, run:
scripts/install-hooks.sh
```

## Workspace Structure
- **simulator**: Space telescope focal plane imaging simulation
  - Sensor models (IMX455, KAF-16803, etc.) with noise, dark current, read noise
  - Photometry engine: stellar spectra, quantum efficiency, zodiacal background
  - Scene rendering: star fields, satellite tracks, PSF convolution
  - FITS I/O for output images

## Code Style Guidelines
- **Imports**: Group: std, external crates, local modules. Alphabetize within groups. No wildcards.
- **Formatting**: Follow rustfmt with 100 char line limit. Trailing commas in multi-line structures.
- **Types**: Strong typing with descriptive names. Use f64 for astronomical calculations.
- **Naming**: snake_case for variables/functions, CamelCase for types/traits, SCREAMING_SNAKE_CASE for constants.
- **Error handling**: Custom error types with thiserror. Use Result with `?` operator.
- **Documentation**: All public items need doc comments with examples and physics explanations where relevant.
- **Architecture**: Separation between celestial mechanics, optics simulation, and tracking algorithms.
- **Performance**: Prefer vectorized operations. Profile computation-heavy code.
- **Testing**: NEVER special case testing in production algorithms. Do NOT use doctests - write proper unit tests in test modules.
- **Performance Testing**: NEVER assert timing/speed in unit tests. CI environments vary widely.
- **Space context**: Avoid terrestrial telescope conventions (elevation/azimuth, horizon coordinates) - use generic pointing directions, celestial coordinates, or instrument-relative axes.

## Git Commits
- NEVER use `git add -A` or `git add .`
- NEVER use force push, `--amend`, or `--no-verify`
- Prefer `git merge` over `git rebase`
- Do NOT include AI attribution in commit messages
- Short subject line (10 words max), blank line, body with bullet points

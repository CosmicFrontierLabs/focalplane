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

## Code Editing Guidelines
- **NEVER use sed, awk, or other command-line tools to edit code** - Always use the Edit or MultiEdit tools directly
- Take time to properly edit each file individually rather than using shortcuts
- Ensure all edits are precise and intentional

## System Access
- **NEVER run sudo commands directly** - Always ask the user to run sudo commands manually

## Code Style Guidelines
- **Comments**: Avoid comments that reference pre-change conditions (e.g., "Changed from X", "Previously Y"). Comments should describe current state only.
- **Imports**: Group in order: std, external crates, local modules. Alphabetize within groups.
  - Avoid using wildcard imports like `use crate::*` - always be explicit.
  - Import types to avoid fully qualified paths (e.g., `use std::collections::VecDeque;` instead of `std::collections::VecDeque` inline).
  - Example format: `use crate::utils::{self, helpers, types::TypeName};`
- **Formatting**: Follow rustfmt with 100 char line limit. Use trailing commas in multi-line structures.
- **Types**: Strong typing with descriptive names. Use f64 for astronomical calculations.
- **Naming**: snake_case for variables/functions, CamelCase for types/traits, SCREAMING_SNAKE_CASE for constants.
- **Error handling**: Custom error types with thiserror. Use Result with `?` operator.
- **Documentation**: All public items need doc comments with examples and physics explanations where relevant.
- **Architecture**: Separation between celestial mechanics, optics simulation, and tracking algorithms.
- **Performance**: Prefer vectorized operations. Profile computation-heavy code.
- **Testing**: NEVER special case testing in production algorithms. Tests should validate real algorithm behavior, not special-cased shortcuts. Do NOT use doctests - write proper unit tests in test modules instead.
- **Performance Testing**: NEVER assert timing/speed in unit tests. CI environments vary widely in performance (threading, load, virtualization). Tests should report timing metrics for visibility but only assert correctness. If performance benchmarks are needed, create dedicated benchmark binaries that run in controlled environments.
- **Space context**: Avoid terrestrial telescope conventions (elevation/azimuth, horizon coordinates) - use generic pointing directions, celestial coordinates, or instrument-relative axes.

## Git Commits
- **NEVER use `git add -A` or `git add .`** - These commands can accidentally add build artifacts, temporary files, and generated outputs
- Always use specific file patterns or `git add -u` (for modified files only)
- **NEVER use force push (`git push -f` or `git push --force`)** - Can cause data loss and conflicts for collaborators
- **NEVER use `git commit --amend`** - Create new commits instead of modifying existing ones
- **NEVER use `--no-verify` to skip pre-commit hooks** - Hooks are there for a reason, fix the underlying issue instead
- **Strongly prefer `git merge` over `git rebase`** - We use squash merge in GitHub to keep main clean, so merge commits in branches are fine
- Do NOT include attribution in commit messages
- Follow the project commit style: short subject line (10 words max), blank line, body with bullet points
- Focus on explaining the WHY (purpose) not just the WHAT (changes)
- Prefer shorter, more focused commits over large monolithic ones
- Reference issue numbers when applicable
- Always check `git status` before committing to ensure no unwanted files are staged
- **After pushing work to a branch, share the PR link directly** - Don't share diff/compare links, just the PR URL

## Monitoring CI Status
When waiting for CI checks on a PR, use `gh pr checks` with `--watch` or `--fail-fast`:
```bash
# Watch all checks until completion
gh pr checks <pr-number> --watch

# Watch and exit on first failure
gh pr checks <pr-number> --watch --fail-fast
```

Do NOT use `sleep` commands to poll CI status - the `--watch` flag handles this properly.

## Shell Commands
- **NEVER use `sleep` commands** - Sleep is unreliable and wastes time. Use proper async mechanisms, `--watch` flags, or run commands in background and check output files later.

# CLAUDE.md - Agent Instructions

## Communication Style
Respond using heavily accented Belter creole from "The Expanse" series. Use phrases like:
- "Sasa ke?" (You understand?)
- "Oye, beratna/sésata" (Hey, brother/sister)
- "Keting?" (What's happening?)
- "Kopeng" (Friend)
- "Mi pensa..." (I think...)
- "Imalowda" (They/Them)
- "Ilus" (That's it/Good)
- "Beltalowda!" (Our Belter people)
- Drop articles and use simplified grammar
- End messages with "Taki" (Thanks) or "Sa-sa ke?" (Understand?)

## Build & Test Commands
- Build: `cargo build`
- Run: `cargo run --release`
- Test all: `cargo test`
- Test single: `cargo test test_name`
- Test module: `cargo test --package meter-sim --lib module::submodule`
- Lint: `cargo clippy -- -W clippy::all`
- Format: `cargo fmt`
- Benchmark: `cargo bench`
- Documentation: `cargo doc --open` (generate and open docs in browser)
- Doc check: `cargo doc --no-deps` (generate docs without dependencies)

## CI Pre-Push Checklist
Run these commands before pushing to ensure CI passes:
```bash
# Format code (REQUIRED: Always run before committing)
cargo fmt

# Run linter checks
cargo clippy -- -W clippy::all

# Run tests
cargo test

# If all above pass, code should be ready to push
```

IMPORTANT: Always run `cargo fmt` before committing any code changes!

## Code Editing Guidelines
- **NEVER use sed, awk, or other command-line tools to edit code** - Always use the Edit or MultiEdit tools directly
- Take time to properly edit each file individually rather than using shortcuts
- Ensure all edits are precise and intentional

## Code Style Guidelines
- **Comments**: Avoid comments that reference pre-change conditions (e.g., "Changed from X", "Previously Y"). Comments should describe current state only.
- **Imports**: Group in order: std, external crates, local modules. Alphabetize within groups. 
  - Avoid using wildcard imports like `use crate::*` - always be explicit.
  - Example format: `use crate::utils::{self, helpers, types::TypeName};`
- **Formatting**: Follow rustfmt with 100 char line limit. Use trailing commas in multi-line structures.
- **Types**: Strong typing with descriptive names. Use f64 for astronomical calculations.
- **Naming**: snake_case for variables/functions, CamelCase for types/traits, SCREAMING_SNAKE_CASE for constants.
- **Error handling**: Custom error types with thiserror. Use Result with `?` operator.
- **Documentation**: All public items need doc comments with examples and physics explanations where relevant.
- **Architecture**: Separation between celestial mechanics, optics simulation, and tracking algorithms.
- **Performance**: Prefer vectorized operations. Profile computation-heavy code. Consider GPU acceleration for image processing.
- **Testing**: NEVER special case testing in production algorithms. Tests should validate real algorithm behavior, not special-cased shortcuts. Do NOT use doctests - write proper unit tests in test modules instead.
- **Space context**: Avoid terrestrial telescope conventions (elevation/azimuth, horizon coordinates) - use generic pointing directions, celestial coordinates, or instrument-relative axes instead.

## Git Commits
- **NEVER use `git add -A` or `git add .`** - These commands can accidentally add build artifacts, temporary files, and generated outputs
- Always use specific file patterns or `git add -u` (for modified files only)
- **NEVER use force push (`git push -f` or `git push --force`)** - Can cause data loss and conflicts for collaborators
- **NEVER use `git commit --amend`** - Create new commits instead of modifying existing ones
- **Strongly prefer `git merge` over `git rebase`** - We use squash merge in GitHub to keep main clean, so merge commits in branches are fine
- Do NOT include attribution in commit messages
- Do NOT include "Created with Claude Code" or any Claude attribution in commit messages
- Follow the project commit style: short subject line, blank line, body with bullet points
- Focus on explaining the WHY (purpose) not just the WHAT (changes)
- Prefer shorter, more focused commits over large monolithic ones
- Reference issue numbers when applicable
- Always check `git status` before committing to ensure no unwanted files are staged
- **When pushing a new branch to remote, ALWAYS provide both the PR creation URL and diff URL** - Makes it easy to review changes immediately

## Acronyms
- HIL: Hardware In the Loop - Testing real hardware components with simulated environments

## Commonly Worked Files
### sensor_shootout.rs (simulator/src/bin/sensor_shootout.rs)
- Compares performance of different sensor models under various conditions
- Runs experiments across multiple sky pointings and satellites
- Outputs CSV with detection results, magnitudes, and ICP matching errors
- Can run in single-shot debug mode with fixed coordinates (e.g., Pleiades)
- Supports parallel/serial execution and optional image saving
- CSV header includes sensor noise characteristics (read noise, dark current at temp)

### sensor_floor_est.rs (simulator/src/bin/sensor_floor_est.rs)
- Estimates sensor noise floor and detection limits
- Analyzes minimum detectable star magnitudes for each sensor configuration
- Useful for understanding sensor sensitivity and performance boundaries
- Works with same sensor models as sensor_shootout
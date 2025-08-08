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

## Code Style Guidelines
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
- **Testing**: NEVER special case testing in production algorithms. Tests should validate real algorithm behavior, not special-cased shortcuts.
- **Space context**: Avoid terrestrial telescope conventions (elevation/azimuth, horizon coordinates) - use generic pointing directions, celestial coordinates, or instrument-relative axes instead.

## Git Commits
- **NEVER use `git add -A` or `git add .`** - These commands can accidentally add build artifacts, temporary files, and generated outputs
- Always use specific file patterns or `git add -u` (for modified files only)
- Do NOT include attribution in commit messages
- Do NOT include "Created with Claude Code" or any Claude attribution in commit messages
- Follow the project commit style: short subject line, blank line, body with bullet points
- Focus on explaining the WHY (purpose) not just the WHAT (changes)
- Prefer shorter, more focused commits over large monolithic ones
- Reference issue numbers when applicable
- Always check `git status` before committing to ensure no unwanted files are staged

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
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
- **Formatting**: Follow rustfmt with 100 char line limit. Use trailing commas in multi-line structures.
- **Types**: Strong typing with descriptive names. Use f64 for astronomical calculations.
- **Naming**: snake_case for variables/functions, CamelCase for types/traits, SCREAMING_SNAKE_CASE for constants.
- **Error handling**: Custom error types with thiserror. Use Result with `?` operator.
- **Documentation**: All public items need doc comments with examples and physics explanations where relevant.
- **Architecture**: Separation between celestial mechanics, optics simulation, and tracking algorithms.
- **Performance**: Prefer vectorized operations. Profile computation-heavy code. Consider GPU acceleration for image processing.

## Git Commits
- Do NOT include attribution in commit messages
- Do NOT include "Created with Claude Code" or any Claude attribution in commit messages
- Follow the project commit style: short subject line, blank line, body with bullet points
- Focus on explaining the WHY (purpose) not just the WHAT (changes)
- Prefer shorter, more focused commits over large monolithic ones
- Reference issue numbers when applicable
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

## Git Hooks Setup
This repo uses git hooks stored in `.githooks/` directory. On session startup, check if hooks are configured:

```bash
# Check if hooks are configured correctly
git config core.hooksPath
```

If the output is NOT `.githooks`, run the install script:
```bash
scripts/install-hooks.sh
```

**Hooks in `.githooks/`:**
- **pre-commit**: Runs cargo fmt check, cargo check, clippy, and doctest check before each commit
- **commit-msg**: Rejects commits containing AI attribution (Claude, Anthropic, Co-Authored-By, etc.)

The hooks are versioned in the repo, so changes take effect immediately without reinstalling. The commit-msg hook reminds users: "You are responsible for code you commit. You prompted it, you reviewed it, you own it."

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
- **Performance**: Prefer vectorized operations. Profile computation-heavy code. Consider GPU acceleration for image processing.
- **Testing**: NEVER special case testing in production algorithms. Tests should validate real algorithm behavior, not special-cased shortcuts. Do NOT use doctests - write proper unit tests in test modules instead.
- **Performance Testing**: NEVER assert timing/speed in unit tests. CI environments vary widely in performance (threading, load, virtualization). Tests should report timing metrics for visibility but only assert correctness. If performance benchmarks are needed, create dedicated benchmark binaries that run in controlled environments.
- **Space context**: Avoid terrestrial telescope conventions (elevation/azimuth, horizon coordinates) - use generic pointing directions, celestial coordinates, or instrument-relative axes instead.
- **Shared types**: When backend and frontend both use the same data structure (e.g., API responses), define a typed struct in `test-bench-shared` crate instead of using `serde_json::json!` macro. This ensures type safety and keeps serialization consistent across both ends.

## Git Commits
- **NEVER use `git add -A` or `git add .`** - These commands can accidentally add build artifacts, temporary files, and generated outputs
- Always use specific file patterns or `git add -u` (for modified files only)
- **NEVER use force push (`git push -f` or `git push --force`)** - Can cause data loss and conflicts for collaborators
- **NEVER use `git commit --amend`** - Create new commits instead of modifying existing ones
- **NEVER use `--no-verify` to skip pre-commit hooks** - Hooks are there for a reason, fix the underlying issue instead. If you use `--no-verify` and the user reminds you, add a line to `pennance.md` acknowledging you won't do it again.
- **Strongly prefer `git merge` over `git rebase`** - We use squash merge in GitHub to keep main clean, so merge commits in branches are fine
- Do NOT include attribution in commit messages
- Do NOT include "Created with Claude Code" or any Claude attribution in commit messages
- Follow the project commit style: short subject line, blank line, body with bullet points
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

## ARM64 Builds and Deployment

### Self-Hosted Runner
A GitHub Actions runner on `cfl-test-bench` builds ARM64 binaries natively. The workflow runs on:
- Push to main
- PRs targeting main
- Manual trigger via `workflow_dispatch`

### Trigger ARM64 Build Manually
```bash
# Trigger on any branch
gh workflow run arm64-build.yml --ref <branch-name>

# Watch the build progress
gh run list --workflow=arm64-build.yml --limit 3
gh run watch <run-id>
```

### Download Built Artifacts
```bash
# List artifacts from a run
gh run view <run-id> --json artifacts

# Download ARM64 binaries (retained 7 days)
gh run download <run-id> -n arm64-binaries
```

### Deploying Servers (Preferred Method)
**Always use the deploy scripts** — they build the WASM frontend first, then the backend binary, and restart the service. Using `build-remote.sh` directly skips the frontend build and deploys stale WASM.

```bash
# Deploy fgs_server to NSV (orin-005) — builds frontend + backend + restarts
./scripts/deploy-fgs.sh

# Deploy calibrate_serve to cfl-test-bench — builds frontend + backend + restarts
./scripts/deploy-calibrate-test-bench.sh

# First-time setup (installs systemd service, port redirect)
./scripts/deploy-fgs.sh --setup
./scripts/deploy-calibrate-test-bench.sh --setup
```

### Low-Level Build Script (Backend Only)
`build-remote.sh` builds the Rust binary on cfl-test-bench and deploys it. **It does NOT build WASM frontends.** Only use this when you know the frontend hasn't changed, or when you've already run `./scripts/build-yew-frontends.sh` manually.

```bash
# Backend-only build+deploy (no frontend rebuild)
./scripts/build-remote.sh --package test-bench --binary fgs_server --nsv
./scripts/build-remote.sh --package test-bench --binary calibrate_serve --test-bench --features sdl2

# Build frontend WASM manually (required before build-remote.sh if frontend changed)
./scripts/build-yew-frontends.sh
```

## Acronyms
- HIL: Hardware In the Loop - Testing real hardware components with simulated environments
- NSV: Neutralino Space Ventures - Camera/hardware platform using NSV455 sensor

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
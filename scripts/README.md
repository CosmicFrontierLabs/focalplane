# Git Hooks Scripts

This directory contains scripts for managing git hooks in the repository.

## Available Scripts

### install-hooks.sh
Installs pre-commit hooks that match the CI pipeline checks:
- Runs `cargo fmt` to check code formatting
- Runs `cargo clippy` with warnings as errors

Usage:
```bash
./scripts/install-hooks.sh
```

### uninstall-hooks.sh
Removes the installed pre-commit hooks.

Usage:
```bash
./scripts/uninstall-hooks.sh
```

## CI Parity
The hooks are designed to match the checks performed in CI:
- Format checking: `cargo fmt --all -- --check`
- Clippy linting: `cargo clippy -- -D warnings`

This ensures that code passing local checks will also pass CI checks.

## Bypassing Hooks
If you need to commit without running the hooks (not recommended):
```bash
git commit --no-verify
```
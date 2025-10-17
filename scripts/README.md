# Development Scripts

This directory contains scripts for git hooks, building, and deploying to Jetson Orin.

## Build & Deployment Scripts

### build-arm64.sh
Cross-compile packages for ARM64 (Jetson Orin).

Usage:
```bash
# Build entire package
./scripts/build-arm64.sh <package-name>

# Build specific binary
./scripts/build-arm64.sh <package-name> <binary-name>

# Examples
./scripts/build-arm64.sh flight-software
./scripts/build-arm64.sh poa_cameras playerone_info
```

### deploy-to-orin.sh
Build and deploy packages to remote Jetson Orin device.

Usage:
```bash
# Deploy package and run command
./scripts/deploy-to-orin.sh --package poa_cameras --binary playerone_info --run './playerone_info --detailed'

# Deploy without building (use existing binaries)
./scripts/deploy-to-orin.sh --package flight-software --skip-build --keep-remote
```

Options:
- `--package PKG` - Package to deploy (flight-software, poa_cameras)
- `--binary BIN` - Specific binary to deploy
- `--skip-build` - Skip build step
- `--keep-remote` - Keep remote directory after deployment
- `--run CMD` - Command to run remotely

Environment variables:
- `ORIN_HOST` - Remote host (default: cosmicfrontiers@192.168.15.229)

## Git Hooks Scripts

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
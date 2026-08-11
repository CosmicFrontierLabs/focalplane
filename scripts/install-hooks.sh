#!/bin/bash
# Install git hooks for this repository

set -e

echo "Configuring git to use .githooks/ directory..."
git config core.hooksPath .githooks
echo "✅ Git hooks configured successfully!"
echo ""
echo "Hooks in .githooks/ will now run automatically:"
echo "  - pre-commit: Format check, cargo check, clippy"
echo "  - commit-msg: Reject commits with AI attribution"

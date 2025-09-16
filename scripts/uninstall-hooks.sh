#!/bin/bash
# Uninstall git pre-commit hooks
#
# This script removes the pre-commit hooks installed by install-hooks.sh

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Removing pre-commit hooks..."

# Remove pre-commit hook if it exists
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    rm "$HOOKS_DIR/pre-commit"
    echo "✅ Pre-commit hook removed."
else
    echo "No pre-commit hook found."
fi

echo "✅ Hooks uninstalled successfully!"
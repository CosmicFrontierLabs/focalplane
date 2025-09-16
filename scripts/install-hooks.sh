#!/bin/bash
# Install git pre-commit hooks for formatting and linting
#
# This script installs pre-commit hooks that match the CI pipeline checks
# to catch issues before pushing to the repository.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing pre-commit hooks..."

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook to run cargo fmt and clippy checks
#
# This matches the CI checks in .github/workflows/

set -e

echo "Checking if code is properly formatted..."

# Check if any Rust files are staged
if git diff --cached --name-only | grep -q '\.rs$'; then
    # Run cargo fmt check
    if ! cargo fmt --all -- --check > /dev/null 2>&1; then
        echo "❌ Code is not properly formatted!"
        echo "Please run 'cargo fmt' before committing."
        exit 1
    fi
    echo "✅ Code formatting is correct."
    
    # Run cargo clippy with warnings as errors (matching CI)
    echo "Running clippy checks..."
    if ! cargo clippy -- -D warnings > /dev/null 2>&1; then
        echo "❌ Clippy found issues!"
        echo "Please run 'cargo clippy -- -D warnings' to see the issues."
        exit 1
    fi
    echo "✅ Clippy checks passed."
fi

exit 0
EOF

# Make the hook executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "The pre-commit hook will:"
echo "  - Check code formatting with 'cargo fmt'"
echo "  - Run clippy linting with warnings as errors"
echo ""
echo "To bypass the hook (not recommended), use: git commit --no-verify"
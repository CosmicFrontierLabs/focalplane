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
# Pre-commit hook to run cargo fmt, check, and clippy
#
# This matches the CI checks in .github/workflows/

set -e

echo "Checking if code is properly formatted..."

# Check if any Rust files are staged
if git diff --cached --name-only | grep -q '\.rs$'; then
    # Run cargo fmt check (show output on failure, matching CI)
    fmt_output=$(cargo fmt --all -- --check 2>&1)
    if [ $? -ne 0 ]; then
        echo "❌ Code is not properly formatted!"
        echo "$fmt_output"
        echo ""
        echo "Please run 'cargo fmt --all' before committing."
        exit 1
    fi
    echo "✅ Code formatting is correct."

    # Run cargo check (matching CI)
    echo "Running cargo check..."
    check_output=$(cargo check --locked --all-targets 2>&1)
    if [ $? -ne 0 ]; then
        echo "❌ Cargo check failed!"
        echo "$check_output"
        exit 1
    fi
    echo "✅ Cargo check passed."

    # Run cargo clippy with warnings as errors (show output on failure, matching CI)
    echo "Running clippy checks..."
    clippy_output=$(cargo clippy -- -D warnings 2>&1)
    if [ $? -ne 0 ]; then
        echo "❌ Clippy found issues!"
        echo "$clippy_output"
        exit 1
    fi
    echo "✅ Clippy checks passed."

    # Check that there are no doctests (we use proper unit tests instead)
    echo "Checking for doctests (should be 0)..."
    doctest_output=$(cargo test --doc 2>&1)
    # Count lines that show non-zero test runs like "running 5 tests"
    doctest_count=$(echo "$doctest_output" | grep -E '^running [1-9][0-9]* tests?$' | wc -l)
    if [ "$doctest_count" -gt 0 ]; then
        echo "❌ Found doctests! We don't use doctests in this codebase."
        echo "$doctest_output" | grep -E '^running [1-9][0-9]* tests?$'
        echo ""
        echo "Please convert doctests to proper unit tests in #[cfg(test)] modules."
        exit 1
    fi
    echo "✅ No doctests found."
fi

exit 0
EOF

# Make the hook executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "The pre-commit hook will:"
echo "  - Check code formatting with 'cargo fmt'"
echo "  - Run 'cargo check --locked --all-targets'"
echo "  - Run clippy linting with warnings as errors"
echo "  - Verify no doctests exist (use unit tests instead)"
echo ""
echo "To bypass the hook (not recommended), use: git commit --no-verify"
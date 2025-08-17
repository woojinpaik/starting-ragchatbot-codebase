#!/bin/bash

# Code Quality Check Script
# Runs all code quality tools and reports results

set -e

echo "🔍 Running Code Quality Checks..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall success
overall_success=true

# Function to run a tool and capture results
run_check() {
    local tool_name="$1"
    local command="$2"
    local emoji="$3"
    
    echo -e "\n${emoji} Running ${tool_name}..."
    if eval "$command"; then
        echo -e "${GREEN}✅ ${tool_name} passed${NC}"
    else
        echo -e "${RED}❌ ${tool_name} failed${NC}"
        overall_success=false
    fi
}

# Run Black (check only)
run_check "Black (formatting check)" "uv run black --check --diff ." "🎨"

# Run isort (check only)
run_check "isort (import sorting check)" "uv run isort --check-only --diff ." "📚"

# Run flake8
run_check "flake8 (linting)" "uv run flake8 ." "🔍"

# Run mypy
run_check "mypy (type checking)" "uv run mypy backend/ --ignore-missing-imports" "🔬"

# Final result
echo -e "\n=================================="
if [ "$overall_success" = true ]; then
    echo -e "${GREEN}🎉 All quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}💥 Some quality checks failed. Please fix the issues above.${NC}"
    exit 1
fi
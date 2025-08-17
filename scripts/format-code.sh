#!/bin/bash

# Code Formatting Script
# Automatically formats code using black and isort

set -e

echo "🎨 Formatting Code..."
echo "===================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n📚 Sorting imports with isort..."
uv run isort .
echo -e "${GREEN}✅ Import sorting completed${NC}"

echo -e "\n🎨 Formatting code with black..."
uv run black .
echo -e "${GREEN}✅ Code formatting completed${NC}"

echo -e "\n${GREEN}🎉 Code formatting complete!${NC}"
echo -e "${YELLOW}💡 Run './scripts/quality-check.sh' to verify all quality checks pass.${NC}"
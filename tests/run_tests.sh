#!/bin/bash
set -e

echo "Running pytest with coverage..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

echo ""
echo "✅ All tests passed!"
echo "Coverage report generated: htmlcov/index.html"
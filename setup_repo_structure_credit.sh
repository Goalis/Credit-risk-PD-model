#!/bin/bash
# setup_repo_structure.sh
# Run this from inside your cloned credit-risk-pd-portfolio repo root.
# Creates the full folder structure for the retail + corporate PD portfolio.

set -e

PROJECTS=(
  "retail-pd"
  "corporate-pd"
)

echo "Creating top-level folders..."
mkdir -p common
mkdir -p scripts
mkdir -p data/credit/corporate/polish_bankruptcy
mkdir -p docs/monitoring/figures

touch common/__init__.py
for f in metrics data_utils plotting; do
  touch "common/${f}.py"
done

touch docs/monitoring/retail_pd_monitoring_report.tex
touch docs/monitoring/corporate_pd_monitoring_report.tex
touch docs/monitoring/figures/.gitkeep

# data/ is gitignored, so drop .gitkeep so the folders still exist after clone
touch data/credit/.gitkeep data/credit/corporate/polish_bankruptcy/.gitkeep

echo "Creating project folders..."
for p in "${PROJECTS[@]}"; do
  mkdir -p "$p/src" "$p/tests" "$p/figures" "$p/notebooks" "$p/docs"
  touch "$p/README.md"
  touch "$p/src/__init__.py"
  touch "$p/figures/.gitkeep"
  cp methodology_template.md "$p/docs/methodology.md" 2>/dev/null || touch "$p/docs/methodology.md"
done

echo "Done. Structure created:"
find . -maxdepth 2 -type d | sort

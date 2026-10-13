#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
mode="${1:-}"
if [[ $# -gt 1 ]] || [[ -n "$mode" && "$mode" != "--analysis" ]]; then
  echo "Usage: bash build.sh [--analysis]" >&2
  exit 2
fi
if ! command -v quarto >/dev/null 2>&1; then
  echo "Quarto is required to render the website: https://quarto.org/docs/get-started/" >&2
  exit 1
fi
if [[ "$mode" == "--analysis" ]]; then
  if [[ ! -f data/raw/guardian_ai_articles_broad.csv ]]; then
    echo "Missing local corpus. See data/README.md before running --analysis." >&2
    exit 1
  fi
  "$PYTHON" scripts/01_build_corpus.py
  "$PYTHON" scripts/03_classify_sections.py
  "$PYTHON" scripts/04_model_topics.py
fi
"$PYTHON" scripts/02_summarize_audit.py
"$PYTHON" scripts/05_build_visualizations.py
"$PYTHON" scripts/06_generate_report_text.py
quarto render --to html
touch docs/.nojekyll
echo "Website ready in docs/. Preview with: quarto preview"

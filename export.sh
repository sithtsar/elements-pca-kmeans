#!/usr/bin/env bash
# Rebuild everything in exports/ from notebook.py and slides.py. Only needs uv.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p exports

reqs="$(mktemp)"
uv export --script notebook.py --no-hashes --quiet > "$reqs"
run=(uv run --no-project --with-requirements "$reqs" --with "nbconvert[webpdf]" --with nbformat --with playwright)

"${run[@]}" playwright install chromium

# AIDS_EXPAND=1 opens every "Go deeper" box, so the static files show all results.
AIDS_EXPAND=1 "${run[@]}" marimo export pdf notebook.py -o exports/report.pdf --no-include-inputs -f
AIDS_EXPAND=1 "${run[@]}" marimo export ipynb notebook.py -o exports/notebook.ipynb --include-outputs -f
"${run[@]}" marimo export html notebook.py -o exports/notebook.html -f

"${run[@]}" marimo export html slides.py -o exports/slides.html --no-include-code -f
"${run[@]}" marimo export pdf slides.py -o exports/slides.pdf --as=slides --no-include-inputs --raster-server=static -f
echo "Done. Files are in exports/"

#!/bin/sh
# Build dist/token-analysis, a single-file macOS executable, with PyInstaller in ./.venv.
# Set PYTHON to choose the interpreter that creates the venv (default: python3).
# Set VERSION to stamp the version that --version prints (default: dev).
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
    "$PYTHON" -m venv .venv
fi
printf "VERSION = '%s'\n" "${VERSION:-dev}" > collector/_version.py
.venv/bin/python -m pip install --quiet --upgrade pip pyinstaller
.venv/bin/pyinstaller --noconfirm --clean --onefile \
    --name token-analysis \
    --paths collector \
    --add-data collector/template.html:. \
    collector/token_analysis.py
echo "built dist/token-analysis"

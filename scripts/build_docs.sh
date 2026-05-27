#!/usr/bin/env bash
# Build both documentation sites (MkDocs narrative + Sphinx API) into docs/build/.
# Used by CI and for local previews.
#
# Usage:
#   bash scripts/build_docs.sh           # build both
#   bash scripts/build_docs.sh narrative # build only the MkDocs narrative
#   bash scripts/build_docs.sh api       # build only the Sphinx API
#
# Outputs:
#   docs/build/narrative/   # MkDocs Material site
#   docs/build/api/         # Sphinx HTML

set -euo pipefail

cd "$(dirname "$0")/.."

TARGET="${1:-both}"
DOCS_DIR="docs"
BUILD_DIR="${DOCS_DIR}/build"

build_narrative() {
    echo "▶ Building MkDocs narrative …"
    (cd "${DOCS_DIR}" && mkdocs build --strict)
    echo "  → output: ${BUILD_DIR}/narrative"
}

build_api() {
    echo "▶ Building Sphinx API reference …"
    sphinx-build -W -b html "${DOCS_DIR}/api" "${BUILD_DIR}/api"
    echo "  → output: ${BUILD_DIR}/api"
}

case "${TARGET}" in
    narrative) build_narrative ;;
    api)       build_api ;;
    both)      build_narrative; build_api ;;
    *)         echo "Unknown target: ${TARGET}"; exit 1 ;;
esac

echo "✓ Documentation build complete."

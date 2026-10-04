#!/usr/bin/env bash
# Entry point. Run from anywhere:  ./reproduce.sh check | route1 | route2-check | route2-smoke | route2-fetch
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
PYTHON=${PYTHON:-python}
case "${1:-help}" in
    check)  exec "$PYTHON" -m route1.prepare ;;
    route1) exec "$PYTHON" -m route1.run ;;
    route2-fetch) exec "$PYTHON" -m route2.fetch "${@:2}" ;;
    route2-check) exec "$PYTHON" -m route2.run checks "${@:2}" ;;
    route2-smoke) exec "$PYTHON" -m route2.run smoke ;;
    *)
        echo "Usage:"
        echo "  ./reproduce.sh check    check the Python environment and every supplied file"
        echo "  ./reproduce.sh route1   rebuild and check all figures and tables of the paper"
        echo "  ./reproduce.sh route2-fetch   download and verify the large Route 2 inputs"
        echo "  ./reproduce.sh route2-check   replay every stage of the pipeline and compare with the published results"
        echo "  ./reproduce.sh route2-smoke   tiny end-to-end runs of the Route 2 code"
        exit 1 ;;
esac

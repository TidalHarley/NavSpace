#!/usr/bin/env bash
# Launch the Flask + SocketIO annotation UI from the NavSpace repo root.
# Use this if you are not already inside NavSpace-main:
#   bash /path/to/NavSpace-main/run_annotation_server.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python annotation_pipeline/websocket_annotation_server.py "$@"

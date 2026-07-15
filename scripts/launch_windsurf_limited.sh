#!/usr/bin/env bash
set -euo pipefail

# Launch Windsurf inside a systemd scope with strict memory limits.
# MemoryHigh: start throttling allocations
# MemoryMax: hard cap — OOM kill the cgroup before the whole system freezes

MEMORY_HIGH="${MEMORY_HIGH:-7G}"
MEMORY_MAX="${MEMORY_MAX:-8G}"

systemd-run --user --scope \
    -p MemoryHigh="$MEMORY_HIGH" \
    -p MemoryMax="$MEMORY_MAX" \
    windsurf "$@"

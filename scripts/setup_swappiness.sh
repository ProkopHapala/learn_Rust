#!/usr/bin/env bash
set -euo pipefail

# Reduce swappiness so Linux prefers dropping caches over aggressive swapping
# This reduces disk thrashing when memory pressure is high

echo "Current swappiness: $(cat /proc/sys/vm/swappiness)"

# Apply temporarily
sudo sysctl vm.swappiness=10

# Make permanent
SYSCTL_CONF="/etc/sysctl.d/99-custom-swappiness.conf"
if ! grep -q '^vm.swappiness' "$SYSCTL_CONF" 2>/dev/null; then
    echo "vm.swappiness = 10" | sudo tee "$SYSCTL_CONF" >/dev/null
    echo "Wrote persistent config to $SYSCTL_CONF"
else
    echo "$SYSCTL_CONF already contains vm.swappiness — update manually if needed"
fi

echo "New swappiness: $(cat /proc/sys/vm/swappiness)"

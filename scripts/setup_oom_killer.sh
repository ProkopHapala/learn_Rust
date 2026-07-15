#!/usr/bin/env bash
set -euo pipefail

# Install and enable earlyoom (lightweight user-space OOM killer)
# Prevents runaway processes (e.g. clangd) from freezing the system

if command -v apt &>/dev/null; then
    sudo apt update && sudo apt install -y earlyoom
elif command -v dnf &>/dev/null; then
    sudo dnf install -y earlyoom
elif command -v pacman &>/dev/null; then
    sudo pacman -S --needed earlyoom
else
    echo "Unsupported package manager. Install earlyoom manually." >&2
    exit 1
fi

sudo systemctl enable --now earlyoom
systemctl status earlyoom --no-pager

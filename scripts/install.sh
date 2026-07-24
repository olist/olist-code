#!/usr/bin/env bash
# Installs the latest olist-code binary release.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.sh | bash
set -euo pipefail

REPO="olist/olist-code"
INSTALL_DIR="${OLIST_CODE_INSTALL_DIR:-$HOME/.local/bin}"
BIN_NAME="olist-code"

os="$(uname -s)"
case "$os" in
  Linux) asset="olist-code-linux-x86_64" ;;
  Darwin) asset="olist-code-macos-arm64" ;;
  *)
    echo "Unsupported OS: $os" >&2
    exit 1
    ;;
esac

mkdir -p "$INSTALL_DIR"
dest="$INSTALL_DIR/$BIN_NAME"

echo "Downloading $asset..."
curl -fsSL "https://github.com/$REPO/releases/latest/download/$asset" -o "$dest"

chmod +x "$dest"

echo "Installed $BIN_NAME to $dest"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "Add it to your PATH: export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac

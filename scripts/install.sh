#!/usr/bin/env bash
# Installs the latest olist-code binary release.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.sh | bash
#
# Requires either the GitHub CLI (gh, already authenticated) or a
# GITHUB_TOKEN with read access to olist/olist-code, since this is a
# private repository.
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

if command -v gh >/dev/null 2>&1; then
  echo "Downloading $asset via gh release download..."
  gh release download --repo "$REPO" --pattern "$asset" --output "$dest" --clobber
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "Downloading $asset via GitHub API..."
  url=$(curl -fsSL \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/releases/latest" \
    | grep "\"browser_download_url\".*$asset\"" \
    | sed -E 's/.*"browser_download_url": *"([^"]+)".*/\1/')
  if [ -z "$url" ]; then
    echo "Could not find asset $asset in the latest release." >&2
    exit 1
  fi
  curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" -o "$dest" "$url"
else
  echo "Error: need either the 'gh' CLI (authenticated) or a GITHUB_TOKEN env var to download from the private repo." >&2
  exit 1
fi

chmod +x "$dest"

echo "Installed $BIN_NAME to $dest"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "Add it to your PATH: export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac

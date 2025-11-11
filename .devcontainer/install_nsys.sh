#!/usr/bin/env bash
# install_nsys.sh - install NVIDIA Nsight Systems (nsys) in a devcontainer
set -euo pipefail

# Use sudo if not root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "This script requires root privileges. Install sudo or run as root." >&2
        exit 1
    fi
fi

command_exists() { command -v "$1" >/dev/null 2>&1; }

# If already installed, exit
if command_exists nsys || command_exists nsight-systems; then
    echo "nsys already installed: $(command -v nsys || true)"
    exit 0
fi

# Try direct apt install (if package available in current repos)
echo "Attempting apt install nsight-systems..."
if $SUDO apt-get update && $SUDO apt-get install -y nsight-systems >/dev/null 2>&1; then
    echo "nsight-systems installed via apt."
    exit 0
fi

# Determine distro (Ubuntu expected). Build NVIDIA repo path like ubuntu2004, ubuntu2204, etc.
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DIST_ID=${ID:-}
    VERSION_ID=${VERSION_ID:-}
else
    echo "Cannot detect OS. Aborting." >&2
    exit 1
fi

if [ "$DIST_ID" != "ubuntu" ] && [ "$DIST_ID" != "debian" ]; then
    echo "This script supports Ubuntu/Debian based systems. Detected: $DIST_ID" >&2
    exit 1
fi

# Convert VERSION_ID "22.04" -> "2204", "20.04" -> "2004"
RELEASE_NUM=${VERSION_ID%%.*}${VERSION_ID#*.}
# safe fallback: take first two components and remove dot
RELEASE_NUM="${RELEASE_NUM//./}"
REPO_NAME="ubuntu${RELEASE_NUM}"

REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${REPO_NAME}/x86_64"

echo "Adding NVIDIA repository: $REPO_URL"

# Add NVIDIA GPG key (uses apt-key adv; if not available, try curl + gpg --dearmour)
if command_exists apt-key; then
    $SUDO apt-key adv --fetch-keys "${REPO_URL}/3bf863cc.pub" || true
fi

if [ ! -f /etc/apt/sources.list.d/nvidia-cuda.list ]; then
    echo "deb ${REPO_URL} /" | $SUDO tee /etc/apt/sources.list.d/nvidia-cuda.list >/dev/null
fi

# Update and install nsight-systems
$SUDO apt-get update
if $SUDO apt-get install -y nsight-systems; then
    echo "nsight-systems installed successfully."
    exit 0
fi

# Fallback: try package names that exist in some repos
for pkg in nsys nsys-cli; do
    if $SUDO apt-get install -y "$pkg"; then
        echo "$pkg installed successfully."
        exit 0
    fi
done

echo "Failed to install nsight-systems via apt. You may need to download and install the Nsight Systems package manually from NVIDIA:"
echo "https://developer.nvidia.com/nsight-systems"
exit 1
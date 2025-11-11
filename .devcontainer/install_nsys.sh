#!/bin/bash

set -e

# Install NVIDIA Nsight Systems
DEB_URL="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb"

wget -O /tmp/nsight-systems.deb "$DEB_URL"
sudo dpkg -i /tmp/nsight-systems.deb
rm /tmp/nsight-systems.deb
echo "NVIDIA Nsight Systems installed successfully."
nsys --version

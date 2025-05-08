#!/bin/bash

# Exit on any error
set -e

# Install Git LFS (for Debian/Ubuntu-based systems)
echo "Installing Git LFS..."
sudo apt-get install git-lfs -y

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Pull LFS-tracked files (assumes you're inside a git repo)
echo "Pulling LFS-tracked files..."
git lfs pull

# Add Docker 
echo "Adding Docker repository..."
curl -sSL https://get.docker.com/ | sudo sh

echo "Installing Docker..."
sudo groupadd -f docker; sudo usermod -aG docker $USER

echo "Done."
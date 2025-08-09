#!/bin/bash
set -e
# Install Docker & Docker Compose (Debian/Ubuntu)
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker
echo 'Docker installed.'

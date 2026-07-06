#!/usr/bin/env bash
# PACE 2026 container setup (Debian 13.5). Installs the C++17 toolchain and builds
# the solver. No other external dependencies are required.
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends g++ make
g++ -O3 -std=c++17 -pipe -pthread -o /usr/local/bin/pace2026_heuristic "$(dirname "$0")/main.cpp"
echo "Built pace2026_heuristic. Run: PACE_TIME_LIMIT=298 pace2026_heuristic < instance.nw > out.txt"

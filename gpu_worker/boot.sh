#!/usr/bin/env bash
# Starts the PhysLens GPU worker on a stock PyTorch image (no custom image build needed).
# First start: installs MapAnything into a venv on the network volume and downloads the Apache-2.0 weights there.
# Later starts: reuse the volume, so only the model load remains.
set -euo pipefail

V=/runpod-volume/physlens
READY="$V/.ready-v1"
mkdir -p "$V"
export HF_HOME="$V/hf" PIP_CACHE_DIR="$V/pip-cache"

# OpenCV (headless) needs glib; the container disk is fresh on every start, so check each time.
if ! ldconfig -p | grep -q libgthread-2.0; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends libglib2.0-0 >/dev/null
fi

if [ ! -f "$READY" ]; then
  (
    flock 9
    if [ ! -f "$READY" ]; then
      echo "[boot] first start: installing MapAnything onto the volume"
      # An interrupted earlier install (or one made with a different base image) is unusable: start clean.
      rm -rf "$V/venv"
      python3 -m venv --system-site-packages "$V/venv"
      printf '%s\n' "$PHYSLENS_CONSTRAINTS" > "$V/constraints.txt"
      # Not quiet: each package shows up in the RunPod logs, so a slow install is visibly alive.
      "$V/venv/bin/pip" install --progress-bar off -c "$V/constraints.txt" \
        "mapanything @ https://github.com/facebookresearch/map-anything/archive/refs/heads/main.zip" runpod
      echo "[boot] packages installed; downloading weights"
      "$V/venv/bin/python" -c "from mapanything.models import MapAnything; MapAnything.from_pretrained('facebook/map-anything-apache')"
      touch "$READY"
      echo "[boot] install complete"
    fi
  ) 9>"$V/.lock"
fi

printf '%s' "$PHYSLENS_HANDLER_B64" | base64 -d > /handler.py
echo "[boot] starting handler"
exec "$V/venv/bin/python" -u /handler.py

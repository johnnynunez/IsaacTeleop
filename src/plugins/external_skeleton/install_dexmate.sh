#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs Dexmate's omniteleop and its bundled dynamixelAPI so that
# dexmate_bridge.py can `import dexcomm` and subscribe to the exoskeleton
# joint stream produced by `omni-arm` (omniteleop/leader/arm_reader.py).
#
# Mirrors the repo-side Dexmate setup documented at:
#   https://docs.dexmate.ai/.../tutorial/teleoperation/teleoperation-exoskeleton/software-setup
#
# Idempotent: re-runs are safe.

set -e
set -u

echo "=== Dexmate omniteleop install ==="
echo

DEXMATE_REPO_URL="${DEXMATE_REPO_URL:-https://github.com/dexmate-ai/omniteleop}"
DEXMATE_REPO_REF="${DEXMATE_REPO_REF:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default install location: sibling of this plugin so it's gitignored alongside
# vendor-specific assets like src/plugins/manus/ManusSDK.
DEXMATE_REPO_DIR="${DEXMATE_REPO_DIR:-$SCRIPT_DIR/omniteleop}"
PIP_CMD="${PIP_CMD:-pip}"

# 1) Clone or update the omniteleop repo.
if [ -d "$DEXMATE_REPO_DIR/.git" ]; then
    echo "[1/3] Updating existing omniteleop checkout at $DEXMATE_REPO_DIR..."
    git -C "$DEXMATE_REPO_DIR" fetch --quiet origin
    git -C "$DEXMATE_REPO_DIR" checkout --quiet "$DEXMATE_REPO_REF"
    git -C "$DEXMATE_REPO_DIR" pull --quiet --ff-only origin "$DEXMATE_REPO_REF" || true
else
    echo "[1/3] Cloning $DEXMATE_REPO_URL ($DEXMATE_REPO_REF) into $DEXMATE_REPO_DIR..."
    if ! command -v git &> /dev/null; then
        echo "Error: git is required but not installed." >&2
        exit 1
    fi
    git clone --branch "$DEXMATE_REPO_REF" "$DEXMATE_REPO_URL" "$DEXMATE_REPO_DIR"
fi
echo

# 2) Install the bundled Dynamixel SDK first (omniteleop depends on it).
echo "[2/3] pip install -e $DEXMATE_REPO_DIR/dynamixelAPI ..."
if [ -d "$DEXMATE_REPO_DIR/dynamixelAPI" ]; then
    "$PIP_CMD" install -e "$DEXMATE_REPO_DIR/dynamixelAPI"
else
    echo "Warning: $DEXMATE_REPO_DIR/dynamixelAPI not found; skipping (upstream may have moved it)." >&2
fi
echo

# 3) Install omniteleop itself.
echo "[3/3] pip install -e $DEXMATE_REPO_DIR ..."
"$PIP_CMD" install -e "$DEXMATE_REPO_DIR"
echo

# Sanity-check: confirm dexcomm imports.
PYTHON_CMD="${PYTHON_CMD:-python3}"
if "$PYTHON_CMD" -c "import dexcomm; from dexcomm.codecs import DictDataCodec" 2>/dev/null; then
    echo "OK: 'import dexcomm' succeeded with $($PYTHON_CMD --version 2>&1)."
else
    echo "Warning: 'import dexcomm' failed after install. Make sure $PIP_CMD targets the same Python ($PYTHON_CMD)." >&2
fi

cat <<EOF

=== Done ===
omniteleop checkout : $DEXMATE_REPO_DIR
Run the bridge      : python3 $SCRIPT_DIR/python/dexmate_bridge.py [--port 53700]
Run the C++ plugin  : ./external_skeleton_plugin dexmate external_skeleton 127.0.0.1 53700
Don't forget        : grant USB perms to the Dynamixel port, e.g. sudo chmod 666 /dev/ttyUSB0
EOF

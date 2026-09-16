#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
clone_pinned() {
    local url="$1" dest="$2" revision="$3"
    if [ ! -e "$dest" ]; then
        git clone "$url" "$dest"
        git -C "$dest" checkout --detach "$revision"
    fi
    if [ "$(git -C "$dest" rev-parse HEAD)" != "$revision" ]; then
        echo "Unexpected revision in $dest. Use a fresh directory or check out $revision." >&2
        exit 1
    fi
    if [ -n "$(git -C "$dest" status --porcelain)" ]; then
        echo "Local changes in $dest; use a clean clone for reproducible installation." >&2
        exit 1
    fi
}
clone_pinned https://github.com/mahmoodlab/TRIDENT.git third_party/TRIDENT e4c98b2280d5b905fe83b00f01afac0f76755880
clone_pinned https://github.com/facebookresearch/segment-anything.git third_party/segment-anything dca509fe793f601edb92606367a655c15ac00fdf
python -m pip install -r requirements-preprocessing.txt -e third_party/TRIDENT -e third_party/segment-anything
python -m pip check

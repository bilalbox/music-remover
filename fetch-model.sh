#!/usr/bin/env bash
# Downloads ZFTurbo's HTDemucs-on-DnR checkpoint and exports the STFT-stripped
# OpenVINO IR into models/htdemucs_dnr/. Python (torch, demucs-infer, openvino
# 2025.4, onnx) is only needed for this one-time export — the shipped tool is Rust.
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
dest="$root/models/htdemucs_dnr"
ckpt="$dest/97d170e1-dbb4db15.th"
url="https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/97d170e1-dbb4db15.th"

mkdir -p "$dest"

if [ -f "$dest/htdemucs_fwd.xml" ]; then
    echo ">> model already present at $dest"
    exit 0
fi

if [ ! -f "$ckpt" ]; then
    echo ">> fetching 97d170e1-dbb4db15.th"
    curl -L --fail --progress-bar -o "$ckpt" "$url"
fi

python="python3"
if [ -x "$root/.venv/bin/python" ]; then
    python="$root/.venv/bin/python"
fi

echo ">> exporting OpenVINO IR (needs torch, onnx, demucs-infer, openvino==2025.4)"
"$python" "$root/scripts/export_htdemucs_dnr.py" --checkpoint "$ckpt" --dest "$dest"

echo ">> model ready at $dest"

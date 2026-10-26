#!/usr/bin/env python3
"""Run JusperLee TIGER-DnR on a mono 44.1 kHz WAV for the music-remover bakeoff."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent / "TIGER"
sys.path.insert(0, str(ROOT))

# Avoid look2hear package __init__ chains (lightning/rich/etc.) — load modules directly.
import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# activations / normalizations are plain modules with no heavy deps
_layers_pkg = type(sys)("look2hear.layers")
_layers_pkg.__path__ = [str(ROOT / "look2hear" / "layers")]
sys.modules["look2hear"] = type(sys)("look2hear")
sys.modules["look2hear"].__path__ = [str(ROOT / "look2hear")]
sys.modules["look2hear.layers"] = _layers_pkg
_load("look2hear.layers.activations", ROOT / "look2hear" / "layers" / "activations.py")
_load(
    "look2hear.layers.normalizations",
    ROOT / "look2hear" / "layers" / "normalizations.py",
)
_models_pkg = type(sys)("look2hear.models")
_models_pkg.__path__ = [str(ROOT / "look2hear" / "models")]
sys.modules["look2hear.models"] = _models_pkg
_load("look2hear.models.base_model", ROOT / "look2hear" / "models" / "base_model.py")
_tiger_dnr = _load(
    "look2hear.models.tiger_dnr", ROOT / "look2hear" / "models" / "tiger_dnr.py"
)
TIGERDNR = _tiger_dnr.TIGERDNR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="mono 44.1 kHz wav")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "tiger-dnr",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="If >0, only process the first N seconds (smoke test)",
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    # Use all cores for the bakeoff CPU path (PyTorch defaults can be conservative).
    nthreads = max(1, (os.cpu_count() or 4))
    torch.set_num_threads(nthreads)
    torch.set_num_interop_threads(min(4, nthreads))
    print(f"device={device} torch={torch.__version__} threads={nthreads}")

    wav, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != 44100:
        raise SystemExit(f"expected 44100 Hz, got {sr}")
    if wav.shape[1] != 1:
        print(f"downmixing {wav.shape[1]} ch -> mono")
        wav = wav.mean(axis=1, keepdims=True)
    if args.max_seconds > 0:
        wav = wav[: int(args.max_seconds * sr)]
    audio_secs = len(wav) / sr
    print(f"audio={audio_secs:.1f}s shape={wav.shape}")

    print("loading JusperLee/TIGER-DnR …")
    load_tic = time.perf_counter()
    model = TIGERDNR.from_pretrained(
        "JusperLee/TIGER-DnR", cache_dir=str(args.cache_dir)
    )
    model.to(device)
    model.eval()
    load_secs = time.perf_counter() - load_tic
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded in {load_secs:.1f}s  params={nparams:.2f}M")

    # Official script: torchaudio.load -> [C, T], then audio[None] -> [1, C, T]
    mixture = torch.from_numpy(wav.T).to(device)  # [1, T]
    mixture = mixture.unsqueeze(0)  # [1, 1, T]

    print("separating (dialog / effect / music; 12s chunks, 4s hop) …")
    infer_tic = time.perf_counter()
    with torch.no_grad():
        dialog, effect, music = model(mixture)
    infer_secs = time.perf_counter() - infer_tic
    rtf = infer_secs / audio_secs
    print(
        f"infer={infer_secs:.1f}s  rtf={rtf:.3f} ({(1/rtf) if rtf else float('inf'):.2f}x realtime)"
    )

    args.outdir.mkdir(parents=True, exist_ok=True)

    def save(name: str, t: torch.Tensor) -> Path:
        # t: [nch, T] or [T]
        x = t.detach().cpu().float().numpy()
        if x.ndim == 2:
            x = x.T  # [T, nch]
        else:
            x = x[:, None]
        # trim/pad to input length
        n = len(wav)
        if len(x) > n:
            x = x[:n]
        elif len(x) < n:
            x = np.pad(x, ((0, n - len(x)), (0, 0)))
        path = args.outdir / name
        sf.write(path, x, sr, subtype="FLOAT")
        return path

    dialog_p = save("tiger_dialog.wav", dialog)
    effect_p = save("tiger_effect.wav", effect)
    music_p = save("tiger_music.wav", music)

    # soundtrack = dialog + effect (music-remover default Keep)
    d = sf.read(dialog_p, dtype="float32", always_2d=True)[0]
    e = sf.read(effect_p, dtype="float32", always_2d=True)[0]
    soundtrack = d + e
    peak = float(np.max(np.abs(soundtrack)))
    if peak > 0.999:
        soundtrack *= 0.999 / peak
    soundtrack_p = args.outdir / "tiger_soundtrack.wav"
    sf.write(soundtrack_p, soundtrack, sr, subtype="FLOAT")

    def rms_db(path: Path) -> float:
        a, _ = sf.read(path, dtype="float32")
        ms = float(np.mean(a.astype(np.float64) ** 2))
        return 10.0 * np.log10(max(ms, 1e-12))

    print(
        f"stems  music {rms_db(music_p):.1f} dB  "
        f"sfx {rms_db(effect_p):.1f} dB  "
        f"speech {rms_db(dialog_p):.1f} dB"
    )
    print(f"wrote {dialog_p.name}, {effect_p.name}, {music_p.name}, {soundtrack_p.name}")


if __name__ == "__main__":
    main()

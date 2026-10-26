#!/usr/bin/env python3
"""Quick objective side-by-side for bakeoff WAVs (resampled to common SR)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr == target_sr:
        return mono
    # linear resample is fine for coarse energy metrics
    n = int(round(len(mono) * target_sr / sr))
    x = np.linspace(0, 1, len(mono), endpoint=False)
    xi = np.linspace(0, 1, n, endpoint=False)
    return np.interp(xi, x, mono).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    ms = float(np.mean(x.astype(np.float64) ** 2))
    return 10.0 * np.log10(max(ms, 1e-12))


def band_energy_db(x: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    # periodogram via rFFT; coarse music-ish band energy
    spec = np.fft.rfft(x.astype(np.float64) * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    e = float(np.mean(np.abs(spec[mask]) ** 2)) if mask.any() else 1e-12
    return 10.0 * np.log10(max(e, 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, required=True, help="original mix")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("candidates", nargs="+", type=Path)
    args = p.parse_args()

    ref = load_mono(args.ref, args.sr)
    print(
        f"{'file':40} {'dur':>6} {'rms':>7} {'Δrms':>7} "
        f"{'resid':>7} {'lo':>7} {'mid':>7} {'hi':>7}"
    )
    ref_rms = rms_db(ref)
    ref_lo = band_energy_db(ref, args.sr, 80, 400)
    ref_mid = band_energy_db(ref, args.sr, 400, 4000)
    ref_hi = band_energy_db(ref, args.sr, 4000, 8000)
    print(
        f"{'ORIGINAL':40} {len(ref)/args.sr:6.1f} {ref_rms:7.1f} {'—':>7} "
        f"{'—':>7} {ref_lo:7.1f} {ref_mid:7.1f} {ref_hi:7.1f}"
    )

    for path in args.candidates:
        x = load_mono(path, args.sr)
        n = min(len(ref), len(x))
        a, b = ref[:n], x[:n]
        resid = a - b
        print(
            f"{path.name:40} {n/args.sr:6.1f} {rms_db(b):7.1f} "
            f"{rms_db(b)-ref_rms:+7.1f} {rms_db(resid):7.1f} "
            f"{band_energy_db(b, args.sr, 80, 400)-ref_lo:+7.1f} "
            f"{band_energy_db(b, args.sr, 400, 4000)-ref_mid:+7.1f} "
            f"{band_energy_db(b, args.sr, 4000, 8000)-ref_hi:+7.1f}"
        )


if __name__ == "__main__":
    main()

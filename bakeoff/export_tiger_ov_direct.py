#!/usr/bin/env python3
"""Spike B: convert TIGER stem via OpenVINO PyTorch frontend (no ONNX).

Also tries an STFT-stripped export (host STFT/iSTFT) if the full wav graph fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import soundfile as sf
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent / "TIGER"
sys.path.insert(0, str(ROOT))

SR = 44100
CHUNK_SECS = 12.0
CHUNK_SAMPLES = int(SR * CHUNK_SECS)
WIN = 2048
HOP = 512
# center=True STFT frames for CHUNK_SAMPLES
N_FRAMES = CHUNK_SAMPLES // HOP + 1  # 1034


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_tiger_dnr(cache_dir: Path):
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
    tiger_dnr = _load(
        "look2hear.models.tiger_dnr", ROOT / "look2hear" / "models" / "tiger_dnr.py"
    )
    model = tiger_dnr.TIGERDNR.from_pretrained(
        "JusperLee/TIGER-DnR", cache_dir=str(cache_dir)
    )
    model.eval()
    return model


class TigerStemWav(nn.Module):
    """Full stem including STFT/iSTFT — wav in, stems out."""

    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem
        self.register_buffer(
            "window", torch.hann_window(stem.win, periodic=True), persistent=False
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.stem(wav)  # uses stem's own stft path


class TigerStemSpec(nn.Module):
    """STFT-stripped: complex RI spectrogram in → complex RI multi-source out.

    Input:  spec_ri [1, 2, F=1025, T=N_FRAMES]  (real, imag)
    Output: est_ri  [1, 2, num_sources, F, T]
    """

    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem

    def forward(self, spec_ri: torch.Tensor) -> torch.Tensor:
        stem = self.stem
        # spec_ri: B, 2, F, T
        B = spec_ri.shape[0]
        spec_real = spec_ri[:, 0]
        spec_imag = spec_ri[:, 1]
        # Fake complex via view — avoid torch.complex for OV friendliness where possible.
        # We'll keep RI throughout.
        subband_feature = []
        band_idx = 0
        sub_real = []
        sub_imag = []
        for i, bw in enumerate(stem.band_width):
            sr = spec_real[:, band_idx : band_idx + bw, :]  # B, bw, T
            si = spec_imag[:, band_idx : band_idx + bw, :]
            sub_real.append(sr)
            sub_imag.append(si)
            ri = torch.stack([sr, si], 1)  # B, 2, bw, T
            subband_feature.append(
                stem.BN[i](ri.reshape(B, bw * 2, -1))
            )
            band_idx += bw
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        sep_output = stem.separator(subband_feature)
        sep_output = sep_output.reshape(B, stem.nband, stem.feature_dim, -1)

        out_real = []
        out_imag = []
        for i, bw in enumerate(stem.band_width):
            this_output = stem.mask[i](sep_output[:, i]).reshape(
                B, 2, 2, stem.num_output, bw, -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            mr = this_mask[:, 0]
            mi = this_mask[:, 1]
            mr = mr - (mr.sum(1).unsqueeze(1) - 1) / stem.num_output
            mi = mi - mi.sum(1).unsqueeze(1) / stem.num_output
            # complex multiply: (sr + j si) * (mr + j mi)
            er = sub_real[i].unsqueeze(1) * mr - sub_imag[i].unsqueeze(1) * mi
            ei = sub_real[i].unsqueeze(1) * mi + sub_imag[i].unsqueeze(1) * mr
            out_real.append(er)
            out_imag.append(ei)
        est_real = torch.cat(out_real, 2)  # B, K, F, T
        est_imag = torch.cat(out_imag, 2)
        return torch.stack([est_real, est_imag], 1)  # B, 2, K, F, T


def stft_ri(wav: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(wav.astype(np.float32))
    window = torch.hann_window(WIN, periodic=True)
    spec = torch.stft(
        x,
        n_fft=WIN,
        hop_length=HOP,
        window=window,
        return_complex=True,
        center=True,
        onesided=True,
    )
    # F, T → stack RI → 1, 2, F, T
    return torch.stack([spec.real, spec.imag], 0).unsqueeze(0).numpy()


def istft_ri(est_ri: np.ndarray, length: int) -> np.ndarray:
    # est_ri: 1, 2, K, F, T
    real = torch.from_numpy(est_ri[0, 0])  # K, F, T
    imag = torch.from_numpy(est_ri[0, 1])
    spec = torch.complex(real, imag)
    window = torch.hann_window(WIN, periodic=True)
    outs = []
    for k in range(spec.shape[0]):
        outs.append(
            torch.istft(
                spec[k],
                n_fft=WIN,
                hop_length=HOP,
                window=window,
                length=length,
                center=True,
                onesided=True,
            ).numpy()
        )
    return np.stack(outs, 0)  # K, T


def convert_and_bench(pt_model: nn.Module, example, outdir: Path, tag: str, devices: list[str]):
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== convert [{tag}] via ov.convert_model(torch) ===")
    t0 = time.perf_counter()
    try:
        ov_model = ov.convert_model(pt_model, example_input=example)
    except Exception as e:
        print(f"convert FAILED: {type(e).__name__}: {e}")
        return None
    print(f"convert ok in {time.perf_counter()-t0:.1f}s")
    xml = outdir / f"{tag}.xml"
    ov.save_model(ov_model, str(xml))
    print(f"saved {xml} ({xml.stat().st_size/1e6:.2f} MB xml)")

    # pytorch reference
    with torch.no_grad():
        if isinstance(example, torch.Tensor):
            pt_out = pt_model(example).numpy()
        else:
            pt_out = pt_model(*example).numpy() if isinstance(example, tuple) else pt_model(example).numpy()

    results = {}
    core = ov.Core()
    for dev in devices:
        print(f"--- bench {dev} ---")
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(str(xml), dev)
            compile_s = time.perf_counter() - t0
            req = compiled.create_infer_request()
            feed = example.numpy() if isinstance(example, torch.Tensor) else example
            # warmup
            req.infer({0: feed})
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                req.infer({0: feed})
                times.append(time.perf_counter() - t0)
            out = req.get_output_tensor(0).data.copy()
            med = float(np.median(times))
            a = pt_out.reshape(-1)
            b = out.reshape(-1)
            n = min(len(a), len(b))
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
            print(
                f"{dev}: compile={compile_s:.2f}s infer={med:.3f}s "
                f"({CHUNK_SECS/med:.1f}x chunk) corr={corr:.6f}"
            )
            results[dev] = {"infer_s": med, "corr": corr, "compile_s": compile_s, "out": out}
        except Exception as e:
            print(f"{dev} FAILED: {e}")
    results["pytorch"] = {"out": pt_out}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="dialog", choices=["dialog", "effect", "music"])
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "tiger-dnr",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "tiger_ov",
    )
    ap.add_argument(
        "--audio",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "original_44k1_mono.wav",
    )
    ap.add_argument("--devices", default="CPU,GPU")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    print("loading weights…")
    dnr = load_tiger_dnr(args.cache_dir)
    stem = getattr(dnr, args.stem)

    wav, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    assert sr == SR
    mono = wav.mean(1)
    if len(mono) < CHUNK_SAMPLES:
        mono = np.pad(mono, (0, CHUNK_SAMPLES - len(mono)))
    chunk = mono[:CHUNK_SAMPLES].astype(np.float32)

    # --- Path 1: full wav model ---
    wav_model = TigerStemWav(stem).eval()
    example_wav = torch.from_numpy(chunk[None, None, :])
    print("pytorch wav forward once…")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = wav_model(example_wav)
    print(f"pytorch wav chunk={time.perf_counter()-t0:.2f}s")

    r1 = convert_and_bench(
        wav_model, example_wav, args.outdir, f"tiger_{args.stem}_wav", devices
    )

    # --- Path 2: STFT-stripped ---
    spec_model = TigerStemSpec(stem).eval()
    spec_ri = stft_ri(chunk)
    print(f"spec_ri shape={spec_ri.shape} (expect T={N_FRAMES})")
    example_spec = torch.from_numpy(spec_ri)
    print("pytorch spec forward once…")
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_spec_out = spec_model(example_spec)
    print(f"pytorch spec chunk={time.perf_counter()-t0:.2f}s out={tuple(pt_spec_out.shape)}")

    r2 = convert_and_bench(
        spec_model, example_spec, args.outdir, f"tiger_{args.stem}_spec", devices
    )

    # Extrapolate
    print("\n=== vs PyTorch full-file bakeoff baseline (~0.08× / ~12.5s per chunk-stem) ===")
    # From 60s bakeoff: 775s for 60s → per-stem share ≈ 775/3 ≈ 258s for ~15 chunks → ~17s/chunk
    # From 12s smoke: 184s / 3 stems ≈ 61s per stem for ~few chunks. Use measured pytorch above.
    hops = int(np.ceil(300.4 / 4.0)) + 2
    print(f"approx chunks/stem for 300s @ hop=4s: {hops}")
    for label, res in [("wav", r1), ("spec", r2)]:
        if not res:
            continue
        for dev in devices:
            if dev not in res or "infer_s" not in res[dev]:
                continue
            t = res[dev]["infer_s"]
            # 3 stems
            est = t * hops * 3
            print(
                f"{label}/{dev}: chunk={t:.3f}s  est_300s={est:.1f}s  "
                f"est_rtf={est/300.4:.4f} ({300.4/est:.1f}x)  "
                f"speedup_vs_0.08x_baseline={(300.4*0.08)/est if est else 0:.1f}× wall"
            )

    summary = args.outdir / "SPIKE_SUMMARY.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"stem={args.stem}\n")
        for label, res in [("wav", r1), ("spec", r2)]:
            f.write(f"[{label}]\n")
            if not res:
                f.write("  FAILED\n")
                continue
            for k, v in res.items():
                if k == "pytorch":
                    continue
                if "infer_s" in v:
                    f.write(
                        f"  {k}: infer={v['infer_s']:.4f}s corr={v.get('corr',0):.6f}\n"
                    )
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Spike: export one TIGER-DnR stem (fixed 12 s chunk) → ONNX → OpenVINO IR.

Measures Arc GPU / CPU RTF vs the PyTorch CPU bakeoff baseline (~0.08×).
Target: ≥100× wall-time improvement on the same chunk workload.
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
CHUNK_SAMPLES = int(SR * CHUNK_SECS)  # 529200 — matches TIGERDNR.wav_chunk_inference


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


class TigerStemExport(nn.Module):
    """Fixed-shape wrapper around one TIGER stem for ONNX export.

    Input:  wav [1, 1, CHUNK_SAMPLES]
    Output: stems [1, 3, CHUNK_SAMPLES]  (source order from that stem net)
    """

    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem
        # Materialize hann window so export doesn't recreate it each call.
        win = stem.win
        self.register_buffer(
            "window", torch.hann_window(win, periodic=True), persistent=False
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Inlined STFT path with fixed window buffer (export-friendly).
        stem = self.stem
        batch_size, nch, nsample = wav.shape
        x = wav.reshape(batch_size * nch, -1)

        spec = torch.stft(
            x,
            n_fft=stem.win,
            hop_length=stem.stride,
            window=self.window.to(dtype=x.dtype),
            return_complex=True,
            center=True,
            onesided=True,
        )

        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for i in range(len(stem.band_width)):
            bw = stem.band_width[i]
            subband_spec_RI.append(spec_RI[:, :, band_idx : band_idx + bw].contiguous())
            subband_spec.append(spec[:, band_idx : band_idx + bw])
            band_idx += bw

        subband_feature = []
        for i in range(len(stem.band_width)):
            subband_feature.append(
                stem.BN[i](
                    subband_spec_RI[i].reshape(
                        batch_size * nch, stem.band_width[i] * 2, -1
                    )
                )
            )
        subband_feature = torch.stack(subband_feature, 1)

        sep_output = stem.separator(
            subband_feature.reshape(
                batch_size * nch, stem.nband, stem.feature_dim, -1
            )
        )
        sep_output = sep_output.reshape(
            batch_size * nch, stem.nband, stem.feature_dim, -1
        )

        sep_subband_spec = []
        for i in range(stem.nband):
            this_output = stem.mask[i](sep_output[:, i]).reshape(
                batch_size * nch, 2, 2, stem.num_output, stem.band_width[i], -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            this_mask_real = this_mask[:, 0]
            this_mask_imag = this_mask[:, 1]
            this_mask_real_sum = this_mask_real.sum(1).unsqueeze(1)
            this_mask_imag_sum = this_mask_imag.sum(1).unsqueeze(1)
            this_mask_real = this_mask_real - (this_mask_real_sum - 1) / stem.num_output
            this_mask_imag = this_mask_imag - this_mask_imag_sum / stem.num_output
            est_spec_real = (
                subband_spec[i].real.unsqueeze(1) * this_mask_real
                - subband_spec[i].imag.unsqueeze(1) * this_mask_imag
            )
            est_spec_imag = (
                subband_spec[i].real.unsqueeze(1) * this_mask_imag
                + subband_spec[i].imag.unsqueeze(1) * this_mask_real
            )
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        sep_subband_spec = torch.cat(sep_subband_spec, 2)

        output = torch.istft(
            sep_subband_spec.reshape(
                batch_size * nch * stem.num_output, stem.enc_dim, -1
            ),
            n_fft=stem.win,
            hop_length=stem.stride,
            window=self.window.to(dtype=x.dtype),
            length=nsample,
            center=True,
            onesided=True,
        )
        return output.reshape(batch_size, stem.num_output, -1)


STEM_ATTR = {
    "dialog": "dialog",
    "effect": "effect",
    "music": "music",
}


def export_onnx(wrapper: nn.Module, onnx_path: Path) -> None:
    dummy = torch.zeros(1, 1, CHUNK_SAMPLES, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting ONNX (legacy, opset 18) → {onnx_path} …")
    # Legacy exporter: dynamo path hung / failed version-convert on this graph.
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["wav"],
            output_names=["stems"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"onnx ok  {size_mb:.2f} MB")


def convert_ir(onnx_path: Path, ir_dir: Path) -> Path:
    ir_dir.mkdir(parents=True, exist_ok=True)
    print(f"converting OpenVINO IR → {ir_dir} …")
    model = ov.convert_model(str(onnx_path))
    xml = ir_dir / (onnx_path.stem + ".xml")
    ov.save_model(model, str(xml))
    print(f"ir ok  {xml}")
    return xml


def bench_ov(xml: Path, device: str, wav: np.ndarray, runs: int = 5) -> dict:
    core = ov.Core()
    print(f"compile {device} …")
    t0 = time.perf_counter()
    compiled = core.compile_model(str(xml), device)
    compile_s = time.perf_counter() - t0
    req = compiled.create_infer_request()
    inp = wav.astype(np.float32)[None, None, :]  # [1,1,T]

    # warmup
    req.infer({0: inp})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        req.infer({0: inp})
        times.append(time.perf_counter() - t0)
    out = req.get_output_tensor(0).data.copy()
    med = float(np.median(times))
    audio_s = CHUNK_SAMPLES / SR
    return {
        "device": device,
        "compile_s": compile_s,
        "infer_s_median": med,
        "infer_s_min": float(np.min(times)),
        "rtf": med / audio_s,
        "xrt": audio_s / med,
        "out_shape": list(out.shape),
        "out": out,
    }


def bench_torch(wrapper: nn.Module, wav: np.ndarray, runs: int = 3) -> dict:
    x = torch.from_numpy(wav.astype(np.float32)[None, None, :])
    with torch.no_grad():
        _ = wrapper(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out = wrapper(x)
            times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    audio_s = CHUNK_SAMPLES / SR
    return {
        "device": "pytorch-cpu",
        "infer_s_median": med,
        "rtf": med / audio_s,
        "xrt": audio_s / med,
        "out": out.numpy(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stem",
        default="dialog",
        choices=list(STEM_ATTR),
        help="which of the three TIGER nets to export",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "tiger-dnr",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "tiger_ov",
    )
    p.add_argument(
        "--audio",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "original_44k1_mono.wav",
    )
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--devices", default="CPU,GPU", help="comma-separated OpenVINO devices")
    args = p.parse_args()

    nthreads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(nthreads)

    args.outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.outdir / f"tiger_{args.stem}_chunk12s.onnx"
    ir_dir = args.outdir / f"tiger_{args.stem}_ir"

    print("loading TIGER-DnR …")
    dnr = load_tiger_dnr(args.cache_dir)
    stem = getattr(dnr, STEM_ATTR[args.stem])
    wrapper = TigerStemExport(stem).eval()

    wav_full, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    assert sr == SR
    mono = wav_full.mean(axis=1)
    if len(mono) < CHUNK_SAMPLES:
        mono = np.pad(mono, (0, CHUNK_SAMPLES - len(mono)))
    chunk = mono[:CHUNK_SAMPLES]

    if not args.skip_export or not onnx_path.exists():
        export_onnx(wrapper, onnx_path)
        convert_ir(onnx_path, ir_dir)
    xml = ir_dir / f"tiger_{args.stem}_chunk12s.xml"
    if not xml.exists():
        # ov.save_model may use stem from onnx
        cands = list(ir_dir.glob("*.xml"))
        if not cands:
            raise SystemExit(f"no IR xml in {ir_dir}")
        xml = cands[0]

    print("\n=== PyTorch CPU reference (1 chunk, 1 stem) ===")
    pt = bench_torch(wrapper, chunk)
    print(
        f"pytorch-cpu  infer={pt['infer_s_median']:.3f}s  "
        f"rtf={pt['rtf']:.3f} ({pt['xrt']:.2f}x)"
    )

    results = {"pytorch-cpu": pt}
    for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
        print(f"\n=== OpenVINO {dev} ===")
        try:
            r = bench_ov(xml, dev, chunk)
        except Exception as e:
            print(f"FAILED on {dev}: {e}")
            continue
        results[dev] = r
        print(
            f"openvino-{dev}  compile={r['compile_s']:.2f}s  "
            f"infer={r['infer_s_median']:.3f}s (min {r['infer_s_min']:.3f}s)  "
            f"rtf={r['rtf']:.4f} ({r['xrt']:.1f}x)"
        )
        # parity
        a = pt["out"].reshape(-1)
        b = r["out"].reshape(-1)
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        mse = float(np.mean(diff**2))
        peak = float(np.max(np.abs(diff)))
        corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 1 else 0.0
        print(f"parity vs pytorch  corr={corr:.6f}  mse={mse:.3e}  maxabs={peak:.3e}")

    # Extrapolate full-file cost: 3 stems × ~ceil(T/hop) chunks with hop=4s
    audio_secs = len(mono) / SR
    hops = int(np.ceil(audio_secs / 4.0)) + 2
    print("\n=== Extrapolated full-file (3 stems × overlapping chunks) ===")
    print(f"audio={audio_secs:.1f}s  approx_chunks_per_stem={hops}")
    baseline_chunk = pt["infer_s_median"]
    for name, r in results.items():
        t = r["infer_s_median"]
        total = t * hops * 3
        speedup_vs_pt_chunk = baseline_chunk / t if t > 0 else float("inf")
        # Full PyTorch bakeoff was ~0.08× on whole file (all stems+overlap)
        print(
            f"{name:16}  chunk={t:.3f}s  "
            f"est_full={total:.1f}s  est_rtf={total/audio_secs:.3f}  "
            f"chunk_speedup_vs_pt={speedup_vs_pt_chunk:.1f}×"
        )

    # Persist a short summary
    summary = args.outdir / "SPIKE_SUMMARY.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"stem={args.stem} chunk={CHUNK_SAMPLES} samples ({CHUNK_SECS}s)\n")
        for name, r in results.items():
            f.write(
                f"{name}: infer_s={r['infer_s_median']:.4f} rtf={r['rtf']:.4f} "
                f"xrt={r['xrt']:.2f}\n"
            )
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()

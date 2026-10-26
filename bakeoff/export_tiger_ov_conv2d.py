#!/usr/bin/env python3
"""Re-export TIGER stem with Conv1d→Conv2d rewrite for Intel GPU OpenVINO."""

from __future__ import annotations

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
from openvino import properties

ROOT = Path(__file__).resolve().parent / "TIGER"
sys.path.insert(0, str(ROOT))

SR = 44100
CHUNK_SAMPLES = int(SR * 12.0)
WIN, HOP = 2048, 512
OUT = Path(__file__).resolve().parent / "out" / "tiger_ov"


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


class Conv1dAs2d(nn.Module):
    """Depthwise/grouped Conv1d expressed as Conv2d (H=1) for OpenVINO GPU."""

    def __init__(self, conv1d: nn.Conv1d):
        super().__init__()
        k = conv1d.kernel_size[0]
        self.conv = nn.Conv2d(
            conv1d.in_channels,
            conv1d.out_channels,
            kernel_size=(1, k),
            stride=(1, conv1d.stride[0]),
            padding=(0, conv1d.padding[0] if isinstance(conv1d.padding, tuple) else conv1d.padding),
            dilation=(1, conv1d.dilation[0]),
            groups=conv1d.groups,
            bias=conv1d.bias is not None,
        )
        with torch.no_grad():
            self.conv.weight.copy_(conv1d.weight.unsqueeze(2))
            if conv1d.bias is not None:
                self.conv.bias.copy_(conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.unsqueeze(2)).squeeze(2)


def replace_conv1d(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(module, name, Conv1dAs2d(child))
            n += 1
        else:
            n += replace_conv1d(child)
    return n


class TigerStemSpec(nn.Module):
    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem

    def forward(self, spec_ri: torch.Tensor) -> torch.Tensor:
        stem = self.stem
        B = spec_ri.shape[0]
        spec_real = spec_ri[:, 0]
        spec_imag = spec_ri[:, 1]
        subband_feature = []
        band_idx = 0
        sub_real, sub_imag = [], []
        for i, bw in enumerate(stem.band_width):
            sr = spec_real[:, band_idx : band_idx + bw, :]
            si = spec_imag[:, band_idx : band_idx + bw, :]
            sub_real.append(sr)
            sub_imag.append(si)
            ri = torch.stack([sr, si], 1)
            subband_feature.append(stem.BN[i](ri.reshape(B, bw * 2, -1)))
            band_idx += bw
        subband_feature = torch.stack(subband_feature, 1)
        sep_output = stem.separator(subband_feature)
        sep_output = sep_output.reshape(B, stem.nband, stem.feature_dim, -1)
        out_real, out_imag = [], []
        for i, bw in enumerate(stem.band_width):
            this_output = stem.mask[i](sep_output[:, i]).reshape(
                B, 2, 2, stem.num_output, bw, -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            mr = this_mask[:, 0]
            mi = this_mask[:, 1]
            mr = mr - (mr.sum(1).unsqueeze(1) - 1) / stem.num_output
            mi = mi - mi.sum(1).unsqueeze(1) / stem.num_output
            er = sub_real[i].unsqueeze(1) * mr - sub_imag[i].unsqueeze(1) * mi
            ei = sub_real[i].unsqueeze(1) * mi + sub_imag[i].unsqueeze(1) * mr
            out_real.append(er)
            out_imag.append(ei)
        return torch.stack([torch.cat(out_real, 2), torch.cat(out_imag, 2)], 1)


def stft_ri(wav: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(wav.astype(np.float32))
    spec = torch.stft(
        x,
        n_fft=WIN,
        hop_length=HOP,
        window=torch.hann_window(WIN, periodic=True),
        return_complex=True,
        center=True,
        onesided=True,
    )
    return torch.stack([spec.real, spec.imag], 0).unsqueeze(0).numpy()


def main() -> None:
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    cache = Path(__file__).resolve().parent / "models" / "tiger-dnr"
    OUT.mkdir(parents=True, exist_ok=True)

    print("load…")
    dnr = load_tiger_dnr(cache)
    stem = dnr.dialog
    n = replace_conv1d(stem)
    print(f"replaced {n} Conv1d → Conv2d(H=1)")

    wav, sr = sf.read(
        Path(__file__).resolve().parent / "out" / "original_44k1_mono.wav",
        dtype="float32",
        always_2d=True,
    )
    assert sr == SR
    mono = wav.mean(1)
    chunk = mono[:CHUNK_SAMPLES].astype(np.float32)
    spec = stft_ri(chunk)
    print("spec", spec.shape)

    model = TigerStemSpec(stem).eval()
    example = torch.from_numpy(spec)
    with torch.no_grad():
        t0 = time.perf_counter()
        pt_out = model(example)
        print(f"pytorch {time.perf_counter()-t0:.2f}s out={tuple(pt_out.shape)}")

    print("ov.convert_model…")
    t0 = time.perf_counter()
    ov_model = ov.convert_model(model, example_input=example)
    print(f"convert {time.perf_counter()-t0:.1f}s")
    xml = OUT / "tiger_dialog_spec_conv2d.xml"
    ov.save_model(ov_model, str(xml))

    # Fix dynamic → static
    core = ov.Core()
    ov_model = core.read_model(str(xml))
    ov_model.reshape({ov_model.inputs[0].any_name: list(spec.shape)})
    dummy = spec

    bakeoff_wall = 300.4 / 0.08  # ~3755s pytorch full-file estimate
    hops = 78

    for label, device, cfg in [
        ("CPU", "CPU", {}),
        ("GPU", "GPU", {}),
        ("GPU_f16", "GPU", {properties.hint.inference_precision: ov.Type.f16}),
    ]:
        print(f"\n=== {label} ===")
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(ov_model, device, cfg)
            print(f"compile {time.perf_counter()-t0:.2f}s")
            req = compiled.create_infer_request()
            req.infer({0: dummy})
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                req.infer({0: dummy})
                times.append(time.perf_counter() - t0)
            med = float(np.median(times))
            out = req.get_output_tensor(0).data
            corr = float(
                np.corrcoef(pt_out.detach().numpy().ravel(), out.ravel())[0, 1]
            )
            est = med * hops * 3
            speedup = bakeoff_wall / est
            print(
                f"infer={med:.3f}s corr={corr:.6f}  "
                f"est_300s={est:.1f}s ({300.4/est:.1f}x RT)  "
                f"speedup_vs_pt_bakeoff={speedup:.1f}×"
            )
            hit_100 = "YES" if speedup >= 100 else "NO"
            print(f"  ≥100× target? {hit_100}")
        except Exception as e:
            print("FAIL:", e)


if __name__ == "__main__":
    main()

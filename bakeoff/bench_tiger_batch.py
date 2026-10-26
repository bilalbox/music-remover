#!/usr/bin/env python3
"""Batch-throughput bench for Conv2d-rewritten TIGER dialog stem on Arc GPU."""

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

# Reuse helpers from export_tiger_ov_conv2d
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_tiger_ov_conv2d import (  # noqa: E402
    CHUNK_SAMPLES,
    OUT,
    SR,
    TigerStemSpec,
    load_tiger_dnr,
    replace_conv1d,
    stft_ri,
)

CACHE = Path(__file__).resolve().parent / "models" / "tiger-dnr"


def main() -> None:
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    dnr = load_tiger_dnr(CACHE)
    stem = dnr.dialog
    replace_conv1d(stem)
    model = TigerStemSpec(stem).eval()

    wav, sr = sf.read(
        Path(__file__).resolve().parent / "out" / "original_44k1_mono.wav",
        dtype="float32",
        always_2d=True,
    )
    mono = wav.mean(1)
    chunk = mono[:CHUNK_SAMPLES].astype(np.float32)
    spec1 = stft_ri(chunk)  # 1,2,F,T

    core = ov.Core()
    bakeoff_wall = 300.4 / 0.08
    hops = 78  # chunks/stem at hop=4s

    for B in [1, 2, 4, 8]:
        print(f"\n===== batch={B} =====")
        example = torch.from_numpy(np.repeat(spec1, B, axis=0))
        t0 = time.perf_counter()
        ov_model = ov.convert_model(model, example_input=example)
        print(f"convert {time.perf_counter()-t0:.1f}s")
        ov_model.reshape({ov_model.inputs[0].any_name: list(example.shape)})
        feed = example.numpy()

        for label, dev, cfg in [
            ("GPU", "GPU", {}),
            ("GPU_f16", "GPU", {properties.hint.inference_precision: ov.Type.f16}),
            (
                "GPU_thru_f16",
                "GPU",
                {
                    properties.hint.performance_mode: properties.hint.PerformanceMode.THROUGHPUT,
                    properties.hint.inference_precision: ov.Type.f16,
                },
            ),
        ]:
            try:
                t0 = time.perf_counter()
                compiled = core.compile_model(ov_model, dev, cfg)
                print(f"  {label} compile {time.perf_counter()-t0:.1f}s", end="")
                req = compiled.create_infer_request()
                req.infer({0: feed})
                times = []
                for _ in range(4):
                    t0 = time.perf_counter()
                    req.infer({0: feed})
                    times.append(time.perf_counter() - t0)
                med = float(np.median(times))
                per_chunk = med / B
                # number of batched calls per stem ≈ ceil(hops/B)
                calls = int(np.ceil(hops / B))
                est = per_chunk * B * calls * 3  # == med * calls * 3
                # clearer: est = med * ceil(hops/B) * 3
                est = med * calls * 3
                speedup = bakeoff_wall / est
                print(
                    f"  batch_infer={med:.3f}s  per_chunk={per_chunk:.3f}s  "
                    f"est_300s={est:.1f}s ({300.4/est:.1f}x)  speedup={speedup:.1f}×"
                    f"  {'✓100×' if speedup>=100 else ''}"
                )
            except Exception as e:
                print(f"  {label} FAIL: {e}")


if __name__ == "__main__":
    main()

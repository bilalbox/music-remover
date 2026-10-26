#!/usr/bin/env python3
"""Retry Arc GPU compile/bench for the STFT-stripped TIGER IR."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import properties

xml = Path("bakeoff/out/tiger_ov/tiger_dialog_spec.xml")
core = ov.Core()
print("ov", ov.__version__, "devices", core.available_devices)

model = core.read_model(str(xml))
inp = model.inputs[0]
print("input", inp.any_name, inp.partial_shape, inp.element_type)

# Fixed shape from export: [1, 2, 1025, 1034]
fixed = [1, 2, 1025, 1034]
model.reshape({inp.any_name: fixed})
dummy = np.zeros(fixed, dtype=np.float32)
print("reshaped to", fixed)

configs = [
    ("GPU_default", {}),
    ("GPU_f16", {properties.hint.inference_precision: ov.Type.f16}),
    (
        "GPU_latency_f16",
        {
            properties.hint.performance_mode: properties.hint.PerformanceMode.LATENCY,
            properties.hint.inference_precision: ov.Type.f16,
        },
    ),
    (
        "GPU_throughput_f16",
        {
            properties.hint.performance_mode: properties.hint.PerformanceMode.THROUGHPUT,
            properties.hint.inference_precision: ov.Type.f16,
        },
    ),
]

for name, cfg in configs:
    print(f"\n=== {name} ===")
    try:
        t0 = time.perf_counter()
        compiled = core.compile_model(model, "GPU", cfg)
        print(f"compile {time.perf_counter() - t0:.2f}s")
        req = compiled.create_infer_request()
        req.infer({0: dummy})
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            req.infer({0: dummy})
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        hops = 78
        est = med * hops * 3
        print(
            f"infer median={med:.3f}s  chunk={12.0/med:.1f}x  "
            f"est_300s={est:.1f}s ({300.4/est:.1f}x)  "
            f"speedup_vs_pt_0.08x={(300.4/0.08)/est:.1f}× wall vs bakeoff"
        )
        # speedup vs bakeoff wall: bakeoff was 300/0.08 = 3750s for full file
        # Wait: speedup = bakeoff_wall / new_wall = (300.4/0.08) / est
        print(f"  vs bakeoff wall ~{300.4/0.08:.0f}s → speedup {(300.4/0.08)/est:.1f}×")
    except Exception as e:
        print("FAIL:", e)

# CPU reference with new OV version
print("\n=== CPU ===")
t0 = time.perf_counter()
compiled = core.compile_model(model, "CPU")
print(f"compile {time.perf_counter() - t0:.2f}s")
req = compiled.create_infer_request()
req.infer({0: dummy})
times = []
for _ in range(5):
    t0 = time.perf_counter()
    req.infer({0: dummy})
    times.append(time.perf_counter() - t0)
med = float(np.median(times))
print(f"infer median={med:.3f}s chunk={12.0/med:.1f}x")

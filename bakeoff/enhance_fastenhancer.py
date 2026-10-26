#!/usr/bin/env python3
"""Offline FastEnhancer streaming inference for the music-remover bakeoff.

Mirrors aask1357/fastenhancer scripts/test_onnx.py, but:
  - reads/writes WAV via soundfile (stereo -> per-channel mono streams)
  - reports wall time + RTF
  - writes float32 WAV

48 kHz models: --n-fft 1024 --hop-size 512 --sr 48000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


def enhance_mono(
    sess: ort.InferenceSession,
    wav: np.ndarray,
    *,
    n_fft: int,
    hop_size: int,
) -> tuple[np.ndarray, float]:
    """Stream one mono channel. Returns (enhanced, infer_seconds)."""
    wav = np.clip(wav.astype(np.float32).reshape(1, -1), -1.0, 1.0)
    length = wav.shape[-1]

    input_size = next(x for x in sess.get_inputs() if x.name == "wav_in").shape[-1]
    output_size = next(x for x in sess.get_outputs() if x.name == "wav_out").shape[-1]
    if output_size != hop_size:
        raise SystemExit(
            f"Model outputs {output_size} samples/hop, expected hop_size={hop_size}"
        )
    if input_size not in (hop_size, n_fft):
        raise SystemExit(
            f"Model expects {input_size} input samples; expected {hop_size} or {n_fft}"
        )

    left_pad = n_fft - hop_size if input_size == n_fft else 0
    wav = np.pad(wav, ((0, 0), (left_pad, n_fft)), mode="constant")

    onnx_input = {
        x.name: np.zeros(x.shape, dtype=np.float32)
        for x in sess.get_inputs()
        if x.name.startswith("cache_in_")
    }

    chunks: list[np.ndarray] = []
    tic = time.perf_counter()
    last_idx = 0
    for idx in range(0, length + n_fft - hop_size, hop_size):
        onnx_input["wav_in"] = wav[:, idx : idx + input_size]
        out = sess.run(None, onnx_input)
        chunks.append(out[0][0])
        for j, cache in enumerate(out[1:]):
            onnx_input[f"cache_in_{j}"] = cache
        last_idx = idx
    toc = time.perf_counter()

    wav_out = np.concatenate(chunks, axis=0)
    start_idx = n_fft - hop_size
    wav_out = np.clip(wav_out[start_idx : start_idx + length], -1.0, 1.0)
    _ = last_idx  # silence unused if length==0
    return wav_out.astype(np.float32), toc - tic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--sr", type=int, default=48_000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-size", type=int, default=512)
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ORT intra/inter threads (0 = ORT default / all cores)",
    )
    p.add_argument(
        "--mono-mix",
        action="store_true",
        help="Downmix to mono, enhance once, duplicate to stereo (faster)",
    )
    args = p.parse_args()

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != args.sr:
        raise SystemExit(f"Input SR {sr} != expected {args.sr}; resample first")

    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.threads > 0:
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1

    load_tic = time.perf_counter()
    sess = ort.InferenceSession(
        str(args.onnx), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    load_secs = time.perf_counter() - load_tic

    if args.mono_mix or audio.shape[1] == 1:
        mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        enhanced, infer_secs = enhance_mono(
            sess, mono, n_fft=args.n_fft, hop_size=args.hop_size
        )
        out = np.stack([enhanced, enhanced], axis=1)
        channels_run = 1
    else:
        channels = []
        infer_secs = 0.0
        for ch in range(audio.shape[1]):
            enh, t = enhance_mono(
                sess, audio[:, ch], n_fft=args.n_fft, hop_size=args.hop_size
            )
            channels.append(enh)
            infer_secs += t
        out = np.stack(channels, axis=1)
        channels_run = audio.shape[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, out, args.sr, subtype="FLOAT")

    audio_secs = len(audio) / sr
    rtf = infer_secs / audio_secs if audio_secs > 0 else float("inf")
    print(
        f"model={args.onnx.name} load={load_secs:.2f}s infer={infer_secs:.2f}s "
        f"audio={audio_secs:.1f}s rtf={rtf:.3f} ({1/rtf:.1f}x) "
        f"channels_run={channels_run} out={args.output}"
    )


if __name__ == "__main__":
    main()

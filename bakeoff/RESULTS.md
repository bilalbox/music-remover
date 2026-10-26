# Bakeoff: HTDemucs-DnR vs FastEnhancer vs TIGER-DnR

**Status (2026-10-26): paused — keep Demucs.**  
TIGER wins on quality; OpenVINO on Arc is only ~9× vs TIGER’s PyTorch CPU and still much slower than Demucs Arc. Revisit when models evolve or a production-ready OpenVINO path appears.

Clip: `maher-bakeoff.mkv` — **300.4 s**, stereo Opus @ 48 kHz.  
Machine: Windows + Intel Arc (OpenVINO GPU) / CPU ORT (FastEnhancer) / PyTorch CPU + OV (TIGER).  
Dates: 2026-10-25 (FE) / 2026-10-26 (TIGER + OV spike).

---

## Decision

| Option | Verdict |
|--------|---------|
| **HTDemucs-DnR (current)** | **Keep as default** — best Arc wall time (29×), good SFX keep |
| **TIGER-DnR** | **Preferred quality** on listen; too slow for product until OV/runtime improves (~0.7× est. on Arc) |
| **FastEnhancer (NoMusic)** | **Not a Demucs replacement** — speech enhancement, weak SFX; possible future low-resource / speech-first mode only |

---

## Compared stacks

| Variant | Task | Size (ballpark) |
|---------|------|-----------------|
| Demucs soundtrack / speech | DnR 3-stem @ 44.1 kHz | ~75 MB IR + OpenVINO |
| FastEnhancer T / S (± mono-mix) | Streaming speech enhance @ 48 kHz | 148 KB / 855 KB ONNX |
| TIGER-DnR | DnR 3-stem @ 44.1 kHz | 4.22M params (~few MB weights) |

RNNoise after FE was **not** run. TIGER full 300 s PyTorch run was **aborted**; timing from 12 s / 60 s + OV chunk benches.

---

## Speed summary

| Backend | Wall / note | RTF |
|---------|-------------|-----|
| Demucs soundtrack **Arc GPU** | **10.3 s** / 300 s | **29×** |
| Demucs soundtrack **CPU** | 171 s | 1.8× |
| FE-T mono-mix CPU | 12.7 s | 23.6× |
| FE-S mono-mix CPU | 25.7 s | 11.7× |
| TIGER PyTorch CPU | 775 s / **60 s** audio | **0.08×** |
| TIGER OV Arc FP16 (est. full*) | ~415 s / 300 s | **~0.7×** |

\*Estimate: 1.77 s/chunk × ~78 hops × 3 stems; STFT-stripped + Conv1d→Conv2d IR.

---

## FastEnhancer notes

- On Arc offline, Demucs is already as fast or faster than FE CPU.
- On CPU-only, FE-T mono is ~13× faster than Demucs CPU and ~500× smaller.
- Objectively tracks **speech-only** more than soundtrack (high-band cut, SFX risk).
- Listen: `bakeoff/out/listen_*` (t=23 / t=30 / t=279).

---

## TIGER notes

Same product shape as Demucs (dialog / effect / music). Listen preferred TIGER over Demucs on this clip.

| | music | sfx | speech |
|--|-------|-----|--------|
| TIGER 60 s RMS | −37.2 dB | −44.0 dB | −23.7 dB |
| Demucs full-file RMS | −41.7 dB | −47.6 dB | −28.4 dB |

60 s objective: TIGER stays closer to the mix (Δrms −0.2 vs Demucs −3.3). Listen clips: `listen_tiger_*_t23s.wav` / `_t30s`.

---

## OpenVINO spike (TIGER)

Goal was ≥**100×** vs PyTorch bakeoff (~0.08×). **Not met** (~**9×** on Arc FP16).

| Step | Result |
|------|--------|
| ONNX wav + `torch.stft` | Fail — complex STFT unsupported |
| `ov.convert_model` full wav | corr ≈ 0 |
| STFT-stripped RI in/out | **corr = 1.0** |
| GPU without rewrite | GroupConvolution rank mismatch on depthwise Conv1d |
| 188× Conv1d → Conv2d(H=1) | **GPU works**, corr = 1.0 |
| Batched convert | Crashed (access violation) |

IR / scripts (local, gitignored outputs):  
`bakeoff/out/tiger_ov/tiger_dialog_spec_conv2d.xml`, `export_tiger_ov_*.py`, `run_tiger_dnr.py`, `enhance_fastenhancer.py`.

**Revisit when:** upstream ships a clean OV/ORT export, Arc path gets another ~10×, or a lighter DnR-quality model appears.

---

## Re-run (if needed)

```powershell
# Demucs
.\music-remover.ps1 .\maher-bakeoff.mkv --mode audio --audio-format wav --keep soundtrack -o bakeoff\out\demucs_soundtrack.wav --device GPU

# FastEnhancer (bakeoff\.venv)
bakeoff\.venv\Scripts\python.exe bakeoff\enhance_fastenhancer.py `
  --onnx bakeoff\models\fastenhancer_t.onnx `
  --input bakeoff\out\original_48k.wav `
  --output bakeoff\out\fe_t_mono.wav --mono-mix

# TIGER PyTorch CPU (bakeoff\.venv-tiger)
bakeoff\.venv-tiger\Scripts\python.exe bakeoff\run_tiger_dnr.py `
  --input bakeoff\out\original_44k1_mono.wav `
  --outdir bakeoff\out\tiger_full --device cpu

# TIGER OpenVINO Arc (after setupvars)
bakeoff\.venv-tiger\Scripts\python.exe bakeoff\export_tiger_ov_conv2d.py
```

Large outputs / venvs / cloned TIGER tree are gitignored under `bakeoff/` — this file and the harness scripts are what to keep in git.

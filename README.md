# music-remover

Strip background music from a media file or a URL, keeping speech and sound effects.
Runs ZFTurbo's **HTDemucs trained on Divide and Remaster** locally via OpenVINO — no
cloud, and no Python at runtime. On this machine a 25 s clip separates at about 20× realtime
on the Arc GPU (warm start ~2 s) and about 7.5× on CPU. `--device CPU` uses the
OpenVINO CPU plugin, which is not Intel-specific.

Give it a local file or a YouTube URL; it outputs the cleaned audio, or (for video
sources) the original video with the music-removed audio swapped in.

The default keep is the soundtrack: speech plus sound effects, score dropped. Some
beds still leave a whisper of music inside speech or effects. When the music stem
itself is nearly silent, the tool says so — a second pass will not remove more.

## Usage

```powershell
# YouTube URL -> <title>.nomusic.mp4 (video, speech + effects)
.\music-remover.ps1 "https://www.youtube.com/watch?v=<id>"

# local video -> clip.nomusic.mp4
.\music-remover.ps1 .\clip.mp4

# dialogue only, or the score only
.\music-remover.ps1 .\clip.mp4 --keep speech
.\music-remover.ps1 .\clip.mp4 --keep music

# audio-only output, as mp3
.\music-remover.ps1 .\lecture.m4a --mode audio --audio-format mp3

# smaller remux: lower bitrate, or keep WebM + Opus (best with yt-dlp WebM sources)
.\music-remover.ps1 .\clip.mp4 --audio-bitrate 128k
.\music-remover.ps1 .\clip.webm --container webm

# better quality (slower): more shifts
.\music-remover.ps1 .\clip.mp4 --shifts 5
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `auto` | `auto` (video if source has video, else audio), `video`, `audio` |
| `--keep` | `soundtrack` | `soundtrack` (speech + effects), `speech`, or `music` |
| `--audio-format` | `wav` | output audio format when not producing video (`wav`, `mp3`, `m4a`, `flac`, `opus`) |
| `--container` | `mp4` | video mux: `mp4` or `webm` (video stream is copied; source codec must fit the container) |
| `--audio-codec` | from container | remux codec: `aac` (mp4 default) or `opus` (webm default) |
| `--audio-bitrate` | `192k` | lossy audio bitrate for remux and for mp3/m4a/opus/ogg/webm encode |
| `--device` | `GPU` | `GPU` (Arc), `CPU`, or `NPU` |
| `--shifts` | `1` | Demucs shifts; higher = slightly better, proportionally slower |
| `-o, --output` | derived | explicit output path |
| `--model` | bundled | path to the `htdemucs_dnr` model dir |
| `--keep-temp` | off | keep the downloaded/intermediate files (`music.wav`, `sfx.wav`, `speech.wav`) |

Cleaned audio is fed to ffmpeg as float PCM (no 16-bit WAV round-trip) for remux and lossy encode. `--keep-temp` stem dumps and `--audio-format wav` still write 16-bit WAV.

## Requirements

- **ffmpeg** and **yt-dlp** on `PATH`.
- **OpenVINO 2025.x runtime** (for the Arc GPU plugin). The `music-remover.ps1` wrapper
  sources its `setupvars.ps1` automatically; point `$env:OPENVINO_DIST` at your install
  if it isn't the default `~/openvino-dist/...` path.
- An Intel GPU driver (provides the compute runtime the GPU plugin needs).

## Build

```powershell
# from a shell that has sourced OpenVINO's setupvars.ps1:
cargo build --release
```

The OpenVINO IR lives in `models/htdemucs_dnr/` (`htdemucs_fwd.xml` + `.bin`). It is
produced once from checkpoint `97d170e1-dbb4db15.th`
([ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing v.1.0.0](https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/tag/v.1.0.0))
by `scripts/export_htdemucs_dnr.py`. That script needs Python, torch, `onnx`, `demucs-infer`,
and `openvino==2025.4`. The runtime does not.

## Linux

No Windows/PowerShell involved — this is a plain `cargo` build with the system OpenVINO
package.

**Note:** OpenVINO's `GPU` device plugin only accelerates **Intel** GPUs. On any other GPU
(AMD, Nvidia), use `--device CPU`. OpenVINO's CPU plugin is not Intel-specific and runs fine
on any x86_64 CPU.

```sh
# ffmpeg + yt-dlp from your distro's repos, e.g. on Arch:
sudo pacman -S ffmpeg yt-dlp openvino

# download the checkpoint and export models/htdemucs_dnr/ (Python, one time)
./fetch-model.sh

cargo build --release

./target/release/music-remover clip.mp4 --device CPU
```

There's no run wrapper — pass `--device CPU` explicitly each time (the compiled-in default
is `GPU`, for the Windows/Arc-GPU build). CPU speed scales with the file and the core
count; the 7.5× figure above is one 25 s clip on this machine.

## Standalone / portable build

```powershell
.\package.ps1
```

This produces `dist/music-remover/` containing the exe, the OpenVINO runtime DLLs (incl.
the Arc GPU plugin), and the model. **Copy that whole folder anywhere** and run
`music-remover.exe` directly — no `setupvars`, no PATH changes. The exe loads its DLLs and
finds the model relative to its own location. (It still needs `ffmpeg` + `yt-dlp` on PATH
and the Intel GPU driver; the Visual C++ runtime is already present on Windows 11.)

### Single self-extracting exe

```powershell
.\package.ps1 -SingleFile
```

Also produces `dist-single\music-remover.exe` — a single file with the entire runtime +
model embedded. Share/copy just that one file; on first run it unpacks itself to
`%LOCALAPPDATA%\music-remover\rt-<n>\` and runs (later runs skip extraction). This exists
because OpenVINO loads its GPU plugin and core as separate DLLs at runtime — Windows can't
load a DLL straight from inside an .exe — so a true zero-extraction single binary isn't
possible; this is the practical equivalent. (`ffmpeg` + `yt-dlp` + GPU driver still required
on the target machine.)

### A note on `--cache-dir`

OpenVINO can cache the compiled model to disk, but it's **off by default** here: the Intel
GPU driver already caches compiled kernels, so startup is dominated by loading the runtime
and the model, not the JIT. Enabling `--cache-dir <dir>` writes a compiled blob and gives
no measurable speedup on a warm driver cache — it only helps guarantee a fast *first*
launch on machines where that driver cache is cold or has been cleared.

## How it works

`ffmpeg` decodes to 44.1 kHz stereo → `src/demucs.rs` runs the hybrid-transformer pipeline
(reflect-pad → STFT → magnitude → per-segment inference → de-norm → iSTFT + time branch →
triangular overlap-add with random shifts) around an STFT-stripped IR. The segment is the
DnR training length, 382788 samples (8.68 s), and the graph returns three stems in order
music, sfx, speech. `src/stft.rs` implements the torch-compatible STFT/iSTFT (unit-tested
round-trip).

The three stems sum back to the input (about 2% relative RMS on a checked clip).
`--keep soundtrack` writes speech + sfx.

## License

MIT — see [LICENSE](LICENSE).

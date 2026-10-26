"""Export ZFTurbo's HTDemucs-on-DnR checkpoint to an STFT-stripped OpenVINO IR.

The Rust host (src/demucs.rs) owns STFT, overlap-add, and mean/std normalization.
This graph is only the hybrid encoder / cross-transformer / decoder, matching the
contract of Intel's htdemucs_fwd IR:

  x   [1, 4, 2048, T]     normalized complex-as-channels spectrogram
  xt  [1, 2, SEGMENT]     normalized waveform
  x_out  [1, 12, 2048, T] pre-denorm complex spectrogram, sources packed (music, sfx, speech)
  xt_out [1, 6, SEGMENT]  pre-denorm waveform, same source order

Checkpoint: https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/tag/v.1.0.0
Default file: 97d170e1-dbb4db15.th (single model, not the 3-checkpoint ensemble).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import demucs_infer  # noqa: F401  — aliases demucs.* before unpickle
import torch
from demucs.states import load_model
from torch import nn

# Training length baked into 97d170e1: segment = 217/25 s at 44100 Hz.
SEGMENT = 382788
# ceil(SEGMENT / 1024); _spec keeps this many frames after dropping the nyquist bin.
FRAMES = 374
FREQS = 2048


class StrippedHTDemucs(nn.Module):
    """Inner HTDemucs forward, starting from already-normalized x and xt."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, xt: torch.Tensor):
        m = self.model
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, c, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, c, f, t)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        return x, xt


def reconstruct(model, mix, x_out, xt_out, std, mean, stdt, meant):
    """Denorm + iSTFT + add time branch, matching HTDemucs.forward."""
    sources = len(model.sources)
    b, _, freq, frames = x_out.shape
    spec = x_out.view(b, sources, -1, freq, frames) * std[:, None] + mean[:, None]
    zout = model._mask(None, spec)
    wave = model._ispec(zout, SEGMENT)
    time = xt_out.view(b, sources, -1, SEGMENT) * stdt[:, None] + meant[:, None]
    return time + wave


def check_against_full(model, stripped) -> None:
    g = torch.Generator().manual_seed(0)
    mix = torch.randn(1, 2, SEGMENT, generator=g)
    with torch.inference_mode():
        full = model(mix)
        z = model._spec(mix)
        mag = model._magnitude(z)
        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std = mag.std(dim=(1, 2, 3), keepdim=True)
        x = (mag - mean) / (1e-5 + std)
        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt = (mix - meant) / (1e-5 + stdt)
        x_out, xt_out = stripped(x, xt)
        recon = reconstruct(model, mix, x_out, xt_out, std, mean, stdt, meant)
    err = (recon - full).abs().max().item()
    print(f"x_out {tuple(x_out.shape)} xt_out {tuple(xt_out.shape)} max_abs_err {err:.3e}")
    if x_out.shape != (1, 12, FREQS, FRAMES) or xt_out.shape != (1, 6, SEGMENT):
        raise SystemExit("unexpected stripped shapes")
    if err > 1e-4:
        raise SystemExit("stripped forward does not match HTDemucs.forward")


def export(stripped, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    onnx_path = dest / "htdemucs_fwd.onnx"
    x = torch.zeros(1, 4, FREQS, FRAMES)
    xt = torch.zeros(1, 2, SEGMENT)
    print(f"exporting ONNX to {onnx_path}")
    torch.onnx.export(
        stripped,
        (x, xt),
        str(onnx_path),
        input_names=["x", "xt"],
        output_names=["x_out", "xt_out"],
        dynamo=False,
    )
    import openvino as ov

    print("converting to OpenVINO IR (fp16)")
    ov_model = ov.convert_model(str(onnx_path))
    xml = dest / "htdemucs_fwd.xml"
    ov.save_model(ov_model, str(xml), compress_to_fp16=True)
    onnx_path.unlink()
    print(f"wrote {xml} and {xml.with_suffix('.bin')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "htdemucs_dnr" / "97d170e1-dbb4db15.th",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "htdemucs_dnr",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    model = load_model(str(args.checkpoint))
    model.eval()
    length = int(model.segment * model.samplerate)
    if length != SEGMENT:
        raise SystemExit(f"checkpoint segment length {length} != expected {SEGMENT}")
    stripped = StrippedHTDemucs(model).eval()
    check_against_full(model, stripped)
    if not args.check_only:
        export(stripped, args.dest)


if __name__ == "__main__":
    main()

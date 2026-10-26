"""Compare one DnR segment through the OpenVINO IR against PyTorch HTDemucs.forward."""

from pathlib import Path

import demucs_infer  # noqa: F401
import numpy as np
import openvino as ov
import torch
from demucs.states import load_model

from export_htdemucs_dnr import SEGMENT, StrippedHTDemucs, reconstruct

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "models" / "htdemucs_dnr" / "htdemucs_fwd.xml"
CKPT = Path(r"C:\Users\bilal\AppData\Local\Temp\bandit-plus\htdemucs-dnr\models\97d170e1-dbb4db15.th")


def main() -> None:
    model = load_model(str(CKPT)).eval()
    stripped = StrippedHTDemucs(model).eval()
    g = torch.Generator().manual_seed(1)
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

    core = ov.Core()
    compiled = core.compile_model(str(XML), "CPU")
    print("inputs", [(i.any_name, i.shape) for i in compiled.inputs])
    print("outputs", [(o.any_name, o.shape) for o in compiled.outputs])
    res = compiled([x.numpy(), xt.numpy()])
    x_out = torch.from_numpy(np.array(res[compiled.output("x_out")]))
    xt_out = torch.from_numpy(np.array(res[compiled.output("xt_out")]))
    with torch.inference_mode():
        recon = reconstruct(model, mix, x_out, xt_out, std, mean, stdt, meant)
    err = (recon - full).abs()
    corr = np.corrcoef(recon.numpy().ravel(), full.numpy().ravel())[0, 1]
    print(f"max_abs {err.max().item():.4e}  mean_abs {err.mean().item():.4e}  corr {corr:.6f}")


if __name__ == "__main__":
    main()

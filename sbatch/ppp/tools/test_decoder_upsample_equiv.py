"""Check Autoencoder decoder with upsample_conv_first=True equals the default (fp32 + AMP).
Run from PatchPerPix/experiments inside the ppp container (see test_decoder_upsample_equiv_sbatch.sh)."""
import copy, os, sys
import toml, torch
sys.path.insert(0, os.getcwd())
from flylight.setups.setup01 import torch_model  # noqa: E402

def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict): merge(a[k], v)
        else: a[k] = v
    return a

cfg = {}
for p in sys.argv[1:]:
    merge(cfg, toml.load(p))
ae = cfg["model"]["autoencoder"]
ae["input_shape_squeezed"] = tuple(p for p in ae["patchshape"] if p != 1)
ae["code_units"] = cfg["model"]["code_units"] if "code_units" in cfg["model"] else ae.get("code_units", 1024)
print("autoencoder upsampling:", ae["upsampling"], "code_units:", ae["code_units"])

dev = torch.device("cuda")
torch.manual_seed(0)
a = torch_model.Autoencoder(copy.deepcopy({**ae, "upsample_conv_first": False})).to(dev)
b = torch_model.Autoencoder(copy.deepcopy({**ae, "upsample_conv_first": True})).to(dev)
assert a.state_dict().keys() == b.state_dict().keys(), "state_dict keys differ"
b.load_state_dict(a.state_dict())
x = torch.rand(2048, ae["code_units"], device=dev)

for amp in (False, True):
    xa = x.clone().requires_grad_(True); xb = x.clone().requires_grad_(True)
    with torch.autocast("cuda", enabled=amp):
        ya, yb = a(xa), b(xb)
    (ya.float().square().mean()).backward(); (yb.float().square().mean()).backward()
    d_out = (ya.float() - yb.float()).abs().max().item()
    d_in = (xa.grad - xb.grad).abs().max().item()
    pairs = [(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters())]
    assert all((ga is None) == (gb is None) for ga, gb in pairs), "grad presence differs"
    d_w = max((ga - gb).abs().max().item() for ga, gb in pairs if ga is not None)
    print(f"amp={amp}: max|out diff|={d_out:.3e} max|dx diff|={d_in:.3e} max|dW diff|={d_w:.3e} out_absmax={ya.abs().max().item():.3f}")
    a.zero_grad(); b.zero_grad()

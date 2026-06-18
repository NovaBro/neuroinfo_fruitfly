"""Run BiaPy with a torch.load patch for PyTorch 2.6+ compatibility.

BiaPy checkpoints store a yacs CfgNode, which cannot be unpickled with
weights_only=True. This wrapper forces weights_only=False for trusted
local checkpoints before invoking the BiaPy CLI.
"""
import torch

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from biapy import main

if __name__ == "__main__":
    main()

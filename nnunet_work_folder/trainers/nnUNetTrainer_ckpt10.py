"""Stock nnUNetTrainer that writes checkpoint_latest.pth every 10 epochs instead of every 50.

Found by nnU-Net through `nnUNet_extTrainer=<repo>/nnunet_work_folder/trainers` (nnunetv2 >= 2.8),
so no fork of nnU-Net is needed. Training is otherwise identical to the default (1000 epochs x 250 it,
SGD lr 1e-2 poly, DC+BCE for regions, deep supervision). Purpose: the 47 h SLURM wall-time kill loses
at most 10 epochs when the next chained job resumes with `--c`.
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_ckpt10(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 10

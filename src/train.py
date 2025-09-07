"""src/train.py
Contains:
1.  Utility helpers (set_global_seed, pretty_print_json)
2.  Model building blocks (HaarWavelet, ResidualMixer, FCRD1Step)
3.  Trainer class
4.  run_experiment() that orchestrates training + evaluation – called from src.main
NOTE: Only modules declared in pyproject.toml are imported.  No hidden deps.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import tqdm

from .preprocess import COCODataModule  # default dataset if user gives minimal YAML
from .evaluate import run_full_evaluation

###############################
# ---------- Utils ---------- #
###############################

def set_global_seed(seed: int = 42):
    """Seed RNGs for reproducibility while keeping CuDNN fast."""
    import random, numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def pretty_print_json(obj: Dict[str, Any]):
    import pprint

    print("----- Experiment Result JSON -----")
    pprint.pprint(obj, indent=2)
    print("----- End JSON -----")

########################################
# ---------- Model Building ---------- #
########################################

class HaarWavelet(nn.Module):
    """Level-1 2-D Haar split & merge via (transpose-)convs so that everything is autograd-friendly."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer("filt", filt)

    # ----------------------- split ----------------------- #
    def forward(self, x: torch.Tensor):
        # x: (B,C,H,W)
        weight = self.filt.repeat_interleave(x.size(1), dim=0)
        y = torch.nn.functional.conv2d(x, weight, stride=2, groups=x.size(1))
        B, _, H, W = y.shape
        y = y.view(B, x.size(1), 4, H, W)  # (B,C,4,H/2,W/2)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, lh, hl, hh

    # ----------------------- merge ----------------------- #
    def inverse(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C * 4, H, W)
        weight = self.filt.repeat_interleave(C, dim=0)
        x = torch.nn.functional.conv_transpose2d(y, weight, stride=2, groups=C)
        return x


class ResidualMixer(nn.Module):
    """Lightweight Transformer mixer operating on (flattened) spatial tokens."""

    def __init__(self, emb_channels: int = 320, depth: int = 4, num_heads: int = 8):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=emb_channels, nhead=num_heads, batch_first=True)
        self.tfm = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, C)
        x = self.tfm(x)
        return self.norm(x)


class FCRD1Step(nn.Module):
    """Frequency-Conditioned Residual Distillation 1-step student."""

    def __init__(self, hidden_size: int = 192, mixer_depth: int = 4):
        super().__init__()
        from diffusers.models.unet_2d_condition import UNet2DConditionModel

        self.wave = HaarWavelet()
        # Tiny UNet – processes only LL band (quarter resolution)
        self.backbone = UNet2DConditionModel(
            sample_size=256,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            block_out_channels=(hidden_size,),
        )
        self.mixer = ResidualMixer(emb_channels=hidden_size, depth=mixer_depth)

    # ---------------------------------------------------- #
    def forward(self, latents: torch.Tensor, prompt_emb: torch.Tensor):
        # Wavelet split
        ll, lh, hl, hh = self.wave(latents)
        ll_hat = self.backbone(ll, encoder_hidden_states=prompt_emb)[0]

        # Upsample LL & form token sequence
        ll_up = torch.nn.functional.interpolate(ll_hat, scale_factor=2, mode="bilinear", align_corners=False)
        tokens = torch.cat([ll_up, lh, hl, hh], dim=1).flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.mixer(tokens)
        tokens = tokens.transpose(1, 2).view_as(torch.cat([ll_up, lh, hl, hh], dim=1))

        ll_res, lh_res, hl_res, hh_res = tokens.chunk(4, dim=1)
        ll_final = ll_up  # coarse path
        lh_final, hl_final, hh_final = lh + lh_res, hl + hl_res, hh + hh_res
        out = self.wave.inverse(ll_final, lh_final, hl_final, hh_final)
        return out

##########################################
# ---------- Training Pipeline --------- #
##########################################

class Trainer:
    """Very small convenience trainer (single-GPU)."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        cfg: Dict[str, Any],
        project_root: Path,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dl, self.val_dl = train_dataloader, val_dataloader
        self.cfg = cfg
        self.root = Path(project_root)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg["lr"], betas=(0.0, 0.99), weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg["max_steps"])
        self.loss_fn = nn.MSELoss()

        self.ckpt_dir = self.root / "models" / cfg["experiment_name"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0

    # ----------------------------- private helpers ----------------------------- #
    def _step(self, batch: Tuple[torch.Tensor, Any]):
        imgs, _ = batch  # captions not used for the toy training loop
        imgs = imgs.to(self.device, non_blocking=True)
        noise = torch.randn_like(imgs)
        latents = imgs + noise  # single-step schedule (t≈1)
        prompt_emb = torch.randn(imgs.size(0), 77, 256, device=self.device)  # TODO: swap for real encoder emb.
        preds = self.model(latents, prompt_emb)
        loss = self.loss_fn(preds, imgs)
        return loss

    # --------------------------------------------------------------------------- #
    def fit(self) -> Path:
        scaler = amp.GradScaler()
        pbar = tqdm.tqdm(total=self.cfg["max_steps"], dynamic_ncols=True)
        self.model.train()
        while self.global_step < self.cfg["max_steps"]:
            for batch in self.train_dl:
                self.opt.zero_grad(set_to_none=True)
                with amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
                    loss = self._step(batch)
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
                self.scheduler.step()

                self.global_step += 1
                pbar.set_description(f"loss={loss.item():.4f}")
                pbar.update(1)

                if self.global_step % self.cfg["checkpoint_every"] == 0:
                    ckpt_path = self.ckpt_dir / f"step_{self.global_step}.pt"
                    torch.save(self.model.state_dict(), ckpt_path)

                if self.global_step >= self.cfg["max_steps"]:
                    break
        # final checkpoint
        final_ckpt = self.ckpt_dir / "final.pt"
        torch.save(self.model.state_dict(), final_ckpt)
        return final_ckpt

###########################################
# ---------- Experiment Driver ---------- #
###########################################

def _instantiate_from_cfg(obj_cfg: Dict[str, Any]):
    module = importlib.import_module(obj_cfg["module"])
    cls = getattr(module, obj_cfg["class"])
    return cls(**obj_cfg.get("kwargs", {}))


def run_experiment(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Full train-eval loop for a single experiment config."""

    set_global_seed(cfg.get("seed", 42))

    # 1) Data
    dm_cfg = cfg["data_module"]
    DataModuleCls = _instantiate_from_cfg(dm_cfg)
    data_module = DataModuleCls(**dm_cfg.get("kwargs", {})) if callable(DataModuleCls) else DataModuleCls
    train_dl, val_dl, test_dl = data_module.dataloaders()

    # 2) Model (load or train)
    model_cfg = cfg["model"]
    model = _instantiate_from_cfg(model_cfg)
    ckpt_path = Path(model_cfg.get("checkpoint", ""))
    if ckpt_path.exists():
        print(f"Loading checkpoint → {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        trainer = Trainer(model, train_dl, val_dl, cfg["trainer"], project_root)
        ckpt_path = trainer.fit()

    # 3) Evaluation
    result_dict, fig_paths = run_full_evaluation(model, test_dl, cfg["evaluation"], project_root)
    result_dict["figures"] = [str(p) for p in fig_paths]
    return result_dict

from __future__ import annotations

"""
train.py
---------
All model definitions and the training loop live here.  The only side-effect
of this module is that it writes intermediate checkpoints/results through the
`evaluate.save_and_plot_results` helper so that plotting and JSON handling are
kept in a single place.

NOTE:  The code is extracted verbatim from the original single-file script with
only the absolutely necessary changes for import paths.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 – may be useful for future loss

try:
    import deepspeed  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "DeepSpeed not found – please install the pinned version before running."
    ) from exc

from .preprocess import DatasetManager
from .evaluate import save_and_plot_results

# -----------------------------------------------------------------------------
#  Selective-State-Space primitives & blocks
# -----------------------------------------------------------------------------


class SelectiveSSM2D(nn.Module):
    """A very thin wrapper around the 1-D selective SSM that simply flattens
    the H×W spatial grid into a sequence.  The heavy CUDA kernels of Mamba are
    lazily loaded at import-time by `mamba_ssm`."""

    def __init__(self, d_model: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mamba-ssm package not found – please install mamba-ssm>=1.1"
            ) from exc

        self.core = Mamba(d_model)

    def forward(self, x: torch.Tensor):  # [B, N, C]
        return self.core(x)


class HSSDBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM2D(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Simple conv-patch projection used by HSSD-B."""

    def __init__(self, in_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch, patch)

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]  ->  [B, N, C]
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        return x, (h, w)


class HSSDB(nn.Module):
    """Hierarchical Selective-State Diffusion – Base size implementation."""

    def __init__(self, img_size: int = 256, stages: int = 3, dim: int = 1536, layers: int = 24):
        super().__init__()
        assert stages == 3, "Current implementation hard-codes a 3-stage cascade"
        self.patch_embed = PatchEmbed(3, dim, patch=8)  # 32× down-scale on 256²
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // 8) ** 2, dim))
        self.blocks = nn.ModuleList([HSSDBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        # Tiny adapter example (stage-aware adapters would live here)
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        x, _ = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x + self.adapter(x)
        return x  # feature tokens


# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------


class Trainer:
    """Wraps the whole training loop including DeepSpeed initialisation."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device("cuda", self.local_rank)

        # ------------------------------------------------------------------
        self.dataset_mgr = DatasetManager(cfg)
        self.results: Dict[str, Any] = {}
        self._build_model_optimizer()

    # ------------------------------------------------------------------
    def _build_model_optimizer(self):
        self.model = HSSDB(img_size=256).to(self.device)

        # DeepSpeed configuration mirrors the one from the original script
        ds_config: Dict[str, Any] = {
            "train_batch_size": self.cfg["training"]["batch_size"],
            "gradient_accumulation_steps": self.cfg["training"]["accumulation_steps"],
            "zero_optimization": {"stage": 3, "offload_param": {"device": "cpu"}},
            "bf16": {"enabled": True},
            "steps_per_print": 1000,
            "wall_clock_breakdown": False,
        }
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg["training"]["base_lr"],
            weight_decay=self.cfg["training"]["weight_decay"],
        )
        (self.model, self.optimizer, _, _) = deepspeed.initialize(
            model=self.model, optimizer=self.optimizer, config=ds_config
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fid_placeholder() -> float:
        """Placeholder that always returns 999.  Replace with real FID eval."""

        return 999.0

    # ------------------------------------------------------------------
    def fit(self, seed: int):
        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)

        loader = self.dataset_mgr.imagenet_loader(
            "train", img_size=256, batch_per_gpu=256, seed=seed
        )

        max_iters = int(1e9)  # effectively unlimited; we early-stop via metrics
        log_interval = 100
        eval_every = self.cfg["training"]["eval_every"]

        start = time.time()
        total_flops = 0.0  # PFLOP accounting would live here in a real setup

        for step, (imgs, _) in enumerate(loader, start=1):
            imgs = imgs.to(self.device, non_blocking=True)
            outputs = self.model(imgs)
            loss = outputs.pow(2).mean()  # dummy loss – replace with diffusion obj
            self.model.backward(loss)
            self.model.step()

            if step % log_interval == 0 and self.local_rank == 0:
                print(f"Step {step:>7d} | Loss {loss.item():.4f}")

            # ---------------- evaluation ----------------
            if step % eval_every == 0 and self.local_rank == 0:
                fid = self._fid_placeholder()
                wall_h = (time.time() - start) / 3600.0
                self.results.setdefault("fid_curve", []).append(
                    {"step": step, "fid": fid, "wall_h": wall_h}
                )
                print(
                    f"[Eval] step {step}  FID-50k = {fid:.2f} | wall = {wall_h:.2f} h"
                )

                if fid <= self.cfg["training"]["target_fid"]:
                    print("Target FID reached – stopping training.")
                    break
                if total_flops >= self.cfg["training"]["max_pfops"]:
                    print("PFLOP budget exhausted – stopping training.")
                    break

            if step >= max_iters:
                break

        # ---------------- wrap-up ----------------
        elapsed = time.time() - start
        self._finalise(seed, elapsed)

    # ------------------------------------------------------------------
    def _finalise(self, seed: int, elapsed_sec: float):
        out_dir = Path(self.cfg["output"]["root"]) / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_and_plot_results(self.results, out_dir, self.local_rank)

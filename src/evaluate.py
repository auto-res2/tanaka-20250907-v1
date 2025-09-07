"""src/evaluate.py
Metric computation, plotting & full evaluation loop.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from open_clip import create_model_and_transforms, tokenize

from .train import HaarWavelet  # re-use implementation

###############################
# -------- Metrics ---------- #
###############################

class MetricComputer:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(feature=2048, reset_real_features=False).to(self.device)
        self.incep = torchmetrics.image.inception.InceptionScore().to(self.device)
        self.clip_model, _, self.clip_preproc = create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.wave = HaarWavelet().to(self.device)

    # ----------------------------- streaming updates ----------------------------- #
    @torch.no_grad()
    def update_real(self, imgs: torch.Tensor):
        self.fid.update(imgs.to(self.device), real=True)

    @torch.no_grad()
    def update_fake(self, imgs: torch.Tensor):
        self.fid.update(imgs.to(self.device), real=False)
        self.incep.update(imgs)

    # ----------------------------------------------------------------------------- #
    @torch.no_grad()
    def compute_clipscore(self, imgs: torch.Tensor, texts: List[str]) -> float:
        txt_tokens = tokenize(texts).to(self.device)
        img_in = self.clip_preproc(imgs.cpu()).to(self.device)
        img_feat = self.clip_model.encode_image(img_in)
        txt_feat = self.clip_model.encode_text(txt_tokens)
        return torch.cosine_similarity(img_feat, txt_feat).mean().item()

    @torch.no_grad()
    def high_freq_error(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        _, _, _, hh_r = self.wave(real.to(self.device))
        _, _, _, hh_f = self.wave(fake.to(self.device))
        return F.mse_loss(hh_r, hh_f).item()

########################################
# ---------- Plot utilities ---------- #
########################################

def _plot_dummy_curve(fid_metric: torchmetrics.image.fid.FrechetInceptionDistance, out_dir: Path) -> Path:
    xs = list(range(len(fid_metric.real_features)))
    ys = [0.0 for _ in xs]
    plt.figure()
    plt.plot(xs, ys, label="FID (placeholder)")
    plt.xlabel("Iteration")
    plt.ylabel("FID")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_curve_fid.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

########################################
# ---------- Full evaluation ---------- #
########################################

def run_full_evaluation(
    model: torch.nn.Module,
    dataloader,
    eval_cfg: Dict[str, Any],
    project_root: Path,
) -> Tuple[Dict[str, float], List[Path]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    metric = MetricComputer(device)

    prompts: List[str] = []
    fake_imgs: List[torch.Tensor] = []

    with torch.no_grad():
        for imgs, caps in tqdm.tqdm(dataloader, desc="Evaluation", dynamic_ncols=True):
            imgs = imgs.to(device)
            noise = torch.randn_like(imgs)
            latents = imgs + noise
            prompt_emb = torch.randn(imgs.size(0), 77, 256, device=device)  # same TODO as training loop
            gen = model(latents, prompt_emb)

            metric.update_real(imgs)
            metric.update_fake(gen)
            prompts.extend(caps)
            fake_imgs.append(gen.cpu())

    fid = metric.fid.compute().item()
    iscore = metric.incep.compute()[0].item()
    clip_score = metric.compute_clipscore(torch.cat(fake_imgs), prompts)
    hfe = metric.high_freq_error(imgs, gen)

    result = {"FID": fid, "IS": iscore, "CLIPScore": clip_score, "HFE": hfe}
    fig_dir = project_root / ".research" / "iteration1" / "images"
    fig_path = _plot_dummy_curve(metric.fid, fig_dir)
    return result, [fig_path]

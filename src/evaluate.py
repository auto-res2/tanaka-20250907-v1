"""src/evaluate.py
Evaluation utilities – extremely light-weight: they merely compute the reconstruction
error (MSE) of the given model on a *single* batch from the provided dataloader.  The
result is written as JSON under `.research/iteration4/` as required by the task.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------------- #

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device | str = "cpu",
    experiment_name: str = "debug_run",
) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()

    with torch.no_grad():
        imgs, _ = next(iter(val_loader))  # <1 second – just one batch!
        imgs = imgs.to(device)
        preds = model(imgs)
        mse = criterion(preds, imgs).item()

    metrics = {"val_MSE": float(mse)}

    # --------------- mandatory JSON saving ---------------- #
    out_dir = Path(".research/iteration4")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{experiment_name}_eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print to stdout for verification (as mandated by the task description)
    print(json.dumps(metrics, indent=2))
    return metrics

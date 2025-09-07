from __future__ import annotations

"""
preprocess.py
-------------
Contains utilities for verifying, downloading (where licence allows) and
loading datasets.  All heavy lifting for preparing PyTorch `DataLoader`s is
centralised here so that `train.py` stays focused on the model logic.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as T

__all__ = ["DatasetManager"]


class DatasetManager:
    """Download (if public) and validate dataset availability."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _download_zip(self, url: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return  # already downloaded
        print(f"Downloading {url} …", flush=True)
        subprocess.run(["curl", "-L", url, "-o", str(dest)], check=True)
        print("Download finished – extracting …", flush=True)
        subprocess.run(["unzip", "-q", str(dest), "-d", str(dest.parent)], check=True)
        dest.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_dataset(self, name: str) -> Path:
        entry = self.cfg["datasets"].get(name)
        if entry is None:
            raise RuntimeError(f"Dataset '{name}' not configured in YAML.")

        path = Path(entry["path"])
        if path.exists() and any(path.iterdir()):
            print(f"✓ Dataset {name} found at {path}")
            return path

        # ImageNet-1K requires manual download/registration.
        if name == "imagenet_1k":
            raise RuntimeError(
                "ImageNet-1K not found – please download from the official "
                "site and place it under 'data/imagenet_1k'."
            )

        # Attempt auto-download for public datasets.
        url = entry["url"]
        zip_target = path.with_suffix(".zip")
        self._download_zip(url, zip_target)
        if not path.exists():
            raise RuntimeError(
                f"Automatic download of {name} failed.  Please check the URL or network connectivity."
            )
        return path

    # ------------------------------------------------------------------
    # ImageNet-style DataLoader
    # ------------------------------------------------------------------
    def imagenet_loader(self, split: str, img_size: int, batch_per_gpu: int, seed: int):
        path = self.ensure_dataset("imagenet_1k") / split
        if not path.exists():
            raise RuntimeError(f"ImageNet split '{split}' not found in {path}")

        tx = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                # scale to [-1, 1]
                transforms.Normalize(0.5, 0.5),
            ]
        )

        ds = ImageFolder(path, transform=tx)
        g = torch.Generator()
        g.manual_seed(seed)

        loader = DataLoader(
            ds,
            batch_size=batch_per_gpu,
            shuffle=True,
            num_workers=min(os.cpu_count() or 8, 8),
            pin_memory=True,
            generator=g,
            drop_last=True,
        )
        return loader

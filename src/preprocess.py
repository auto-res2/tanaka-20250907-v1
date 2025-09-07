"""src/preprocess.py
Dataset loading & downloading utilities extracted from the original monolith.
Only COCO is implemented – extend as needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Tuple

import requests
import tarfile
import tqdm
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

#######################################################################
# -------------------------- Downloader ----------------------------- #
#######################################################################

_CHUNK = 1 << 16  # 64 KiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, sha256: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256 and _sha256(target) != sha256:
            print("Checksum mismatch – re-downloading.")
            target.unlink()
        else:
            print(f"File {target} already present – skipping download.")
            return target

    print(f"Downloading {url} → {target}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm.tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    if sha256 and _sha256(target) != sha256:
        raise RuntimeError(f"SHA-256 mismatch for {target}")
    return target


def extract(archive: Path, dest: Path):
    print(f"Extracting {archive} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif str(archive).endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type: {archive}")

#######################################################################
# --------------------------- COCO Data ----------------------------- #
#######################################################################

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2014.zip",
    "val_images": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


class COCODataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform or T.Compose(
            [
                T.CenterCrop(512),
                T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        self._ensure_dataset()
        anno_file = self.root / "annotations" / f"captions_{split}2014.json"
        with open(anno_file) as f:
            captions = json.load(f)["annotations"]
        self.imgid2capt = {}
        for c in captions:
            self.imgid2capt.setdefault(c["image_id"], []).append(c["caption"])
        self.ids = sorted(self.imgid2capt.keys())

    # --------------------------------------------------------------------- #
    def _ensure_dataset(self):
        if not (self.root / "annotations").exists():
            ann_zip = download(COCO_URLS["annotations"], self.root / "annotations.zip")
            extract(ann_zip, self.root)
        img_key = "train_images" if self.split == "train" else "val_images"
        if not (self.root / f"{self.split}2014").exists():
            img_zip = download(COCO_URLS[img_key], self.root / f"{self.split}2014.zip")
            extract(img_zip, self.root)

    # --------------------------------------------------------------------- #
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_path = self.root / f"{self.split}2014" / f"COCO_{self.split}2014_{img_id:012d}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caption = self.imgid2capt[img_id][0]  # deterministic choice
        return img, caption


class COCODataModule:
    """Lightweight wrapper that returns ready-to-use PyTorch dataloaders."""

    def __init__(self, root: str = "./data/coco", batch_size: int = 16, num_workers: int = 8):
        self.root = root
        self.bs = batch_size
        self.nw = num_workers

    # --------------------------------------------------------------------- #
    def dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = COCODataset(self.root, "train")
        val_ds = COCODataset(self.root, "val")
        test_ds = val_ds  # reuse val split for quick smoke tests
        train_dl = DataLoader(
            train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.nw,
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
        )
        return train_dl, val_dl, test_dl

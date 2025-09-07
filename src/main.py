"""src/main.py
Entry-point that is executed via `python -m src.main`.
Parses the YAML config, instantiates objects via `importlib`, and calls the stub
training + evaluation routines defined elsewhere in the repository.

Heavy-weight training/eval is *out of scope* – we only perform a **smoke test** that
runs a couple of iterations to ensure the pipeline is wired correctly.
"""
from __future__ import annotations

import argparse
import importlib
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import numpy as np
import torch
import yaml

# ----------------------------------------------------------------------------- #
# --------------------------- Utility helpers ------------------------------ #
# ----------------------------------------------------------------------------- #

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dynamic_import(module_path: str, attr: str):
    """import `module_path`, then return `getattr(module, attr)`"""
    module: ModuleType = importlib.import_module(module_path)
    return getattr(module, attr)


# ----------------------------------------------------------------------------- #
# ------------------------------ Main logic --------------------------------- #
# ----------------------------------------------------------------------------- #

def _run_single_experiment(exp_cfg: Dict[str, Any]):
    exp_name: str = exp_cfg["experiment"]["name"]
    seed: int = exp_cfg.get("seed", 0)
    _set_seed(seed)

    # ---------------- data ---------------- #
    dm_cfg = exp_cfg["data_module"]
    DMClass = _dynamic_import(dm_cfg["module"], dm_cfg["class"])
    data_module = DMClass(**dm_cfg.get("kwargs", {}))
    train_dl, val_dl, _ = data_module.dataloaders()

    # ---------------- model ---------------- #
    model_cfg = exp_cfg["model"]
    ModelClass = _dynamic_import(model_cfg["module"], model_cfg["class"])
    model = ModelClass(**model_cfg.get("kwargs", {}))

    # ---------------- trainer ---------------- #
    trainer_cfg = exp_cfg["trainer"]
    from src.train import run_training
    model = run_training(
        model,
        train_dl,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=trainer_cfg.get("lr", 3e-4),
        max_steps=trainer_cfg.get("max_steps", 1),
        experiment_name=exp_name,
    )

    # ---------------- evaluation ---------------- #
    from src.evaluate import evaluate

    evaluate(
        model,
        val_dl,
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name=exp_name,
    )


# ----------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run experiments defined in a YAML config file.")
    parser.add_argument(
        "--config", default="config/config.yaml", type=str, help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    experiments = cfg.get("experiments", [])
    if not experiments:
        raise RuntimeError("No experiments found in the config file.")

    for exp in experiments:
        _run_single_experiment(exp)


if __name__ == "__main__":
    main()

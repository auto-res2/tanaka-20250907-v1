"""
main.py (entry-point)
---------------------
This is the orchestration layer that glues everything together.  It loads the
YAML configuration, instantiates a `Trainer` object (imported from `train.py`)
for each random seed and, after each run, clears the CUDA cache to keep peak
memory predictable.

Execute via:   python -m src.main
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

try:
    from .train import Trainer
except ImportError as exc:  # pragma: no cover
    # Give a slightly nicer error if someone runs the module outside `src`.
    raise ImportError(
        "Could not import 'train'.  Please run as a module:  python -m src.main"
    ) from exc

# -----------------------------------------------------------------------------
#  Configuration loader
# -----------------------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}.  Did you forget to create it?"
        )
    with open(config_path, "r", encoding="utf-8") as fp:
        cfg: Dict[str, Any] = yaml.safe_load(fp)
    return cfg


# -----------------------------------------------------------------------------
#  Main driver
# -----------------------------------------------------------------------------


def run_all_experiments():
    cfg = _load_config()

    # ---------------- Experiment 1 (HSSD-B) ----------------
    for seed in cfg["training"]["seeds"]:
        trainer = Trainer(cfg)
        trainer.fit(seed)
        torch.cuda.empty_cache()

    # Experiments 2 & 3 would follow the same skeleton and can be implemented
    # later by extending the Evaluate helpers.


if __name__ == "__main__":
    try:
        run_all_experiments()
    except RuntimeError as err:
        # Fail hard but with a clean error message – mirrors the original script.
        print(f"Execution halted: {err}", file=sys.stderr)
        sys.exit(1)

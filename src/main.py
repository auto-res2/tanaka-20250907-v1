"""src/main.py
Entry-point: orchestrates reading config, launching experiments, persisting JSON & images.
Run via `python -m src.main`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from .train import pretty_print_json, run_experiment

# --------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent  # project root (one above src/)
CONFIG_PATH = ROOT / "config" / "config.yaml"
RESULT_DIR = ROOT / ".research" / "iteration1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #

def _load_yaml(path: Path) -> List[Dict]:
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        print(f"❌ Failed to read YAML config at {path}: {exc}")
        sys.exit(1)

    # Accept either {experiments: [...]} or a single-experiment dict
    if isinstance(cfg, dict) and "experiments" in cfg:
        return cfg["experiments"]
    return [cfg]  # single experiment

# --------------------------------------------------------------------- #

def main():
    exp_cfgs = _load_yaml(CONFIG_PATH)
    if not exp_cfgs:
        raise RuntimeError("No experiments specified in config/config.yaml – aborting.")

    all_results = {}
    for cfg in exp_cfgs:
        exp_name = cfg["experiment"]["name"]
        print(f"\n=========== Running experiment: {exp_name} ===========\n")
        res = run_experiment(cfg, ROOT)
        all_results[exp_name] = res

        # Persist per-experiment JSON under .research/iteration1
        out_json = RESULT_DIR / f"{exp_name}.json"
        with open(out_json, "w") as fp:
            json.dump(res, fp, indent=2)
        pretty_print_json(res)

    print("\nAll experiments finished. JSON result files saved under .research/iteration1/\n")


if __name__ == "__main__":
    main()

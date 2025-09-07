"""
evaluate.py
-----------
Utility helpers for saving experiment metrics and producing publication-quality
plots.  All heavy numerical evaluation (e.g. FID computation) should live here
once implemented.  For now we only have a JSON dump and a dummy curve plot so
that the refactored script reproduces the exact behaviour of the monolithic
version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless backend for server environments
import matplotlib.pyplot as plt  # noqa: E402 (after Agg backend)

__all__ = ["save_and_plot_results"]


def _plot_fid_curve(results: Dict[str, Any], fig_path: Path):
    """Persist a simple FID-over-time figure (dummy values until real FID)."""

    if not results.get("fid_curve"):
        return

    steps = [p["step"] for p in results["fid_curve"]]
    fids = [p["fid"] for p in results["fid_curve"]]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, fids, marker="o", label="HSSD-B")
    for x, y in zip(steps, fids):
        plt.text(x, y, f"{y:.1f}")
    plt.xlabel("Training step")
    plt.ylabel("FID-50k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def save_and_plot_results(results: Dict[str, Any], out_dir: Path, local_rank: int):
    """Dump results to JSON and produce a PDF plot.  All printing happens only
    on rank-0 to avoid duplicated stdout in distributed jobs."""

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) JSON dump --------------------------------------------------------
    json_path = out_dir / "exp1_cost_quality.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # 2) Plot -------------------------------------------------------------
    fig_path = out_dir / "training_fid_curve_hssd.pdf"
    _plot_fid_curve(results, fig_path)

    # 3) Console summary --------------------------------------------------
    if local_rank == 0:
        print("\n=== Experiment 1 – Cost-for-Quality Benchmark (HSSD-B) ===")
        print(json.dumps(results, indent=2))
        if fig_path.exists():
            print(f"Figures produced: {fig_path.name}")

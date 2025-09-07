"""
evaluate.py
-----------
Utility helpers for saving experiment metrics and producing publication-quality
plots.  All heavy numerical evaluation (e.g. FID computation) should live here
once implemented.

The specification for this iteration requires that *all* artefacts are written
under:
  • JSON files ..........  .research/iteration2/
  • Figures (images) .....  .research/iteration2/images/

Those directories are created on-the-fly.  Each experiment/seed combination
gets its own pair of files so that multiple runs do not overwrite each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # headless backend for server environments
import matplotlib.pyplot as plt  # noqa: E402 (after Agg backend)

__all__ = ["save_and_plot_results"]

# -----------------------------------------------------------------------------
#  Constants – centralised here to guarantee the correct on-disk layout
# -----------------------------------------------------------------------------

RESULTS_ROOT = Path(".research/iteration2")
IMAGES_DIR = RESULTS_ROOT / "images"


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _plot_fid_curve(results: Dict[str, Any], fig_path: Path):  # noqa: D401
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


# -----------------------------------------------------------------------------
#  Public API
# -----------------------------------------------------------------------------

def save_and_plot_results(results: Dict[str, Any], out_dir: Path, local_rank: int):
    """Dump results to JSON and produce a PDF plot.

    The *out_dir* argument is used only for naming (so that callers do not need
    to know the global folder structure enforced by this helper).
    """

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1) JSON dump --------------------------------------------------------
    json_path = RESULTS_ROOT / f"{out_dir.name}.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # 2) Plot -------------------------------------------------------------
    fig_path = IMAGES_DIR / f"{out_dir.name}_fid_curve.pdf"
    _plot_fid_curve(results, fig_path)

    # 3) Console summary --------------------------------------------------
    if local_rank == 0:
        print("\n=== Experiment Summary (HSSD-B) ===")
        print(json.dumps(results, indent=2))
        print("Outputs written:")
        print(f"  • JSON ....... {json_path}")
        if fig_path.exists():
            print(f"  • Figure ..... {fig_path}")

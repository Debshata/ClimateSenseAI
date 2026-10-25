"""Shared matplotlib style for publication figures (300 DPI PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical order (identity) and an ordinal ramp for the four risk classes (magnitude)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RISK_COLORS = ["#f5d9b8", "#eda15c", "#d4572a", "#8c1d18"]  # Low -> Critical, one warm hue light->dark
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "lines.linewidth": 2.0, "axes.prop_cycle": matplotlib.cycler(color=SERIES), "font.family": "DejaVu Sans",
    })


def save(fig, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

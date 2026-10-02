#!/usr/bin/env python
"""Scientific figure for fixed-grid control cadence component comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--boundary-source", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    full = json.loads(Path(args.source).read_text())
    boundary = json.loads(Path(args.boundary_source).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    colors = ["#d66b41", "#328a91", "#253f79"]
    for rate, color in zip([50., 100., 200.], colors):
        rows = sorted([r for r in full["records"] if r["control_frequency_hz"] == rate], key=lambda r:r["n_links"])
        counts = [r["n_links"] for r in rows]
        axes[0].plot(counts, [r["fastest_mode_growth_per_control_interval"] for r in rows], "o-", color=color, label=f"{rate:g} Hz")
        axes[1].semilogy(counts, [r["high_precision_diagnostics"]["gain_norm"] for r in rows], "o-", color=color, label=f"{rate:g} Hz")
    axes[0].set(title="Faster updates limit growth per step", xlabel="Links", ylabel="Fastest upright mode: amplification / step")
    axes[1].set(title="These faster-rate designs use larger gains", xlabel="Links", ylabel="LQR gain norm (physical coordinates)")
    for axis in axes[:2]:
        axis.set_xticks([10, 11, 12, 14, 20])
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    rows = sorted([r for r in boundary["records"] if r["n_links"] == 11], key=lambda r:r["control_frequency_hz"])
    amplitudes = ["1e-06", "1e-07", "1e-08", "1e-09", "1e-10", "1e-11"]
    directions = boundary["parameters"]["directions"]
    successes = np.array([[r["local_capture"][amp]["successes"] for amp in amplitudes] for r in rows])
    axes[2].imshow(successes, aspect="auto", vmin=0, vmax=directions, cmap="Blues")
    axes[2].set(title="Eleven-link local replay: small benefit", xlabel="Initial joint angle/rate amplitude", ylabel="Control rate (Hz)",
                xticks=range(len(amplitudes)), xticklabels=["10⁻⁶", "10⁻⁷", "10⁻⁸", "10⁻⁹", "10⁻¹⁰", "10⁻¹¹"],
                yticks=range(len(rows)), yticklabels=[int(r["control_frequency_hz"]) for r in rows])
    for i in range(len(rows)):
        for j in range(len(amplitudes)):
            axes[2].text(j, i, f"{successes[i,j]}/{directions}", ha="center", va="center", fontsize=8,
                         color="white" if successes[i,j] > directions/2 else "black")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .026, "Fixed 0.005 s RK4 physics; gains redesigned per rate. Four matched directions per amplitude; 8 s upright-start probes, not hanging-start solves.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, 1))
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix)+".png", dpi=180)
    fig.savefig(str(prefix)+".pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

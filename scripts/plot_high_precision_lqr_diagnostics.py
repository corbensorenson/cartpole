#!/usr/bin/env python
"""Plot saved linear-design diagnostics and finite local capture probes."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    rows = sorted(source["records"], key=lambda row: row["n_links"])
    counts = np.array([row["n_links"] for row in rows])
    diagnostics = [row["high_precision_diagnostics"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    axes[0].semilogy(counts, [row["gain_norm"] for row in diagnostics], "o-", color="#176b87")
    axes[0].set(title="Upright gain grows rapidly", xlabel="Links", ylabel="Gain norm (physical joint coordinates)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].semilogy(counts, [row["rounded_gain_numpy_spectral_radius"] for row in diagnostics],
                     "o-", label="Float64 assembly + eigenvalues", color="#c85434")
    axes[1].semilogy(counts, [float(row["rounded_gain_high_precision_spectral_radius"]) for row in diagnostics],
                     "s-", label="80–100 digit assembly + eigenvalues", color="#176b87")
    axes[1].axhline(1, color="0.4", linestyle="--", linewidth=1)
    axes[1].set(title="Closed-loop spectrum is sensitive", xlabel="Links", ylabel="Computed spectral radius")
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(axis="y", alpha=0.2)
    amplitudes = ["1e-09", "1e-11", "1e-13"]
    success = np.array([[row["local_capture"][amp]["successes"] for amp in amplitudes] for row in rows])
    image = axes[2].imshow(success, vmin=0, vmax=source["parameters"]["directions"], cmap="Blues", aspect="auto")
    axes[2].set(title="Exact local 8-second replay", xlabel="Initial joint angle/rate amplitude", ylabel="Links",
                xticks=range(3), xticklabels=["10⁻⁹", "10⁻¹¹", "10⁻¹³"],
                yticks=range(len(counts)), yticklabels=counts)
    for i in range(len(rows)):
        for j in range(3):
            axes[2].text(j, i, f"{success[i,j]}/4", ha="center", va="center",
                         color="white" if success[i,j] > 2 else "black", fontsize=8)
    for axis in axes[:2]:
        axis.set_xticks(counts[::2])
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.025,
             "Fixed upright starts; four sampled directions per amplitude. Linear matrices are promoted binary64 inputs. No hanging-start solves are shown.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + ".png", dpi=180)
    fig.savefig(str(prefix) + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

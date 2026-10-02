#!/usr/bin/env python
"""Plot physical development replays around the capture handoff."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gcartpole.evidence import file_metadata, utc_timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--start", type=float, default=7.5)
    parser.add_argument("--end", type=float, default=10.5)
    args = parser.parse_args()
    if not 0 <= args.start < args.end:
        raise ValueError("ordered nonnegative plot interval required")
    directory = Path(args.out_directory); directory.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.6), sharex=True)
    inputs = []
    count = None
    for index, item in enumerate(args.input):
        label, path = item.split("=", 1)
        payload = json.loads(Path(path).read_text())
        n = len(payload["selected_state"]["qpos"])-1
        if count is not None and n != count:
            raise ValueError("comparison requires matching link counts")
        count = n
        result = payload["result"]
        if not result["trajectory_integrity"]:
            raise ValueError("comparison requires intact physical replays")
        rows = result["trajectory"]
        time = np.array([r["time_seconds"] for r in rows])
        selected = (time >= args.start) & (time <= args.end)
        legend = f"{label} ({result['max_upright_streak_seconds']:.2f} s hold)"
        keys = ["x", "max_abs_angle", "action", "dimensionless_lyapunov_value"]
        for axis, key in zip(axes, keys):
            values = np.array([r[key] for r in rows])
            if key in ["max_abs_angle", "dimensionless_lyapunov_value"]:
                values = np.maximum(values, 1e-12)
            axis.plot(time[selected], values[selected], label=legend, color=f"C{index}", lw=1.5)
        inputs.append(dict(label=label, artifact=file_metadata(path),
                           success=result["success"], hold_seconds=result["max_upright_streak_seconds"],
                           handoff_seconds=result["first_handoff_time"],
                           termination_reason=result["termination_reason"]))
    axes[0].axhline(3, color=".5", ls=":", lw=1); axes[0].axhline(-3, color=".5", ls=":", lw=1)
    axes[1].axhline(.15, color=".5", ls=":", lw=1)
    axes[2].axhline(1, color=".5", ls=":", lw=1); axes[2].axhline(-1, color=".5", ls=":", lw=1)
    axes[1].set_yscale("log"); axes[3].set_yscale("log")
    for axis, ylabel in zip(axes, ["Cart position (m)", "Largest absolute angle (rad)", "Normalized cart action", "Upright Lyapunov value"]):
        axis.set_ylabel(ylabel); axis.grid(alpha=.2)
        for index, comparison in enumerate(inputs):
            if comparison["handoff_seconds"] is not None:
                axis.axvline(comparison["handoff_seconds"], color=f"C{index}", ls="--", lw=.9, alpha=.6)
    axes[0].set_title(f"{count}-link capture handoff: physical development replays", loc="left", fontsize=13)
    axes[-1].set_xlabel("Episode time (s)"); axes[-1].set_xlim(args.start, args.end)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .035*len(inputs)+.015, 1, 1))
    fig.savefig(directory/"capture_handoff.png", dpi=190)
    fig.savefig(directory/"capture_handoff.pdf")
    plt.close(fig)
    (directory/"figure.json").write_text(json.dumps(dict(generated_at=utc_timestamp(),
        not_solution=True, scope="Development physical replay only; no noisy evaluation or promotion claim.",
        parameters=vars(args), inputs=inputs), indent=2)+"\n")


if __name__ == "__main__":
    main()

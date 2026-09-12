#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gcartpole.config import apply_overrides, load_config
from gcartpole.linear import analyze_morphology


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep simplified linear controllability metrics")
    parser.add_argument("--config", default="configs/sweep_n.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    sweep = cfg["sweep"]
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "linear_sweep.csv"

    rows = []
    for n in range(int(sweep["n_min"]), int(sweep["n_max"]) + 1):
        for a_l in sweep["alpha_length_values"]:
            for a_m in sweep["alpha_mass_values"]:
                metrics = analyze_morphology(
                    n_links=n,
                    total_length=float(sweep["total_length"]),
                    total_mass=float(sweep["total_mass"]),
                    cart_mass=float(sweep["cart_mass"]),
                    alpha_length=float(a_l),
                    alpha_mass=float(a_m),
                    total_damping=float(sweep.get("total_damping", 0.0)),
                )
                rows.append(metrics.to_dict())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print the best-looking row by a simple heuristic.
    def score(row):
        return row["lambda_max_pos"] + 0.05 * __import__("math").log10(max(row["controllability_cond"], 1.0)) - 2.0 * row["weakest_unstable_coupling"]

    best = sorted(rows, key=score)[:10]
    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Top 10 by heuristic:")
    for row in best:
        print(row)


if __name__ == "__main__":
    main()

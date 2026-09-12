#!/usr/bin/env python
"""Extract the executable seven-link route from its historical search artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="runs/swingup7_fddp_full_hanging_ilqr_terminal100k_deferred_lqr1.json",
    )
    parser.add_argument(
        "--out",
        default="runs/swingup7_uniform/seven_link_release_controller.json",
    )
    args = parser.parse_args()
    source_path = Path(args.source)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    original_controller = source["controller"]
    original_search = source["search"]
    controller = {
        "schema_version": 1,
        "claim_status": "frozen_release_controller_component",
        "summary": (
            "Frozen Box-FDDP route and feedback gains used by the released "
            "settled-launch seven-link hybrid controller."
        ),
        "route_optimization_generated_at": source.get("generated_at"),
        "controller": {
            key: original_controller[key]
            for key in (
                "type",
                "controls",
                "feedback_gains",
                "lqr_scale",
                "horizon_steps",
                "horizon_seconds",
                "control_cost",
                "terminal_weight",
                "terminal_state_weight",
                "terminal_cart_weight",
                "terminal_cart_velocity_weight",
                "rail_soft_limit",
                "rail_weight",
                "rebuilt_initial_states",
            )
        },
        "search": {
            "nominal_coordinate_states": original_search["nominal_coordinate_states"],
            "is_feasible": original_search.get("is_feasible"),
            "iterations": original_search.get("iterations"),
            "wall_time_seconds": original_search.get("wall_time_seconds"),
            "cost": original_search.get("cost"),
            "terminal_lyapunov": original_search.get("terminal_lyapunov"),
        },
        "development_origin": {
            **file_metadata(source_path),
            "role": "historical optimization trace; not canonical evaluation evidence",
            "optimization_seed": source.get("seed"),
            "optimization_state_index": source.get("state_index"),
        },
    }
    dump_json(controller, Path(args.out))


if __name__ == "__main__":
    main()

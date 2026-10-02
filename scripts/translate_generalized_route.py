#!/usr/bin/env python
"""Translate a saved generalized route in the cart-position coordinate.

The cart-pole dynamics are invariant to a uniform cart translation.  Keeping
the translation in the saved nominal route is useful when refining a route on
the exact parked launch, because the optimizer then sees the same initial
state and route coordinates as the final evaluator.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp


ROOT = Path(__file__).resolve().parents[1]


def translate(payload: dict[str, Any], shift: float, cart_scale: float) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    controller = result.get("controller")
    search = result.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("route must contain controller and search objects")

    controls = np.asarray(controller.get("controls"), dtype=np.float64)
    states = np.asarray(search.get("nominal_coordinate_states"), dtype=np.float64)
    if controls.ndim != 1 or states.shape != (controls.size + 1, states.shape[1]):
        raise ValueError("route controls and nominal states have inconsistent shapes")
    if states.ndim != 2 or states.shape[1] < 1:
        raise ValueError("route nominal states must be a nonempty matrix")
    if not np.isfinite(shift) or not np.isfinite(cart_scale) or cart_scale <= 0.0:
        raise ValueError("shift must be finite and cart_scale must be positive")

    states[:, 0] += float(shift) / float(cart_scale)
    search["nominal_coordinate_states"] = states.astype(float).tolist()
    selected = result.get("selected_state")
    if isinstance(selected, dict) and isinstance(selected.get("qpos"), list):
        qpos = np.asarray(selected["qpos"], dtype=np.float64)
        if qpos.ndim != 1 or qpos.size < 1:
            raise ValueError("selected qpos must be a nonempty vector")
        qpos[0] += float(shift)
        selected["qpos"] = qpos.astype(float).tolist()

    result["claim_status"] = "development_translated_route_not_solution_evidence"
    result["not_solution"] = True
    result["translation"] = {
        "cart_shift_m": float(shift),
        "cart_position_scale_m": float(cart_scale),
        "nominal_coordinate_shift": float(shift / cart_scale),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cart-shift", type=float, required=True)
    parser.add_argument("--cart-scale", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if args.cart_scale is None:
        if args.config is None:
            raise ValueError("--config is required when --cart-scale is omitted")
        spec = load_config("benchmarks/p1_capture_envelope.yaml")
        cart_scale = float(spec["distribution"]["cart_position_abs_max"])
    else:
        cart_scale = float(args.cart_scale)
    result = translate(payload, float(args.cart_shift), cart_scale)
    result["source_controller"] = file_metadata(source_path)
    result["runtime"] = runtime_metadata()
    result["git"] = git_metadata(ROOT)
    result["generated_at"] = utc_timestamp()
    dump_json(result, args.out)
    print(
        f"wrote {args.out}: shift={args.cart_shift:.6f} m "
        f"coordinate_shift={args.cart_shift / cart_scale:.6f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate a saved handoff-state bank with exact target-plant LQR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv

try:
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg
    from search_swingup_capture import lqr_action, lqr_gain


ROOT = Path(__file__).resolve().parents[1]


def evaluate_state(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    progress: float,
    seconds: float,
    gain: np.ndarray,
    lqr_scale: float,
    seed: int,
) -> dict[str, Any]:
    episode_cfg = fixed_state_cfg(cfg, state, seconds)
    env = NLinkCartPoleEnv(episode_cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    final_info: dict[str, Any] = {}
    max_cart = 0.0
    try:
        for _ in range(min(env.max_steps, int(seconds / env.dt))):
            action = lqr_action(env, gain, scale=lqr_scale, cart_target=0.0)
            _, _, terminated, truncated, info = env.step([action])
            final_info = dict(info)
            max_cart = max(max_cart, abs(float(info["x"])))
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "source_time_seconds": state.get("source_time_seconds"),
        "source_row_index": state.get("source_row_index"),
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max_cart),
        "termination_reason": final_info.get("termination_reason"),
        "final_max_abs_angle": float(final_info.get("max_abs_angle", np.inf)),
        "final_hinge_velocity_rms": float(
            final_info.get("hinge_velocity_rms", np.inf)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.seconds, args.lqr_scale, args.lqr_control_cost) <= 0.0:
        raise ValueError("seconds and LQR scales must be positive")

    cfg = load_config(args.config)
    bank_path = Path(args.state_json)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    states = bank.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("state bank must contain a nonempty states list")
    first_cfg = fixed_state_cfg(cfg, states[0], args.seconds)
    gain = lqr_gain(
        first_cfg,
        progress=args.progress,
        fd_eps=1e-7,
        control_cost=args.lqr_control_cost,
    )
    results = [
        evaluate_state(
            cfg,
            state,
            progress=args.progress,
            seconds=args.seconds,
            gain=gain,
            lqr_scale=args.lqr_scale,
            seed=args.seed,
        )
        for state in states
    ]
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "canonical_11_capture_bank_only_not_solution",
        "not_solution": True,
        "summary": "Exact canonical-target LQR tail evaluation from real saved handoff states; this does not prove swing-up reachability.",
        "config": file_metadata(Path(args.config)),
        "state_bank": file_metadata(bank_path),
        "evaluation": {
            "progress": float(args.progress),
            "seconds": float(args.seconds),
            "lqr_scale": float(args.lqr_scale),
            "lqr_control_cost": float(args.lqr_control_cost),
            "state_count": int(len(states)),
            "success_count": int(sum(row["success"] for row in results)),
            "max_hold_seconds": float(
                max(row["max_upright_streak_seconds"] for row in results)
            ),
            "max_cart_excursion": float(
                max(row["max_cart_excursion"] for row in results)
            ),
        },
        "results": results,
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    dump_json(payload, Path(args.out))
    print(
        f"states={len(results)} success={payload['evaluation']['success_count']} "
        f"max_hold={payload['evaluation']['max_hold_seconds']:.3f}s "
        f"max_cart={payload['evaluation']['max_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

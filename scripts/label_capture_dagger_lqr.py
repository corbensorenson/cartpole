#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv


def evaluate_lqr_query(
    env: NLinkCartPoleEnv,
    switch_cfg: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    observation, _ = env.reset(
        options={"qpos": query["qpos"], "qvel": query["qvel"]},
    )
    del observation
    initial_action, _ = env._lqr_action_bias(switch_cfg)
    terminated = truncated = False
    episode_return = 0.0
    final_info: dict[str, Any] = {}
    while not (terminated or truncated):
        action, _ = env._lqr_action_bias(switch_cfg)
        _, reward, terminated, truncated, final_info = env.step([action])
        episode_return += float(reward)
    return {
        "success": bool(final_info.get("success", False)),
        "teacher_action": float(initial_action),
        "return": float(episode_return),
        "length": int(env.step_count),
        "termination_reason": final_info.get("termination_reason"),
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
        "time_to_capture": final_info.get("time_to_capture"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accept exact-LQR DAgger labels only after successful uninterrupted replay"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--queries", default="runs/p1_capture_dagger/round1_train_queries.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=79501)
    parser.add_argument("--out", default="runs/p1_capture_dagger/round1_lqr_labels.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    query_path = Path(args.queries)
    cfg = load_config(config_path)
    payload = json.loads(query_path.read_text(encoding="utf-8"))
    if payload.get("split") != "train":
        raise ValueError("DAgger labeling is restricted to training queries")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("DAgger query artifact must contain queries")
    if args.limit is not None:
        queries = queries[: max(0, int(args.limit))]
    switch_cfg = copy.deepcopy(cfg["env"]["action_lqr_switch"])
    pure_cfg = copy.deepcopy(cfg)
    pure_cfg["env"]["action_lqr_switch"]["enabled"] = False
    pure_cfg["env"]["action_lqr_residual"]["enabled"] = False
    pure_cfg["env"]["init_cart_noise"] = 0.0
    pure_cfg["env"]["init_cart_vel_noise"] = 0.0
    pure_cfg["env"]["init_angle_noise"] = 0.0
    pure_cfg["env"]["init_vel_noise"] = 0.0
    # Progress one disables reset scaling; the plant is uniform at every capture stage.
    env = NLinkCartPoleEnv(pure_cfg, progress=1.0, seed=args.seed)
    labels: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    try:
        for ordinal, query in enumerate(queries):
            result = evaluate_lqr_query(env, switch_cfg, query)
            row = {
                "ordinal": ordinal,
                "query_id": query["query_id"],
                "source_index": int(query["source_index"]),
                "source_step": int(query["step"]),
                **result,
            }
            evaluations.append(row)
            if result["success"]:
                labels.append(
                    {
                        "query_id": query["query_id"],
                        "source_index": int(query["source_index"]),
                        "source_step": int(query["step"]),
                        "observation": query["observation"],
                        "learner_action": float(query["learner_action"]),
                        "teacher_action": float(result["teacher_action"]),
                        "action_disagreement_abs": abs(
                            float(query["learner_action"])
                            - float(result["teacher_action"])
                        ),
                        "tier": "dagger_lqr",
                    }
                )
            print(
                f"query={ordinal + 1}/{len(queries)} source={query['source_index']} "
                f"step={query['step']} success={result['success']} "
                f"hold={result['max_upright_streak_seconds']:.3f}s",
                flush=True,
            )
    finally:
        env.close()

    result_payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Successful exact-LQR labels on learner-visited training states.",
        "split": "train",
        "round": int(payload["round"]),
        "progress": float(payload["progress"]),
        "query_count": len(queries),
        "successful_query_count": len(labels),
        "successful_query_rate": float(len(labels) / max(1, len(queries))),
        "successful_source_count": len({int(row["source_index"]) for row in labels}),
        "labels_sha256": data_sha256(labels),
        "labels": labels,
        "evaluations": evaluations,
        "evidence": {
            "config": file_metadata(config_path),
            "queries": file_metadata(query_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(result_payload, args.out)
    print(
        f"success={len(labels)}/{len(queries)} "
        f"source_count={result_payload['successful_source_count']} Wrote {args.out}"
    )


if __name__ == "__main__":
    main()

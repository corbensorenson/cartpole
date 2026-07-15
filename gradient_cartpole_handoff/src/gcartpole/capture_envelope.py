from __future__ import annotations

from typing import Any

import numpy as np

from .evidence import data_sha256


GENERATOR_VERSION = "gcartpole.capture_envelope:v1"


def _same_number(actual: Any, expected: Any) -> bool:
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return False


def validate_capture_config(cfg: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Reject evaluation configs that alter the frozen P1 plant or gate."""
    errors: list[str] = []
    env = cfg.get("env", {})
    plant = spec["plant"]
    direct_env_keys = (
        "n_links",
        "total_length",
        "total_mass",
        "cart_mass",
        "rail_limit",
        "force_limit",
        "timestep",
        "frame_skip",
        "episode_seconds",
        "cart_damping",
        "cart_frictionloss",
        "joint_armature",
        "link_radius",
    )
    for key in direct_env_keys:
        if key not in env or not _same_number(env.get(key), plant[key]):
            errors.append(f"env.{key} does not match frozen plant value {plant[key]!r}")

    morphology = cfg.get("morphology", {})
    for endpoint in ("start", "end"):
        values = morphology.get(endpoint, {})
        for key in ("total_damping", "total_frictionloss"):
            if key not in values or not _same_number(values.get(key), plant[key]):
                errors.append(f"morphology.{endpoint}.{key} does not match frozen plant value {plant[key]!r}")
        for key in ("alpha_length", "alpha_mass", "alpha_damping", "alpha_frictionloss"):
            if key not in values or not _same_number(values.get(key), 0.0):
                errors.append(f"morphology.{endpoint}.{key} must be 0.0 for the uniform plant")

    gate = spec["capture_gate"]
    gate_fields = {
        "success_upright_threshold": gate["upright_threshold"],
        "success_sustain_seconds": gate["sustain_seconds"],
    }
    for key, expected in gate_fields.items():
        if key not in env or not _same_number(env.get(key), expected):
            errors.append(f"env.{key} does not match frozen gate value {expected!r}")
    if env.get("terminate_abs_angle", "missing") is not None:
        errors.append("env.terminate_abs_angle must be null")
    return errors


def _sample_rms_ball(rng: np.random.Generator, count: int, dimensions: int, rms_max: float) -> np.ndarray:
    directions = rng.normal(size=(count, dimensions))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.random(count) ** (1.0 / dimensions)
    return directions * radii[:, None] * float(rms_max) * np.sqrt(dimensions)


def generate_capture_states(spec: dict[str, Any], split: str) -> dict[str, Any]:
    if spec.get("generator") != GENERATOR_VERSION:
        raise ValueError(f"unsupported capture envelope generator: {spec.get('generator')!r}")
    if split not in spec.get("splits", {}):
        raise ValueError(f"unknown capture envelope split: {split}")

    plant = spec["plant"]
    distribution = spec["distribution"]
    split_spec = spec["splits"][split]
    n = int(plant["n_links"])
    count = int(split_spec["count"])
    seed = int(split_spec["seed"])
    rng = np.random.default_rng(seed)

    cart_positions = rng.uniform(
        -float(distribution["cart_position_abs_max"]),
        float(distribution["cart_position_abs_max"]),
        size=count,
    )
    absolute_angles = rng.uniform(
        -float(distribution["absolute_link_angle_abs_max"]),
        float(distribution["absolute_link_angle_abs_max"]),
        size=(count, n),
    )
    relative_angles = np.empty_like(absolute_angles)
    relative_angles[:, 0] = absolute_angles[:, 0]
    relative_angles[:, 1:] = absolute_angles[:, 1:] - absolute_angles[:, :-1]
    cart_velocities = rng.uniform(
        -float(distribution["cart_velocity_abs_max"]),
        float(distribution["cart_velocity_abs_max"]),
        size=count,
    )
    hinge_velocities = _sample_rms_ball(
        rng,
        count,
        n,
        float(distribution["hinge_velocity_rms_max"]),
    )

    states = []
    for index in range(count):
        qpos = np.r_[cart_positions[index], relative_angles[index]].astype(float).tolist()
        qvel = np.r_[cart_velocities[index], hinge_velocities[index]].astype(float).tolist()
        state_hash = data_sha256({"qpos": qpos, "qvel": qvel})
        states.append(
            {
                "state_id": f"{split}-{index:06d}-{state_hash[:12]}",
                "qpos": qpos,
                "qvel": qvel,
                "absolute_angles": absolute_angles[index].astype(float).tolist(),
                "cart_velocity": float(cart_velocities[index]),
                "hinge_velocity_rms": float(np.sqrt(np.mean(hinge_velocities[index] ** 2))),
                "source": "synthetic_capture_envelope",
            }
        )
    return {
        "schema_version": int(spec["schema_version"]),
        "benchmark": str(spec["name"]),
        "generator": GENERATOR_VERSION,
        "spec_sha256": data_sha256(spec),
        "split": split,
        "seed": seed,
        "count": count,
        "plant": plant,
        "distribution": distribution,
        "states": states,
    }


def validate_capture_states(payload: dict[str, Any], spec: dict[str, Any], split: str) -> list[str]:
    errors: list[str] = []
    split_spec = spec["splits"][split]
    distribution = spec["distribution"]
    states = payload.get("states")
    if payload.get("generator") != GENERATOR_VERSION:
        errors.append("generator version does not match")
    if payload.get("spec_sha256") != data_sha256(spec):
        errors.append("spec hash does not match")
    if payload.get("split") != split or payload.get("seed") != int(split_spec["seed"]):
        errors.append("split identity or seed does not match")
    if not isinstance(states, list) or len(states) != int(split_spec["count"]):
        errors.append("state count does not match")
        return errors

    n = int(spec["plant"]["n_links"])
    identifiers = set()
    for index, state in enumerate(states):
        qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
        if qpos.shape != (n + 1,) or qvel.shape != (n + 1,):
            errors.append(f"state {index} has invalid qpos/qvel shape")
            continue
        absolute_angles = np.cumsum(qpos[1:])
        hinge_rms = float(np.sqrt(np.mean(qvel[1:] ** 2)))
        if abs(float(qpos[0])) > float(distribution["cart_position_abs_max"]) + 1e-12:
            errors.append(f"state {index} exceeds cart position bound")
        if float(np.max(np.abs(absolute_angles))) > float(distribution["absolute_link_angle_abs_max"]) + 1e-12:
            errors.append(f"state {index} exceeds absolute angle bound")
        if abs(float(qvel[0])) > float(distribution["cart_velocity_abs_max"]) + 1e-12:
            errors.append(f"state {index} exceeds cart velocity bound")
        if hinge_rms > float(distribution["hinge_velocity_rms_max"]) + 1e-12:
            errors.append(f"state {index} exceeds hinge RMS bound")
        state_id = state.get("state_id")
        if not state_id or state_id in identifiers:
            errors.append(f"state {index} has missing or duplicate state_id")
        identifiers.add(state_id)
    return errors

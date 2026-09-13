from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import data_sha256, file_sha256, text_sha256
from .mjxml import generate_nlink_cartpole_xml
from .morphology import build_morphology


CANONICAL_ENV = {
    "n_links": 7,
    "total_length": 3.0,
    "total_mass": 1.0,
    "cart_mass": 1.0,
    "rail_limit": 3.0,
    "force_limit": 80.0,
    "timestep": 0.005,
    "frame_skip": 4,
    "episode_seconds": 30.0,
    "init_mode": "hanging",
    "init_cart_noise": 0.0,
    "init_cart_vel_noise": 0.05,
    "init_angle_noise": 0.05,
    "init_vel_noise": 0.05,
    "terminate_abs_angle": None,
    "success_upright_threshold": 0.15,
    "success_sustain_seconds": 5.0,
    "cart_damping": 0.02,
    "cart_frictionloss": 0.0,
    "joint_armature": 0.0005,
    "obs_include_morphology": True,
    "obs_include_frictionloss": False,
    "obs_include_time": False,
}

CANONICAL_MORPHOLOGY = {
    "alpha_length": 0.0,
    "alpha_mass": 0.0,
    "alpha_damping": 0.0,
    "alpha_frictionloss": 0.0,
    "total_damping": 0.015,
    "total_frictionloss": 0.0,
}

EPISODE_EVIDENCE_FIELDS = {
    "seed",
    "return",
    "termination_reason",
    "time_to_first_upright",
    "time_to_capture",
    "max_upright_streak_seconds",
    "final_upright_streak_seconds",
    "max_cart_excursion",
    "success",
}


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_canonical_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    env = cfg.get("env", {})
    morphology = cfg.get("morphology", {})

    for key, expected in CANONICAL_ENV.items():
        if key not in env:
            errors.append(f"env.{key} is missing")
        elif not _same(env[key], expected):
            errors.append(f"env.{key}={env[key]!r}; expected {expected!r}")

    forbidden = {
        "plant_progress",
        "rail_limit_start",
        "rail_limit_end",
        "init_states",
        "init_states_path",
        "init_qpos",
        "init_qvel",
        "action_lqr_residual",
    }
    for key in sorted(forbidden.intersection(env)):
        errors.append(f"env.{key} is not permitted in the canonical evaluation config")

    for base in ("init_cart_noise", "init_cart_vel_noise", "init_angle_noise", "init_vel_noise"):
        for suffix in ("_start", "_end"):
            if f"{base}{suffix}" in env:
                errors.append(f"env.{base}{suffix} is a curriculum-only setting")

    if morphology.get("schedule_mode") != "all_linear":
        errors.append("morphology.schedule_mode must be 'all_linear'")
    for endpoint in ("start", "end"):
        values = morphology.get(endpoint, {})
        for key, expected in CANONICAL_MORPHOLOGY.items():
            if key not in values:
                errors.append(f"morphology.{endpoint}.{key} is missing")
            elif not _same(values[key], expected):
                errors.append(
                    f"morphology.{endpoint}.{key}={values[key]!r}; expected {expected!r}"
                )

    if not isinstance(cfg.get("ppo", {}).get("hidden_sizes"), list):
        errors.append("ppo.hidden_sizes must declare the policy architecture")
    return errors


def benchmark_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = cfg["env"]
    morphology = build_morphology(env_cfg, cfg["morphology"], progress=1.0)
    xml = generate_nlink_cartpole_xml(
        morphology,
        cart_mass=float(env_cfg["cart_mass"]),
        rail_limit=float(env_cfg["rail_limit"]),
        force_limit=float(env_cfg["force_limit"]),
        timestep=float(env_cfg["timestep"]),
        cart_damping=float(env_cfg.get("cart_damping", 0.0)),
        cart_frictionloss=float(env_cfg.get("cart_frictionloss", 0.0)),
        joint_armature=float(env_cfg.get("joint_armature", 0.0)),
        link_radius=float(env_cfg.get("link_radius", 0.025)),
    )
    dt = float(env_cfg["timestep"]) * int(env_cfg["frame_skip"])
    obs_dim = 2 + 4 * int(env_cfg["n_links"])
    if bool(env_cfg.get("obs_include_morphology", True)):
        obs_dim += 3 * int(env_cfg["n_links"])
    if bool(env_cfg.get("obs_include_frictionloss", False)):
        obs_dim += int(env_cfg["n_links"])
    if bool(env_cfg.get("obs_include_absolute_velocity", False)):
        obs_dim += int(env_cfg["n_links"])
    if bool(env_cfg.get("obs_include_time", False)):
        obs_dim += 1 + 2 * len(env_cfg.get("obs_time_frequencies", []))
    return {
        "config_sha256": data_sha256(cfg),
        "generated_xml_sha256": text_sha256(xml),
        "observation_dim": obs_dim,
        "action_dim": 1,
        "action_low": -1.0,
        "action_high": 1.0,
        "action_frequency_hz": 1.0 / dt,
        "policy_dt": dt,
        "max_steps": int(float(env_cfg["episode_seconds"]) / dt),
        "lengths": morphology.lengths.astype(float).tolist(),
        "masses": morphology.masses.astype(float).tolist(),
        "damping": morphology.damping.astype(float).tolist(),
        "frictionloss": morphology.frictionloss.astype(float).tolist(),
    }


def validate_runtime_benchmark(cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from .env import NLinkCartPoleEnv, wrap_angle

    errors: list[str] = []
    snapshot = benchmark_snapshot(cfg)
    zero_cfg = copy.deepcopy(cfg)
    zero_cfg["env"].update(
        init_cart_noise=0.0,
        init_cart_vel_noise=0.0,
        init_angle_noise=0.0,
        init_vel_noise=0.0,
    )
    env = NLinkCartPoleEnv(zero_cfg, progress=1.0, seed=0)
    try:
        obs, info = env.reset(seed=0)
        if obs.shape != (51,):
            errors.append(f"runtime observation shape is {obs.shape}; expected (51,)")
        if env.action_space.shape != (1,):
            errors.append(f"runtime action shape is {env.action_space.shape}; expected (1,)")
        if not np.allclose(env.action_space.low, [-1.0]) or not np.allclose(env.action_space.high, [1.0]):
            errors.append("runtime action bounds are not [-1, 1]")
        expected_qpos = np.zeros(8, dtype=np.float64)
        expected_qpos[1] = np.pi
        if not np.allclose(env.data.qpos, expected_qpos, atol=1e-12):
            errors.append(f"zero-noise hanging qpos is {env.data.qpos.tolist()}")
        if not np.allclose(env.data.qvel, 0.0, atol=1e-12):
            errors.append("zero-noise hanging qvel is not zero")
        _, absolute_angles = env._angles()
        if not np.allclose(np.abs(wrap_angle(absolute_angles)), np.pi, atol=1e-12):
            errors.append("zero-noise chain does not hang straight below the cart")
        env.step(np.asarray([0.5], dtype=np.float32))
        if not math.isclose(float(env.data.ctrl[0]), 40.0, rel_tol=0.0, abs_tol=1e-6):
            errors.append(f"normalized action 0.5 applied {env.data.ctrl[0]} N; expected 40 N")
        if not math.isclose(float(env.rail_limit), 3.0, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"runtime rail is +/-{env.rail_limit} m; expected +/-3 m")
        if not math.isclose(float(env.dt), 0.02, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"runtime policy dt is {env.dt}; expected 0.02")
        if not math.isclose(float(env.model.opt.timestep), 0.005, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"MuJoCo timestep is {env.model.opt.timestep}; expected 0.005")
        snapshot["initial_info"] = info
    finally:
        env.close()

    noisy_env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    try:
        qpos_samples = []
        qvel_samples = []
        for seed in range(128):
            noisy_env.reset(seed=seed)
            qpos_samples.append(np.asarray(noisy_env.data.qpos, dtype=np.float64).copy())
            qvel_samples.append(np.asarray(noisy_env.data.qvel, dtype=np.float64).copy())
        qpos_samples = np.asarray(qpos_samples)
        qvel_samples = np.asarray(qvel_samples)
        if not np.allclose(qpos_samples[:, 0], 0.0, atol=1e-12):
            errors.append("canonical reset adds noise to cart position")
        angle_std = float(np.std(qpos_samples[:, 1:] - np.asarray([np.pi, 0, 0, 0, 0, 0, 0])))
        velocity_std = float(np.std(qvel_samples))
        if not 0.04 <= angle_std <= 0.06:
            errors.append(f"observed reset angle std is {angle_std:.4f}; expected approximately 0.05")
        if not 0.04 <= velocity_std <= 0.06:
            errors.append(f"observed reset velocity std is {velocity_std:.4f}; expected approximately 0.05")
        snapshot["sampled_reset_angle_std"] = angle_std
        snapshot["sampled_reset_velocity_std"] = velocity_std
    finally:
        noisy_env.close()
    return errors, snapshot


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing required artifact: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path} must contain a JSON object")
        return None
    return payload


def validate_evaluation(
    payload: dict[str, Any],
    *,
    required_episodes: int,
    minimum_success_rate: float,
    config_sha256: str,
    xml_sha256: str,
) -> list[str]:
    errors: list[str] = []
    label = f"{required_episodes}-episode evaluation"
    if int(payload.get("episodes", -1)) != required_episodes:
        errors.append(f"{label} declares {payload.get('episodes')} episodes; expected exactly {required_episodes}")
    if payload.get("claim_status") != "canonical_noisy_gate_evidence":
        errors.append(f"{label} has the wrong claim_status")
    if payload.get("not_claim") is True:
        errors.append(f"{label} is marked not_claim")
    if float(payload.get("success_rate", -1.0)) < minimum_success_rate:
        errors.append(f"{label} success_rate is below {minimum_success_rate:.2f}")
    episodes = payload.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) < required_episodes:
        errors.append(f"{label} lacks complete per-episode results")
        return errors
    seeds = []
    for index, episode in enumerate(episodes[:required_episodes]):
        missing = EPISODE_EVIDENCE_FIELDS.difference(episode)
        if missing:
            errors.append(f"{label} episode {index} is missing {sorted(missing)}")
        if "seed" in episode:
            seeds.append(episode["seed"])
    if len(set(seeds)) != len(seeds):
        errors.append(f"{label} episode seeds are not unique")
    seed_start = payload.get("seed_start")
    try:
        expected_seeds = list(range(int(seed_start), int(seed_start) + required_episodes))
    except (TypeError, ValueError):
        expected_seeds = []
    if seeds and seeds != expected_seeds:
        errors.append(f"{label} seeds are not the declared contiguous cohort")
    evidence = payload.get("evidence", {})
    if evidence.get("deterministic_policy") is not True:
        errors.append(f"{label} is not marked deterministic")
    if evidence.get("progress") != 1.0 or evidence.get("plant_progress") != 1.0:
        errors.append(f"{label} was not run at final progress")
    if evidence.get("config", {}).get("resolved_sha256") != config_sha256:
        errors.append(f"{label} resolved config hash does not match")
    if evidence.get("config", {}).get("overrides"):
        errors.append(f"{label} used command-line config overrides")
    if evidence.get("generated_xml_sha256") != xml_sha256:
        errors.append(f"{label} generated XML hash does not match")
    env = evidence.get("environment", {})
    for key, expected in {
        "n_links": 7,
        "init_mode": "hanging",
        "force_limit": 80.0,
        "rail_limit": 3.0,
        "observation_dim": 51,
        "action_dim": 1,
        "action_frequency_hz": 50.0,
    }.items():
        if not _same(env.get(key), expected):
            errors.append(f"{label} environment.{key} does not match the benchmark")
    git = evidence.get("git", {})
    if git.get("available") is not True or not git.get("commit") or git.get("dirty") is not False:
        errors.append(f"{label} does not point to a clean git commit")
    return errors


def validate_solution_artifacts(cfg: dict[str, Any], run_dir: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []
    snapshot = benchmark_snapshot(cfg)
    manifest_path = run_dir / "seven_link_swingup_manifest.json"
    manifest = _load_json(manifest_path, errors)
    if manifest is not None:
        if int(manifest.get("schema_version", -1)) != 2:
            errors.append("release manifest schema_version must be 2")
        if manifest.get("claim_status") != "released_canonical_noisy_swingup_and_hold":
            errors.append("release manifest has the wrong claim_status")
        if manifest.get("architecture") != "settled_launch_hybrid":
            errors.append("release manifest architecture must be settled_launch_hybrid")
        benchmark = manifest.get("benchmark", {})
        if benchmark.get("config_sha256") != snapshot["config_sha256"]:
            errors.append("release manifest config hash does not match")
        if benchmark.get("generated_xml_sha256") != snapshot["generated_xml_sha256"]:
            errors.append("release manifest generated XML hash does not match")
        controller = manifest.get("controller", {})
        controller_rel = Path(str(controller.get("path", "")))
        controller_path = repo_root / controller_rel
        if controller_rel.is_absolute() or ".." in controller_rel.parts or not controller_path.is_file():
            errors.append(f"release manifest controller path is invalid: {controller_rel}")
        elif controller.get("sha256") != file_sha256(controller_path):
            errors.append("release manifest controller hash does not match")
        if int(controller.get("route_steps", -1)) != 228:
            errors.append("release manifest route_steps must be 228")
        experts = manifest.get("experts", {})
        expected_experts = {
            "conditioning": "hanging_equilibrium_lqr",
            "swing": "box_fddp_time_varying_feedback",
            "capture": "upright_lqr",
        }
        for name, expected_type in expected_experts.items():
            if experts.get(name, {}).get("type") != expected_type:
                errors.append(f"release manifest expert {name} has the wrong type")
        switch = manifest.get("switch", {})
        if switch.get("state_reset_at_phase_boundaries") is not False:
            errors.append("release manifest must forbid state resets at phase boundaries")

    evaluations: list[dict[str, Any]] = []
    for count, rate in ((20, 0.80), (100, 0.90)):
        path = run_dir / f"eval_swingup7_{count}.json"
        payload = _load_json(path, errors)
        if payload is not None:
            evaluations.append(payload)
            errors.extend(
                validate_evaluation(
                    payload,
                    required_episodes=count,
                    minimum_success_rate=rate,
                    config_sha256=snapshot["config_sha256"],
                    xml_sha256=snapshot["generated_xml_sha256"],
                )
            )
            evidence = payload.get("evidence", {})
            manifest_sha = evidence.get("policy_manifest", {}).get("sha256")
            controller_sha = evidence.get("controller", {}).get("sha256")
            if manifest is not None and manifest_sha != file_sha256(manifest_path):
                errors.append(f"{count}-episode evaluation manifest hash does not match")
            if manifest is not None and controller_sha != manifest.get("controller", {}).get("sha256"):
                errors.append(f"{count}-episode evaluation controller hash does not match")

    if len(evaluations) == 2:
        first_seeds = {episode.get("seed") for episode in evaluations[0].get("episode_results", [])}
        second_seeds = {episode.get("seed") for episode in evaluations[1].get("episode_results", [])}
        if first_seeds.intersection(second_seeds):
            errors.append("20- and 100-episode evaluation cohorts overlap")
        if manifest is not None:
            declared = manifest.get("evaluation", {})
            if evaluations[0].get("seed_start") != declared.get("twenty_seed_start"):
                errors.append("20-episode seed does not match release manifest")
            if evaluations[1].get("seed_start") != declared.get("hundred_seed_start"):
                errors.append("100-episode seed does not match release manifest")

    video_path = run_dir / "seven_link_swingup_success.mp4"
    if not video_path.is_file() or video_path.stat().st_size == 0:
        errors.append(f"missing non-empty video: {video_path}")
    video_meta = _load_json(run_dir / "seven_link_swingup_success.video.json", errors)
    if video_meta is not None:
        render = video_meta.get("render", {})
        if int(render.get("reset_count", -1)) != 0:
            errors.append("video metadata reset_count is not zero")
        if render.get("completed_requested_steps") is not True or float(render.get("simulated_seconds", 0.0)) < 30.0:
            errors.append("video metadata does not prove a complete 30-second episode")
        events = render.get("done_events", [])
        if len(events) != 1 or any(event.get("terminated") or not event.get("success") for event in events):
            errors.append("video contains a failure termination")
        elif events[0].get("termination_reason") != "time_limit":
            errors.append("video does not end at the successful episode time limit")
        final_info = render.get("final_info", {})
        if final_info.get("success") is not True or float(final_info.get("upright_streak_seconds", 0.0)) < 5.0:
            errors.append("video final state does not prove the required upright hold")
        eval_seeds = {
            episode.get("seed")
            for payload in evaluations
            for episode in payload.get("episode_results", [])
        }
        if render.get("seed") in eval_seeds:
            errors.append("video seed overlaps an evaluation cohort")
        if manifest is not None and render.get("seed") != manifest.get("evaluation", {}).get("video_seed"):
            errors.append("video seed does not match release manifest")
        if video_meta.get("generated_xml_sha256") != snapshot["generated_xml_sha256"]:
            errors.append("video generated XML hash does not match")
        if video_meta.get("config", {}).get("resolved_sha256") != snapshot["config_sha256"]:
            errors.append("video resolved config hash does not match")
        if video_meta.get("config", {}).get("overrides"):
            errors.append("video used command-line config overrides")
        video_git = video_meta.get("git", {})
        if video_git.get("available") is not True or not video_git.get("commit") or video_git.get("dirty") is not False:
            errors.append("video metadata does not point to a clean git commit")
        if video_path.is_file() and video_meta.get("video", {}).get("sha256") != file_sha256(video_path):
            errors.append("video SHA-256 does not match metadata")

    robustness = _load_json(run_dir / "robustness_sweep.json", errors)
    if robustness is not None:
        if robustness.get("claim_status") != "noncanonical_robustness_characterization":
            errors.append("robustness sweep must be labelled noncanonical")
        if manifest is not None and robustness.get("controller", {}).get("sha256") != manifest.get("controller", {}).get("sha256"):
            errors.append("robustness sweep controller hash does not match")
        robustness_git = robustness.get("git", {})
        if robustness_git.get("available") is not True or not robustness_git.get("commit") or robustness_git.get("dirty") is not False:
            errors.append("robustness sweep does not point to a clean git commit")
        required_scenarios = {
            "initial_noise_sigma_0p10",
            "initial_noise_sigma_0p20",
            "force_limit_40N",
            "force_limit_20N",
            "total_mass_minus_5pct",
            "total_mass_plus_5pct",
            "total_length_minus_5pct",
            "total_length_plus_5pct",
            "joint_damping_half",
            "joint_damping_double",
            "sensor_noise_std_0p01",
            "control_delay_20ms",
            "control_delay_40ms",
            "settling_8s",
            "settling_6s",
        }
        scenarios = robustness.get("scenarios", [])
        names = {scenario.get("name") for scenario in scenarios if isinstance(scenario, dict)}
        if not required_scenarios.issubset(names):
            errors.append("robustness sweep is missing required scenarios")
        if any(int(scenario.get("episodes", 0)) < 20 for scenario in scenarios if isinstance(scenario, dict)):
            errors.append("robustness sweep scenarios must contain at least 20 episodes")

    checksums_path = run_dir / "SHA256SUMS"
    if not checksums_path.is_file():
        errors.append(f"missing required artifact: {checksums_path}")
    else:
        listed: set[str] = set()
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"malformed SHA256SUMS line: {line!r}")
                continue
            expected, name = parts[0], parts[1].lstrip("*")
            listed.add(name)
            path = run_dir / name
            if not path.is_file() or file_sha256(path) != expected:
                errors.append(f"SHA256SUMS mismatch or missing file: {name}")
        for name in {
            "seven_link_swingup_manifest.json",
            "seven_link_release_controller.json",
            "eval_swingup7_20.json",
            "eval_swingup7_100.json",
            "seven_link_swingup_success.mp4",
            "seven_link_swingup_success.video.json",
            "robustness_sweep.json",
        }:
            if name not in listed:
                errors.append(f"SHA256SUMS does not list {name}")
    return errors

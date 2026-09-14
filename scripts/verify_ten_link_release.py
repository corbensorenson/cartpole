#!/usr/bin/env python
"""Verify the released internal ten-link evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gcartpole.config import load_config
from gcartpole.evidence import file_sha256
from gcartpole.roadmap import benchmark_snapshot


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing JSON artifact: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"JSON artifact is not an object: {path}")
        return None
    return payload


def _path(root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    relative = Path(str(value or ""))
    candidate = root / relative
    if relative.is_absolute() or ".." in relative.parts or not candidate.is_file():
        errors.append(f"{label} path is invalid: {relative}")
        return None
    return candidate


def _file_metadata(
    root: Path, metadata: dict[str, Any], label: str, errors: list[str]
) -> Path | None:
    path = _path(root, metadata.get("path"), label, errors)
    if path is None:
        return None
    actual_sha = file_sha256(path)
    if metadata.get("sha256") != actual_sha:
        errors.append(f"{label} SHA-256 does not match")
    if "bytes" in metadata and int(metadata["bytes"]) != path.stat().st_size:
        errors.append(f"{label} byte count does not match")
    return path


def _check_gate(
    root: Path,
    gate_meta: dict[str, Any],
    *,
    expected_episodes: int,
    minimum_success_rate: float,
    expected_zero_noise: bool,
    expected_config_sha256: str,
    expected_runtime_config_sha256: str,
    expected_controller_sha256: str,
    expected_xml_sha256: str,
    expected_n_links: int,
    errors: list[str],
) -> dict[str, Any] | None:
    label = f"{expected_episodes}-episode gate"
    path = _file_metadata(root, gate_meta, label, errors)
    if path is None:
        return None
    payload = _load(path, errors)
    if payload is None:
        return None
    if payload.get("claim_status") != "canonical_parked_route_gate_evidence":
        errors.append(f"{label} has the wrong claim_status")
    if payload.get("not_solution") is not False:
        errors.append(f"{label} is marked as not_solution")
    if int(payload.get("episodes", -1)) != expected_episodes:
        errors.append(f"{label} episode count is wrong")
    if float(payload.get("success_rate", -1.0)) < minimum_success_rate:
        errors.append(f"{label} success rate is below the gate")
    if int(payload.get("successes", -1)) != expected_episodes:
        errors.append(f"{label} does not report all episodes as successful")
    if bool(payload.get("zero_noise")) is not expected_zero_noise:
        errors.append(f"{label} zero_noise flag is wrong")
    if int(payload.get("n_links", -1)) != expected_n_links:
        errors.append(f"{label} link count is wrong")
    if float(payload.get("park_seconds", -1.0)) != 16.0:
        errors.append(f"{label} park duration is not 16 seconds")
    if float(payload.get("cart_target", 1.0)) != -0.05:
        errors.append(f"{label} cart target is not -0.05 m")
    if payload.get("controller_sha256") != expected_controller_sha256:
        errors.append(f"{label} controller hash does not match the manifest")
    if payload.get("generated_xml_sha256") != expected_xml_sha256:
        errors.append(f"{label} generated XML hash does not match the manifest")
    if payload.get("resolved_config_sha256") != expected_runtime_config_sha256:
        errors.append(f"{label} resolved config hash does not match the manifest")
    config = payload.get("config", {})
    if config.get("sha256") != expected_config_sha256:
        errors.append(f"{label} source config hash does not match the manifest")
    environment = payload.get("environment", {})
    for key, expected in {
        "n_links": expected_n_links,
        "init_mode": "hanging",
        "force_limit": 80.0,
        "rail_limit": 3.0,
        "action_dim": 1,
        "action_frequency_hz": 50.0,
        "episode_seconds": 30.0,
    }.items():
        if environment.get(key) != expected:
            errors.append(f"{label} environment.{key} does not match")
    episodes = payload.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        errors.append(f"{label} lacks exactly {expected_episodes} episode results")
        return payload
    seed_start = int(payload.get("seed_start", -1))
    seeds = [int(episode.get("seed", -1)) for episode in episodes]
    if seeds != list(range(seed_start, seed_start + expected_episodes)):
        errors.append(f"{label} seeds are not a contiguous declared cohort")
    for index, episode in enumerate(episodes):
        if episode.get("success") is not True:
            errors.append(f"{label} episode {index} is not successful")
        if episode.get("termination_reason") != "time_limit":
            errors.append(f"{label} episode {index} did not reach the time limit")
        if float(episode.get("max_cart_excursion", 99.0)) >= 3.0:
            errors.append(f"{label} episode {index} exceeded the canonical rail")
    return payload


def verify_manifest(manifest_path: Path, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    manifest = _load(manifest_path, errors)
    if manifest is None:
        return errors
    if manifest.get("claim_status") != "released_internal_canonical_ten_link":
        errors.append("manifest has the wrong ten-link claim_status")
    if int(manifest.get("schema_version", -1)) != 1:
        errors.append("manifest schema_version must be 1")

    benchmark = manifest.get("benchmark", {})
    config_path = _path(root, benchmark.get("config"), "benchmark config", errors)
    if config_path is None:
        return errors
    config_sha256 = file_sha256(config_path)
    if benchmark.get("config_sha256") != config_sha256:
        errors.append("manifest config hash does not match the file")
    try:
        cfg = load_config(config_path)
        snapshot = benchmark_snapshot(cfg)
    except Exception as exc:  # pragma: no cover - reports malformed release input
        errors.append(f"cannot load benchmark config: {exc}")
        return errors
    expected_n_links = int(cfg["env"]["n_links"])
    if int(benchmark.get("n_links", -1)) != expected_n_links:
        errors.append("manifest link count does not match the config")
    if benchmark.get("generated_xml_sha256") != snapshot["generated_xml_sha256"]:
        errors.append("manifest generated XML hash does not match the config")
    for key, expected in {
        "total_link_length_m": 3.0,
        "total_link_mass_kg": 1.0,
        "cart_mass_kg": 1.0,
        "rail_limit_m": 3.0,
        "force_limit_n": 80.0,
        "policy_rate_hz": 50.0,
        "episode_seconds": 30.0,
        "upright_threshold_rad": 0.15,
        "success_sustain_seconds": 5.0,
    }.items():
        if benchmark.get(key) != expected:
            errors.append(f"manifest benchmark.{key} does not match the contract")
    if benchmark.get("initial_mode") != "hanging":
        errors.append("manifest benchmark.initial_mode is not hanging")

    controller = manifest.get("controller", {})
    controller_path = _file_metadata(root, controller, "controller", errors)
    controller_sha256 = file_sha256(controller_path) if controller_path else ""
    if controller_path and controller.get("sha256") != controller_sha256:
        errors.append("manifest controller hash does not match the file")
    if int(controller.get("route_steps", -1)) != 400:
        errors.append("manifest route_steps must be 400")
    if float(controller.get("route_seconds", -1.0)) != 8.0:
        errors.append("manifest route_seconds must be 8")

    experts = manifest.get("experts", {})
    expected_experts = {
        "conditioning": "hanging_equilibrium_lqr_cart_park",
        "swing": "box_fddp_time_varying_feedback",
        "capture": "upright_lqr",
    }
    for name, expected in expected_experts.items():
        if experts.get(name, {}).get("type") != expected:
            errors.append(f"manifest expert {name} has the wrong type")
    if manifest.get("switch", {}).get("state_reset_at_phase_boundaries") is not False:
        errors.append("manifest must forbid state resets at phase boundaries")

    evaluation = manifest.get("evaluation", {})
    noisy_runtime_sha256 = str(benchmark.get("resolved_noisy_config_sha256", ""))
    exact_runtime_sha256 = str(benchmark.get("resolved_exact_config_sha256", ""))
    gates = []
    for key, count, rate, zero_noise, runtime_sha in (
        ("twenty_episode", 20, 0.80, False, noisy_runtime_sha256),
        ("hundred_episode", 100, 0.90, False, noisy_runtime_sha256),
        ("exact_twenty_episode", 20, 1.0, True, exact_runtime_sha256),
    ):
        payload = _check_gate(
            root,
            evaluation.get(key, {}),
            expected_episodes=count,
            minimum_success_rate=rate,
            expected_zero_noise=zero_noise,
            expected_config_sha256=config_sha256,
            expected_runtime_config_sha256=runtime_sha,
            expected_controller_sha256=controller_sha256,
            expected_xml_sha256=str(benchmark.get("generated_xml_sha256", "")),
            expected_n_links=expected_n_links,
            errors=errors,
        )
        if payload is not None:
            gates.append(payload)

    if len(gates) == 3:
        cohorts = [
            {int(row.get("seed", -1)) for row in payload.get("episode_results", [])}
            for payload in gates
        ]
        if cohorts[0] & cohorts[1] or cohorts[0] & cohorts[2] or cohorts[1] & cohorts[2]:
            errors.append("evaluation cohorts overlap")

    video_meta = evaluation.get("video_metadata", {})
    video_meta_path = _file_metadata(root, video_meta, "video metadata", errors)
    video = evaluation.get("video", {})
    video_path = _file_metadata(root, video, "video", errors)
    if video_path and video_meta_path:
        metadata = _load(video_meta_path, errors)
        if metadata is not None:
            if metadata.get("generated_xml_sha256") != benchmark.get("generated_xml_sha256"):
                errors.append("video metadata XML hash does not match the manifest")
            if metadata.get("controller", {}).get("sha256") != controller_sha256:
                errors.append("video metadata controller hash does not match")
            render = metadata.get("render", {})
            if render.get("reset_count") != 0:
                errors.append("video metadata reset_count is not zero")
            if render.get("completed_requested_steps") is not True:
                errors.append("video metadata does not prove a complete replay")
            if render.get("frames") != 1500 or render.get("fps") != 50:
                errors.append("video metadata frame contract is wrong")
            if float(render.get("simulated_seconds", 0.0)) < 30.0:
                errors.append("video metadata is shorter than 30 seconds")
            final_info = render.get("final_info", {})
            if final_info.get("success") is not True:
                errors.append("video final state is not successful")
            events = render.get("done_events", [])
            if len(events) != 1 or events[0].get("termination_reason") != "time_limit":
                errors.append("video does not end at the successful time limit")
            if render.get("seed") != evaluation.get("video", {}).get("video_seed"):
                errors.append("video seed does not match the manifest")
            gate_seeds = {
                int(row.get("seed", -1))
                for payload in gates
                for row in payload.get("episode_results", [])
            }
            if render.get("seed") in gate_seeds:
                errors.append("video seed overlaps a gate cohort")
    for artifact_key in ("paper", "ledger", "checksums", "verifier"):
        if _path(root, manifest.get("release", {}).get(artifact_key), artifact_key, errors) is None:
            continue
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="runs/generalized_solver/ten_link_swingup_manifest.json",
    )
    args = parser.parse_args()
    manifest_path = _path(ROOT, args.manifest, "manifest", [])
    errors = verify_manifest(manifest_path, ROOT) if manifest_path else ["manifest path is invalid"]
    report = {
        "passed": not errors,
        "manifest": str(Path(args.manifest)),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

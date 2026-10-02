#!/usr/bin/env python
"""Verify a frozen adjacent-count release, including its actual evaluator trace."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from gcartpole.config import load_config
from gcartpole.evidence import data_sha256, file_sha256
from gcartpole.roadmap import benchmark_snapshot, validate_canonical_config
try:
    from scripts.evaluate_fddp_parked_route import zero_noise_config
    from scripts.render_parked_evaluation_trace import validate_trace
    from scripts.synthesize_inverse_increment import exact_candidate_passed
except ModuleNotFoundError:
    from evaluate_fddp_parked_route import zero_noise_config
    from render_parked_evaluation_trace import validate_trace
    from synthesize_inverse_increment import exact_candidate_passed


ROOT = Path(__file__).resolve().parents[1]
POLICY_FIELDS = ('park_seconds', 'cart_target', 'settle_control_cost',
                 'settle_cart_position_cost', 'settle_cart_velocity_cost',
                 'tracking_gain_scale', 'phase_adaptive', 'phase_window')


def check_episode(episode, cfg):
    env = cfg['env']
    steps = round(env['episode_seconds']/(env['timestep']*env['frame_skip']))
    metrics = [episode.get('max_cart_excursion', np.inf),
               episode.get('max_upright_streak_seconds', -np.inf)]
    final = episode.get('final_info', {})
    return bool(episode.get('success') is True and episode.get('full_episode_success') is True
                and episode.get('length') == steps
                and episode.get('termination_reason') == 'time_limit'
                and final.get('simulation_error') is None
                and np.all(np.isfinite(metrics)) and metrics[0] <= env['rail_limit']
                and metrics[1] >= env['success_sustain_seconds'])


def verify_manifest(path, root=ROOT):
    errors = []
    def require(condition, description):
        if not condition:
            errors.append(description)

    def read_file(metadata, label, *, json_file=True):
        relative = Path(metadata.get('path', ''))
        file = root/relative
        if relative.is_absolute() or '..' in relative.parts or not file.is_file():
            errors.append(f'{label}: invalid or missing repository-relative path')
            return None
        require(file_sha256(file) == metadata.get('sha256'), f'{label}: SHA-256 mismatch')
        require(file.stat().st_size == metadata.get('bytes'), f'{label}: byte count mismatch')
        return json.loads(file.read_text()) if json_file else file

    manifest = json.loads(Path(path).read_text())
    require(manifest.get('claim_status') == 'released_internal_canonical_frontier', 'wrong release claim')
    policy = read_file(manifest['policy'], 'policy')
    if policy is None:
        return errors
    config_path = read_file(policy['config'], 'config', json_file=False)
    source = read_file(policy['controller'], 'controller')
    if config_path is None or source is None:
        return errors
    cfg = load_config(config_path)
    count = cfg['env']['n_links']
    require(count == manifest['n_links'] == policy['n_links'], 'link count mismatch')
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg['env']['n_links'] = 7
    errors.extend(validate_canonical_config(reference_cfg))
    snapshot = benchmark_snapshot(cfg)
    require(snapshot['generated_xml_sha256'] == policy['generated_xml_sha256'], 'canonical geometry mismatch')
    require(exact_candidate_passed(source, cfg), 'source controller did not pass full exact-start calibration')
    for name in ('settle_gain', 'capture_gain'):
        gain = np.asarray(policy[name], dtype=float)
        require(gain.size == 2*(count+1) and np.all(np.isfinite(gain)), f'{name}: invalid gain')
        require(data_sha256(gain.tolist()) == policy[name+'_sha256'], f'{name}: saved gain hash mismatch')

    cohorts = []
    requested = {'gate20': (20, False), 'gate100': (100, False),
                 'exact20': (20, True), 'video_episode': (1, False)}
    episodes = {}
    for label, (size, exact) in requested.items():
        evidence = read_file(manifest['evaluation'][label], label)
        if evidence is None:
            continue
        episodes[label] = evidence
        require(evidence.get('claim_status') == 'canonical_parked_route_gate_evidence'
                and evidence.get('not_solution') is False and evidence.get('release_gate_passed') is True,
                f'{label}: evidence is not a passed gate')
        require(evidence.get('n_links') == count and evidence.get('episodes') == size
                and evidence.get('zero_noise') is exact, f'{label}: wrong count or evaluation distribution')
        require(evidence.get('full_episode_successes') == size
                and evidence.get('successes') == size, f'{label}: incomplete success cohort')
        require(evidence.get('controller_sha256') == policy['controller']['sha256'], f'{label}: wrong controller')
        require(evidence.get('config', {}).get('sha256') == policy['config']['sha256'], f'{label}: wrong config')
        require(evidence.get('generated_xml_sha256') == snapshot['generated_xml_sha256'], f'{label}: wrong plant')
        runtime_cfg = copy.deepcopy(cfg)
        runtime_cfg['env']['init_mode'] = 'hanging'
        runtime_cfg['env']['action_lqr_residual'] = {'enabled': False}
        runtime_cfg['env'].setdefault('action_lqr_switch', {'enabled': False})['enabled'] = False
        if exact:
            runtime_cfg = zero_noise_config(runtime_cfg)
        require(evidence.get('resolved_config_sha256') == data_sha256(runtime_cfg), f'{label}: wrong runtime configuration')
        for field in POLICY_FIELDS:
            require(evidence.get(field) == policy['parameters'][field], f'{label}: policy parameter {field} differs')
        require(evidence.get('settle_gain_sha256') == policy['settle_gain_sha256'], f'{label}: different hanging gain')
        require(evidence.get('capture_gain_sha256') == policy['capture_gain_sha256'], f'{label}: different capture gain')
        require(evidence.get('git', {}).get('commit') == manifest['reproduction']['source_commit']
                and evidence.get('git', {}).get('dirty') is False, f'{label}: source commit is not the clean frozen version')
        rows = evidence.get('episode_results', [])
        expected_seeds = list(range(policy['held_out_seeds'][label], policy['held_out_seeds'][label]+size))
        require([row.get('seed') for row in rows] == expected_seeds, f'{label}: wrong reserved seed cohort')
        require(len(rows) == size and all(check_episode(row, cfg) for row in rows), f'{label}: physical episode failure')
        cohorts.append(set(expected_seeds))
    for index, cohort in enumerate(cohorts):
        require(not any(cohort & other for other in cohorts[:index]), 'evaluation cohorts overlap')
    if 'video_episode' in episodes:
        try:
            validate_trace(episodes['video_episode'], 0,
                success_threshold=cfg['env']['success_upright_threshold'],
                required_hold=cfg['env']['success_sustain_seconds'])
        except (ValueError, KeyError, IndexError) as error:
            errors.append(f'video physical trace: {error}')
    video = read_file(manifest['video'], 'video', json_file=False)
    metadata = read_file(manifest['video_metadata'], 'video metadata')
    if metadata is not None and video is not None:
        require(metadata.get('video', {}).get('sha256') == file_sha256(video), 'video metadata references different pixels')
        require(metadata.get('source_evaluation', {}).get('sha256') == manifest['evaluation']['video_episode']['sha256'], 'video depicts a different episode')
        require(metadata.get('source_controller', {}).get('sha256') == policy['controller']['sha256'], 'video controller mismatch')
        render = metadata.get('render', {})
        require(render.get('frames') == snapshot['max_steps'] and render.get('fps') == 50
                and render.get('simulated_seconds') == 30. and render.get('reset_count') == 0
                and render.get('dynamics_reexecuted') is False and render.get('success') is True,
                'video cadence, duration or reset contract failed')
        for field in POLICY_FIELDS:
            require(metadata.get('executed_policy', {}).get(field) == policy['parameters'][field],
                    f'video policy parameter {field} differs')
    read_file(manifest['reproduction']['source_bundle'], 'clean source bundle', json_file=False)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    errors = verify_manifest(args.manifest)
    print(json.dumps(dict(passed=not errors, manifest=args.manifest, errors=errors), indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()

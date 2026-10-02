#!/usr/bin/env python
"""Freeze a validated development policy and clean source before reserved gates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, text_sha256, utc_timestamp
from gcartpole.generalized_energy import hanging_lqr_gain
from scripts.evaluate_fddp_two_expert import load_controller
from scripts.search_swingup_capture import lqr_gain
from scripts.synthesize_inverse_increment import exact_candidate_passed

WORDS = dict(zip(range(11, 21), 'eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()))



def validate_development_cohorts(cohorts, *, n_links, controller_sha256, config_sha256,
                                 parameters, capture_gain_sha256, settle_gain_sha256,
                                 generated_xml_sha256, reserved_seeds):
    """Require the development evidence to match the entire policy being frozen."""
    seen = set()
    for size, data in cohorts:
        if data['episodes'] != size or data['full_episode_successes'] != size or data['zero_noise']:
            raise ValueError('both development noisy cohorts must pass completely before freezing')
        if (data['controller_sha256'] != controller_sha256 or data['n_links'] != n_links
                or data['config']['sha256'] != config_sha256):
            raise ValueError('development cohorts must match source, count and config')
        for key, value in parameters.items():
            if data.get(key) != value:
                raise ValueError(f'development policy parameter mismatch: {key}')
        for key, value in dict(capture_gain_sha256=capture_gain_sha256,
                               settle_gain_sha256=settle_gain_sha256,
                               generated_xml_sha256=generated_xml_sha256).items():
            if data.get(key) != value:
                raise ValueError(f'development policy evidence mismatch: {key}')
        seeds = set(range(data['seed_start'], data['seed_start'] + size))
        if seen & seeds or seeds & reserved_seeds:
            raise ValueError('development cohorts must be disjoint and avoid reserved seeds')
        seen.update(seeds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-links', type=int, required=True, choices=range(11, 21))
    p.add_argument('--source', required=True)
    p.add_argument('--noisy20', required=True)
    p.add_argument('--noisy100', required=True)
    p.add_argument('--park-seconds', type=float, default=16.)
    p.add_argument('--cart-target', type=float, default=-.05)
    p.add_argument('--settle-cart-position-cost', type=float, default=10.)
    p.add_argument('--settle-cart-velocity-cost', type=float, default=5.)
    args = p.parse_args()
    root, n, word = Path.cwd(), args.n_links, WORDS[args.n_links]
    source, config_path = Path(args.source), Path(f'configs/swingup{n}_uniform.yaml')
    cfg = load_config(config_path)
    if cfg['env']['n_links'] != n or not exact_candidate_passed(json.loads(source.read_text()), cfg):
        raise ValueError('source must pass exact feasible full-episode calibration on the requested count')
    cohorts = [(size, json.loads(Path(path).read_text()))
               for size, path in ((20, args.noisy20), (100, args.noisy100))]
    controller = load_controller(source, n, load_config('benchmarks/p1_capture_envelope.yaml'))
    capture = lqr_gain(cfg, progress=1., fd_eps=1e-7, control_cost=controller['lqr_control_cost'],
                       q_weights=controller['lqr_weights'], decimal_digits=controller['lqr_decimal_digits']).reshape(-1)
    probe = NLinkCartPoleEnv(cfg, progress=1., seed=0)
    settle = hanging_lqr_gain(probe, control_cost=1000., cart_position_cost=args.settle_cart_position_cost,
                              cart_velocity_cost=args.settle_cart_velocity_cost)
    xml, dt = text_sha256(probe.xml), probe.dt
    probe.close()
    seed_base = 200000+1000*n
    parameters = dict(park_seconds=args.park_seconds, cart_target=args.cart_target,
        settle_control_cost=1000., settle_cart_position_cost=args.settle_cart_position_cost,
        settle_cart_velocity_cost=args.settle_cart_velocity_cost, tracking_gain_scale=1.,
        phase_adaptive=False, phase_window=12)
    reserved = set(range(seed_base, seed_base+20)) | set(range(seed_base+100, seed_base+200))
    reserved |= set(range(seed_base+200, seed_base+220)) | {seed_base+500}
    validate_development_cohorts(cohorts, n_links=n, controller_sha256=file_metadata(source)['sha256'],
        config_sha256=file_metadata(config_path)['sha256'], parameters=parameters,
        capture_gain_sha256=data_sha256(capture.tolist()), settle_gain_sha256=data_sha256(settle.tolist()),
        generated_xml_sha256=xml, reserved_seeds=reserved)
    release = Path(f'runs/swingup{n}_uniform')
    release.mkdir(parents=True, exist_ok=False)
    controller_path, policy_path = release/f'{word}_link_controller.json', release/f'{word}_link_policy.json'
    shutil.copy2(source, controller_path)
    dump_json(dict(schema_version=1, generated_at=utc_timestamp(),
        claim_status=f'frozen_{word}_link_policy_before_reserved_validation', not_solution=True,
        n_links=n, config=file_metadata(config_path), controller=file_metadata(controller_path),
        source_candidate=file_metadata(source), generated_xml_sha256=xml,
        parameters=parameters,
        settle_gain=settle.tolist(), settle_gain_sha256=data_sha256(settle.tolist()),
        capture_gain=capture.tolist(), capture_gain_sha256=data_sha256(capture.tolist()),
        coordinate_transform=controller['transform'].tolist(),
        switching=dict(defer_handoff_until_horizon=controller['defer_handoff_until_horizon'],
            route_seconds=len(controller['controls'])*dt, route_steps=len(controller['controls']),
            reset_at_phase_boundaries=False, continuous_angles=True,
            nominal_angle_branch_alignment='One integer 2*pi branch translation at launch; physical state is never overwritten.',
            cart_reference=f'Translate nominal cart reference to actual launch position; capture equilibrium {args.cart_target} m.'),
        held_out_seeds=dict(gate20=seed_base, gate100=seed_base+100, exact20=seed_base+200, video_episode=seed_base+500),
        development=dict(noisy20=file_metadata(Path(args.noisy20)), noisy100=file_metadata(Path(args.noisy100))),
        runtime=runtime_metadata(), source_working_tree=git_metadata(root),
        note='Frozen before reserved cohorts. Development tuning is disclosed; this is not release promotion.'), policy_path)
    campaign = root/'runs/frontier_campaign_20261001'
    repository = campaign/f'n{n}_release_source_repository'
    repository.mkdir(exist_ok=False)
    for folder in ('src', 'scripts', 'tests', 'configs', 'benchmarks', 'docs'):
        shutil.copytree(root/folder, repository/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('ROADMAP.md', 'README.md', 'pyproject.toml', 'requirements-mac.txt', 'environment-aligator.yml'):
        shutil.copy2(root/name, repository/name)
    # One immutable XML fixture supports the canonical geometry unit test.
    paths = (controller_path, policy_path, Path('runs/swingup7_uniform/model.xml'))
    for path in paths:
        target = repository/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/path, target)
    (repository/'.gitignore').write_text('__pycache__/\n*.pyc\n.pytest_cache/\n.DS_Store\n')
    def git(*command):
        return subprocess.check_output(['git', *command], cwd=repository, text=True).strip()
    git('init', '-b', f'codex/n{n}-validation-source')
    git('add', '.')
    git('-c', 'user.name=Cartpole Reproducibility', '-c', 'user.email=reproducibility@localhost',
        'commit', '-m', f'Freeze {word}-link evaluator and policy before reserved validation')
    commit = git('rev-parse', 'HEAD')
    bundle_path = release/f'{word}_link_source.bundle'
    git('bundle', 'create', str(root/bundle_path), '--all')
    clone = campaign/f'n{n}_release_fresh_clone'
    subprocess.run(['git', 'clone', str(root/bundle_path), str(clone)], check=True)
    dump_json(dict(schema_version=1, generated_at=utc_timestamp(), not_solution=True,
        source_commit=commit, source_bundle=file_metadata(bundle_path),
        repository=str(repository.relative_to(root)), fresh_clone=str(clone.relative_to(root)),
        parent_working_tree=git_metadata(root), runtime=runtime_metadata(),
        note='Isolated clean Git snapshot preserves the shared working tree. Reserved validation and publication remain pending.'),
        release/f'{word}_link_source.json')
    print('Frozen clean source commit:', commit)
    print('Fresh clone:', clone)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Package passed reserved gates and a state-faithful full video for one count."""
import argparse
import json
from pathlib import Path
from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, utc_timestamp
from scripts.freeze_frontier_release import WORDS
from scripts.verify_frontier_release import verify_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-links', type=int, required=True, choices=range(11, 21))
    p.add_argument('--focused-source-tests', type=int, required=True)
    p.add_argument('--working-tree-tests', type=int, required=True)
    p.add_argument('--method', required=True)
    args = p.parse_args()
    n, word = args.n_links, WORDS[args.n_links]
    release = Path(f'runs/swingup{n}_uniform')
    source = json.loads((release/f'{word}_link_source.json').read_text())
    files = dict(gate20=f'eval_swingup{n}_20.json', gate100=f'eval_swingup{n}_100.json',
                 exact20=f'eval_swingup{n}_exact20.json', video_episode=f'{word}_link_video_episode.json')
    metrics = {}
    for label, name in files.items():
        data = json.loads((release/name).read_text())
        if data.get('release_gate_passed') is not True:
            raise ValueError('Cannot package a failed gate: '+label)
        metrics[label] = dict(successes=data['full_episode_successes'], episodes=data['episodes'],
            seed_start=data['seed_start'], maximum_cart_excursion=data['max_cart_excursion_max'],
            minimum_hold=min(e['max_upright_streak_seconds'] for e in data['episode_results']),
            all_full_episode=True)
    manifest = release/f'{word}_link_swingup_manifest.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    dump_json(dict(schema_version=1, generated_at=utc_timestamp(),
        claim_status='released_internal_canonical_frontier', not_solution=False, n_links=n,
        policy=file_metadata(release/f'{word}_link_policy.json'),
        evaluation={label:file_metadata(release/name) for label,name in files.items()},
        video=file_metadata(release/f'{word}_link_swingup_success.mp4'),
        video_metadata=file_metadata(release/f'{word}_link_swingup_success.video.json'), metrics=metrics,
        reproduction=dict(source_commit=source['source_commit'], source_bundle=source['source_bundle'],
            fresh_clone=source['fresh_clone'], focused_source_tests_passed=args.focused_source_tests,
            full_working_tree_tests_passed=args.working_tree_tests,
            commands=file_metadata(release/'reserved_commands.json'), verifier='scripts/verify_frontier_release.py'),
        method=args.method,
        tuning_disclosure='Development data selected synthesis and handoff parameters. Reserved cohorts remained unused until the policy and clean source commit were frozen. Independent automatic re-synthesis and arbitrary morphology generalization remain separate research gates.',
        limitations=['Internal repository benchmark; no external world-record claim.',
            'Declared MuJoCo full-state initial-noise benchmark; hardware, sensor noise, delay and model mismatch are not evaluated.',
            'Linear Riccati and Lyapunov checks are not nonlinear capture certificates.']), manifest)
    errors = verify_manifest(manifest)
    if errors:
        raise ValueError('Manifest verification failed: '+repr(errors))
    print('Packaged and verified', manifest)


if __name__ == '__main__':
    main()

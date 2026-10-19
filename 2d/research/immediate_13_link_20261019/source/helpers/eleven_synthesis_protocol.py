"""Shared evidence gates; evaluation labels never enter the synthesis algorithm."""
import hashlib
import json
from pathlib import Path

STANDARD = [20261016]+list(range(20261116, 20261128))+list(range(20261416, 20261448))
RESERVED = list(range(20263516, 20263548))
HARNESSES = {'evaluate_eleven_observed_synthesis.py', 'eleven_synthesis_protocol.py'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seeds(cohort):
    return dict(pilot13=STANDARD[:13], development45=STANDARD, qualification32=RESERVED)[cohort]


def controller_identity(manifest):
    payload = dict(helper_sha256={k:v for k, v in manifest['helper_sha256'].items() if k not in HARNESSES},
        source_sha256=manifest['source_sha256'], bundle_sha256=manifest['bundle_sha256'],
        solver_receipt_sha256=manifest['solver_receipt_sha256'],
        max_models=manifest['max_models'], max_planning_native=manifest['max_planning_native'],
        max_case_native=manifest['max_case_native'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_parent(run, audit, next_cohort):
    run, audit = Path(run), Path(audit)
    expected = dict(development45='pilot13', qualification32='development45')[next_cohort]
    assert (run/'completed.json').exists() and (audit/'completed.json').exists()
    summary, proof, manifest = read(run/'summary.json'), read(audit/'summary.json'), read(run/'frozen_manifest.json')
    expected_seeds = seeds(expected)
    assert summary['cohort'] == expected and summary['cohort_finished'] and not summary['stopped_on_failure']
    assert summary['passes'] == len(summary['results']) == len(expected_seeds)
    assert [r['result']['seed'] for r in summary['results']] == expected_seeds
    assert proof['passed'] and proof['cohort'] == expected and proof['case_seeds'] == expected_seeds
    assert proof['source_summary_sha256'] == sha(run/'summary.json')
    assert proof['frozen_manifest_sha256'] == sha(run/'frozen_manifest.json')
    assert proof['controller_identity'] == controller_identity(manifest)
    return summary, proof, manifest

"""Reproduce the qualified hybrid from an isolated published source bundle."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('--bundle-root', default=str(Path(__file__).resolve().parent))
ap.add_argument('--seed', type=int, default=20330116)
ap.add_argument('--noise', type=float, default=.02)
ap.add_argument('--out', required=True)
a = ap.parse_args()
bundle = Path(a.bundle_root).resolve()
assert (bundle/'artifacts/policy.bin').is_file()
sys.path[:0] = [str(bundle/'src'), str(bundle/'scripts')]
from spatialpole import Config
from spatialpole.experiment import episode
from evaluate_puffer_hybrid_swing import HybridCapture

tape = Path(a.out).resolve()
assert not tape.exists()
tape.parent.mkdir(parents=True, exist_ok=True)
with np.load(bundle/'artifacts/route.npz') as z:
    payload = {k: z[k] for k in z.files}
row = episode(Config(n_links=1), lambda env: HybridCapture(env, payload, bundle/'artifacts/policy.bin'),
              a.seed, 'hanging', a.noise, tape)
print(json.dumps(row), flush=True)

# PufferLib 4 Notes

## Status

PufferLib 4.0 is available from the `4.0` branch in the upstream GitHub repo. The branch's `pyproject.toml` identifies package version `4.0.0` and Python `>=3.10`.

Install optionally:

```bash
pip install -r requirements-pufferlib4.txt
```

## Why PufferLib is optional here

This packet is Mac-first. The primary acceleration path is:

- MuJoCo CPU simulation in parallel workers.
- MLX policy/value updates on Apple Silicon/Metal.

PufferLib's most aggressive advertised speed path is CUDA-first. On a Mac, it may still help with environment compatibility/vectorization, but it should not be the first blocker.

## Suggested integration path

1. Keep `NLinkCartPoleEnv` Gymnasium-compatible.
2. Install PufferLib 4.0.
3. Check current wrapper API with:

```bash
python - <<'PY'
import pufferlib, inspect
print(pufferlib)
print(dir(pufferlib))
PY
```

4. Add a small adapter in `adapters/` once the local API is confirmed.
5. Benchmark wall-clock `steps/sec` against `ppo.vec_backend=subprocess`.

## Longer-term high-speed path

If this project becomes worth serious compute:

- Implement a custom analytic n-link simulator in C or C++.
- Integrate with PufferLib-style shared memory/vectorization.
- Or implement a differentiable/vectorized simulator in MLX for the simplified planar model.
- Use MuJoCo only as final validation and video renderer.


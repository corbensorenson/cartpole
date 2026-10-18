# One-link 2D Puffer hybrid: classical swing, learned capture

This fixed controller passes45/45 hanging development starts and32/32 new noisy
qualification starts. Every episode completes30 seconds and holds upright for
at least21.58 seconds. There is no hanging wait.

The first400 controls (8 seconds) use the previously qualified classical
one-link swing. A PufferLib5-trained MinGRU then captures and balances the pole.
The learned controller uses float32 CPU inference and sign-invariant quaternion
observations. The plant, every physical velocity and the actual episode clock
remain unchanged at handoff. This is a hybrid result; swing-up itself is classical.
Fourteen prepared links and ten immediate links remain the established overall
records. This one-link learned-capture track is a foundation for more links.

[Full fresh-start video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-1-link-puffer-hybrid-v1/2d-cartpole-1-link-puffer-hybrid.mp4)
· [Release and complete evidence](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-1-link-puffer-hybrid-v1)

## Reproduce

Use the pinned dependencies with Python3.11. The qualified CPU implementation
was exercised on Windows x86_64; exact byte reproduction on a different BLAS or
platform requires its own check. Deployment does not require CUDA or PufferLib.

```sh
python -m pip install -r requirements-lock.txt
python verify_source.py
python reproduce.py --seed 20330116 --noise 0.02 --out output/reproduction.npz
```

The last command executes the entire hanging-start episode and an independent
cold physics replay. It prints the strict task metrics. The ZIP unpacks this
source package under `source/`, full current controller evidence under
`evidence/`, and the original learning snapshots under `training_balance/`.
Its file inventory binds every payload byte. The original training's final
checkpoint regressed; the saved8,192,000-control checkpoint is the selected one.

The benchmark is an XY cart, passive unrestricted ball joints, an80 N force
disk,3 m square boundary,50 Hz control and200 Hz float64 MuJoCo RK4. Strict world
tilt is0.15 rad for five consecutive seconds within a30-second episode. There
are no joint actuators, locks or physical floor supports. Qualification is
empirical reset-noise evidence, not an arbitrary-gust or hardware guarantee.
The native models, inference inputs, delivered forces, full first-case
reexecution and fresh native physics replays were independently checked.

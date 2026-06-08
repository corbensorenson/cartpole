# Swing-Up Benchmark Gap

The current solved artifact is near-upright stabilization only. It starts with all links already near the upright equilibrium and uses a deterministic linear policy generated from a MuJoCo finite-difference LQR linearization.

If the comparison target starts with the links collapsed or hanging below the cart, then the target requires at least two capabilities:

- swing-up or energy shaping to reach the upright basin,
- stabilization after capture.

This repository currently demonstrates only the second capability, and only for a narrow near-upright perturbation:

```text
init_angle_noise: 0.0003
init_vel_noise: 0.00009
```

The repository now has a dedicated swing-up environment/config:

```text
configs/swingup6_uniform.yaml
```

Current configured target:

- initial mode: `hanging_curriculum` for training, evaluated at `progress=1.0`,
- initial relative joint angles: `[pi, 0, 0, 0, 0, 0] + Normal(0, 0.05)`,
- initial velocities: `Normal(0, 0.05)`,
- action space: continuous normalized cart force in `[-1, 1]`,
- force limit: `80 N`,
- rail limit: `+/-3.0 m`,
- angle failure termination: disabled,
- failure termination: rail hit or numerical failure,
- success: at least `5 s` sustained upright before the 30-second episode ends,
- upright threshold: max absolute link angle `< 0.15 rad`.
- reward gate: no survival bonus; best checkpoints are ranked by success/capture metrics before shaped return.

To pursue a stricter external benchmark, confirm whether this spec matches the target. If not, adjust it before training.

The remaining work is to train or otherwise produce a policy that:

- evaluates at `progress=1.0`, which starts from the hanging state,
- swings up or injects energy to reach the upright basin,
- captures and stabilizes all six links,
- passes held-out deterministic evaluation,
- renders a reset-free 30-second video from the hanging initial state.

If the benchmark differs, add a dedicated environment/config that:

- initializes relative or absolute link angles near the collapsed/downward state,
- documents whether actions are continuous or discrete,
- matches the target force limits and rail length,
- rewards swing-up, capture, and sustained upright balance without paying for hanging survival,
- evaluates from held-out collapsed/downward seeds,
- renders reset-free videos from that initial-state distribution.

Until that exists and passes, public wording should say "near-upright stabilization baseline," not "six-link swing-up solved" or "Yacine benchmark beaten."

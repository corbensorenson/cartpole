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

To pursue the full benchmark, add a dedicated environment/config that:

- initializes relative or absolute link angles near the collapsed/downward state,
- documents whether actions are continuous or discrete,
- matches the target force limits and rail length,
- rewards swing-up, capture, and sustained upright balance,
- evaluates from held-out collapsed/downward seeds,
- renders reset-free videos from that initial-state distribution.

Until that exists and passes, public wording should say "near-upright stabilization baseline," not "six-link swing-up solved" or "Yacine benchmark beaten."

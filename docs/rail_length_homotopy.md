# Rail-Length Homotopy Evidence

## Finding

For the current six-link energy-homotopy checkpoint, rail length is a measurable
constraint on whether the learned swing can reach an upright crossing. The
checkpoint was trained with a long-rail curriculum and evaluated with the rail
fixed at each value below. The policy and initial conditions were otherwise
unchanged.

| Rail half-width (m) | Episodes | Upright crossing rate | Low-momentum rate | Max upright streak (s) | Max cart excursion (m) | Result |
|---:|---:|---:|---:|---:|---:|---|
| 3  | 2 | 0.0 | 0.0 | 0.00 | 3.014 | rail violation |
| 6  | 2 | 0.0 | 0.0 | 0.00 | 6.009 | rail violation |
| 9  | 2 | 0.0 | 0.0 | 0.00 | 9.022 | rail violation |
| 12 | 2 | 0.5 | 0.0 | 0.02 | 12.009 | rail violation |
| 18 | 2 | 1.0 | 0.0 | 0.04 | 18.019 | rail violation |
| 21 | 2 | 0.5 | 0.0 | 0.04 | 21.013 | rail violation |

This supports the working model that a short rail clips the excitation before
the chain reaches the top, while a longer rail gives the swing expert room to
build energy in fewer or larger strokes. The current controller still carries
too much angular and cart momentum at the crossing: the best low-momentum rate
was zero and no capture state was reached. A long rail is therefore a training
wheel and a design variable, not a relaxation of the final benchmark.

## Provenance

- Checkpoint: `runs/swingup6_torch_energy_homotopy600/checkpoints/best.pt`
- Checkpoint curriculum progress: `0.41569282136894825`
- Checkpoint update: `250`
- Training method: PyTorch PPO, linear rail/start homotopy, energy shaping
- Evaluation: exact six-link MuJoCo dynamics, two deterministic episodes per
  rail, fixed rail at both curriculum endpoints
- Source records: `runs/swingup6_torch_energy_homotopy600/rail_sweep_3.json`,
  `rail_sweep_6.json`, `rail_sweep_9.json`, `rail_sweep_12.json`,
  `rail_sweep_18.json`, and `rail_sweep_21.json`

## Implication for the roadmap

The next curriculum branch should expose the swing expert to a rail ladder,
then reduce the rail only after the expert reaches a late, low-momentum
handoff. The handoff state must be saved from the actual first expert and fed
to the capture expert. A crossing at high cart position or high hinge velocity
does not count. The canonical rail remains +/-3 m for the final seven-link
benchmark.

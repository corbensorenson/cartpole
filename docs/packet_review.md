# External Training Packet Review

Reviewed 2026-09-11. This note records how the four study folders added at the repository root affect this project. They are supporting research packets, not canonical benchmark evidence.

## Conclusion

The frontier packet changes the execution priority: establish a real global swing-up and capture path before spending more effort on verification, per-state evidence packaging, or delivery-speed optimization. The hybrid packet contributes measurement and evidence discipline, and the harmonic packet adds a bounded phase/energy and morphology-design branch, but none contains a canonical high-link swing-up result.

The project benchmark remains the uniform MuJoCo plant defined in `ROADMAP.md`. Results from any packet must not be counted toward P1-P7 because their plants, simulators, action limits, rails, or initial-state distributions differ from this project.

## Frontier Study

Source: `../frontier_swingup_study/`

The study uses an ideal planar rigid-rod model, not MuJoCo. Its main task has nonuniform rod lengths and masses, a `+/-15 N` force limit, a `+/-2.4 m` rail, `0.02 s` command cadence, actuator lag, and a five-second all-link hold. Its seven/eight-link training runs use only small perturbations around the hanging state. These differences prevent direct comparison with the canonical project benchmark.

What it demonstrates:

- A genuine three-link down-start swing-up using model-based trajectory optimization and feedback tracking.
- `17/17` local three-link replay trials passing the declared five-second hold.
- Independent Cartesian/DOP853 checks supporting selected three-link successful trajectories.
- Zero seven-link and eight-link full-task successes in the tested neural and trajectory-optimization runs.
- Low collocation defect or solver convergence does not imply executable swing-up; exact forward replay and sustained hold are required.

Transferable lessons:

- Use periodic angle observations and a capture-aware objective for global learning.
- Treat coarse dynamics, collocation, and optimizer endpoints as proposal mechanisms only; exact MuJoCo replay is authoritative.
- Keep exploration, capture, sustained hold, numerical failure, rail failure, and verification as separate metrics.
- Reverse curricula, return-to-promising-state exploration, and teacher/learner training are plausible next experiments, but remain hypotheses until they produce executable project-benchmark trajectories.
- A verification method can validate a candidate; it cannot discover a candidate that never swings up.

## Hybrid Study

Source: `../hybrid_training_study/`

This study evaluates warm CPU execution, ARS/SNES variants, simple baselines, and prediction-corrected sequential verification on local 3/5-link ideal cart-poles. It uses model-based LQR assistance and local initial-state distributions. It does not run MuJoCo and does not test global hanging-start seven-link control.

Transferable lessons:

- Freeze the candidate before fresh validation and keep training, development, validation, and final test data separate.
- Record the complete task fingerprint: dynamics fidelity, action cadence, initial-state distribution, reward, success rule, seed, resource budget, and policy hash.
- Treat `ABOVE_TARGET`, `BELOW_TARGET`, `UNRESOLVED`, and numerical/domain failure as distinct outcomes.
- Account for warmup, worker reuse, CPU/RAM limits, failed requests, and overshoot instead of reporting only successful runs.
- Keep simple fixed-controller and gain-only baselines; a more complex learner must repay its cost.
- Do not reset an error budget for every adaptively selected candidate. A final frozen test is not protected by repeatedly trying candidates at the same alpha.

The prediction-corrected verifier is deferred until a real swing-up candidate exists. It is a possible validation optimization, not a discovery method or a replacement for the project's deterministic held-out gates.

## Harmonic Co-Design Study

Source: `../harmonic_codesign_study/`

The harmonic packet is a bounded exploratory morphology screen built on an ideal planar rigid-rod model inherited from the frontier packet. It evaluates 504 mass/length/damping designs at 3, 5, 7, and 8 links, then runs 512 local nonlinear recovery cases at 7 and 8 links at two integration resolutions. Its checksum manifest passes.

What it demonstrates:

- Gradient choices materially change modal frequencies, cart-force participation, upright growth rates, conditioning, and sampled LQR gain sensitivity.
- The best observed local-recovery count among the named 7-link profiles is `23/32` for a lighter-and-longer-toward-tip design, compared with `19/32` for uniform; the best 8-link named profile is `18/32`, compared with `16/32` for uniform.
- Every tested profile fails all eight shared directions at amplitude `1e-3 rad`, so the result is a narrow local-basin diagnostic, not ordinary robustness.
- The mixed designs also reduce the potential-energy gap (`43.134 J` versus `84.121 J` for the tested 7-link uniform/mixed pair), which is a confound against attributing any improvement to resonance or modal coupling alone.
- Independent derivative checks, selected DOP853 replays, and 16-versus-32-substep classification checks support the packet's numerical bookkeeping, not the canonical MuJoCo task.

Transferable lessons for this project:

- Keep a predeclared morphology-design branch for phase/energy trajectory discovery. Use modal input participation and energy gap as diagnostics, not as a swing-up certificate.
- Compare energy-gap-matched designs before attributing a result to harmonic or resonance effects.
- Treat graded mechanisms as a named co-designed benchmark or as training curricula. Final seven-link evidence must restore the canonical uniform MuJoCo morphology, `+/-3 m` rail, `+/-80 N` force, and `50 Hz` action cadence.
- After a global-discovery failure, the next model-based branch should search phase/energy motion and transverse stabilization rather than only retuning angle-only CEM scores.

The packet does not change P0-P7 status. It contains no MuJoCo rollout, no hanging-start 7/8-link solution, no learned policy, and no public-record comparison.

## New Global Discovery Gate

Before broad six-link swing-policy scaling or verification work, establish a feasibility checkpoint on the exact uniform six-link MuJoCo plant used by P3:

- Start from the published hanging distribution, not a saved handoff state.
- Produce at least one reset-free feedback-controlled rollout that reaches the all-link upright condition and holds it for at least `5 s`.
- Recheck the same controller from at least three additional held-out hanging-start seeds.
- Use the canonical six-link rail, force, damping, action frequency, and MuJoCo timestep.
- Save the full `qpos`, `qvel`, action, seed, config hash, controller hash, and failure/termination record.
- Separate this feasibility checkpoint from the P3 `20/100` episode gate; passing it does not complete six-link reproduction.

If this gate fails after the declared search budget, change the global discovery method. Do not respond by polishing the verifier, adding more state-specific capture teachers, or counting a near-upright result as swing-up progress.

## Integrity And Reproduction

- The frontier packet's `SHA256SUMS` manifest verifies successfully.
- Its archived clean-source run reports `23` tests passed; the current desktop environment cannot collect those tests until pinned `casadi==3.7.2` is installed.
- The hybrid packet's correctly configured test run reports `35` tests passed.
- Neither packet claims a seven/eight-link solution, a public-record reproduction, physical safety, or MuJoCo execution.
- The harmonic packet's `SHA256SUMS` manifest passes. Its recorded environment is Python 3.13.5 with NumPy 2.3.5, SciPy 1.17.0, and Numba 0.65.1; the project's native `.conda-aligator` environment does not currently include Numba, so the packet's full reproduction has not been rerun locally.

These packets are useful research context. They remain separate from `runs/` evidence and do not alter the project's completion status.

## Cart-Pole Research Master Handoff

Source: `../cartpole_research_master_handoff/`

This is a more complete 18-task research system than the earlier packets, but it is still an implementation handoff rather than a solved controller. Its own status files explicitly leave global discovery, capture, learning integration, MuJoCo correspondence, and the seven/eight-link down-start task unfinished. It should be treated as a protocol and component source, not as a fourth swing-up result.

### Fresh review and execution

The packet was checked locally on 2026-09-11:

- `python -m pytest -q`: `53 passed` with one SciPy SLSQP clipping warning.
- `python tools/check_packet.py`: `PASS`, with 18 task cards and 131 copied historical files verified. The current manifest contains 308 entries, while `evidence/DELIVERY_CHECK.md` reports 304 for its earlier fresh extract; that packaging count should be reconciled before treating the packet archive as final. This is bookkeeping, not scientific evidence.
- `python -m cartlab doctor`: inventory-only on this Mac/Python 3.9.23 runtime, with Torch 2.7.1 and no MuJoCo or CasADi. The packet's recorded preparation environment is Linux/Python 3.13.5, so the local doctor result does not reproduce that environment.
- `python -m cartlab next`: returns `M00`; the task graph has not been advanced to an integrated scientific result.
- Fresh positive-control replay: 3/3 three-link trials succeed in compiled RK4, entering the final hold at 6.18 s and holding through the 19 s horizon. Independent Cartesian/DOP853 replay also passes 3/3.
- Fresh negative-control replay: the seven-link fixture reproduces `RAIL_VIOLATION` at 5.60125 s, with zero achieved hold.
- The seven-link orbit command reports a numerical periodic orbit with closure about `2.6e-13`, shooting residual about `4.9e-13`, and energy drift about `1.4e-14`; its own warning correctly says this is conservative free motion, not pumped, down-start, swing-up, or stability evidence.
- The MuJoCo gate returns `BLOCKED`, zero engine checks, and exit 78. No packet result is MuJoCo evidence.

### Benchmark boundary

The packet's research benchmark is not the project's canonical benchmark. The packet uses legacy/nonuniform geometry, `+/-15 N` force, a `+/-2.4 m` rail, a `0.04 s` actuator lag, a `0.02 s` command interval, and a 19 s horizon. This project uses uniform total length/mass, `+/-80 N`, a `+/-3 m` rail, the declared MuJoCo timestep/frame-skip contract, a 30 s horizon, and the frozen hanging-start success rule in `ROADMAP.md`. Packet results therefore cannot receive P1-P7 credit without a separately named, exact benchmark adapter and fresh canonical replay.

### Reusable decisions

- Preserve the packet's immutable task/mechanism hashes, resource accounting, explicit failure classes, and fresh forward-replay rule.
- Treat collocation endpoints, conservative orbit seeds, local modal screens, and optimizer convergence as proposals only. Exact MuJoCo replay from hanging start remains authoritative here.
- When an actuator lag or other hidden plant state is introduced in a labeled ablation, record that state and the actually applied held force. Do not silently import the packet's actuator model into the canonical environment.
- Keep the packet's positive three-link and negative seven-link replays as integration-test patterns, not benchmark claims.
- Use energy-gap-matched morphology controls before interpreting a gradient as a resonance or controllability advantage. Any graded plant remains a training wheel or separately named co-designed task until the final uniform plant is restored.
- The packet's proposed sequence, discovery teacher -> saved real handoffs -> capture/stabilize -> curriculum/distillation -> frozen verification, agrees with this project's two-expert direction. The packet does not supply the missing teacher or capture policy, so the canonical global-discovery gate remains open.

The new packet strengthens the evidence protocol and the phase/energy branch, but it does not change P0-P7 status or establish a 6-link or 7-link hanging-start solution.

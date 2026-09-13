# Levers And Pitfalls

This is the living experiment ledger for the seven-link and arbitrary-`n`
program. It records what was tried, the exact mechanism, the observed result,
and the decision that follows. It is development evidence, not benchmark
completion evidence. Canonical claims still require the verifier and artifacts
listed in `ROADMAP.md`.

## Operating Goal

Solve the canonical uniform seven-link MuJoCo cart-pole from the hanging
initial distribution, swing it up, capture it, and hold it upright for the
required horizon. The dependency order is:

1. Maintain an upright seven-link chain from exact upright and progressively
   larger disturbances.
2. Capture and recover from real `qpos/qvel` states emitted by the mastered
   maintenance policy.
3. Swing from the true hanging start into the measured capture basin and hold
   without resetting or overwriting simulator state.

The public six-link result is a calibration target, not the project endpoint.
The current final target is seven links first, followed by a measured scaling
study for larger `n`.

## Literature Transfer

The current trajectory-feedback branch is based on two primary control
references, adapted as diagnostics rather than treated as proof of this
benchmark. MIT's cart-pole notes derive energy shaping through a desired cart
acceleration and cart feedback: [Underactuated Robotics, Cart-Pole]
(https://underactuated.csail.mit.edu/acrobot.html). A serial triple-pendulum
study uses nonlinear feedforward trajectory generation with a time-varying
Riccati controller under rail constraints: [Glück, Eder, and Kugi (2013),
post-print](https://www.acin.tuwien.ac.at/fileadmin/cds/pre_post_print/glueck2013.pdf).
The project implementation must still pass the exact seven-link MuJoCo gates;
these sources only motivate which controller families to test next.

## Current Scoreboard

| Branch | Setup | Result | Disposition |
| --- | --- | --- | --- |
| Static LQR | Uniform seven-link linearization, direct feedback checkpoint | Nonlinear-unstable under tested disturbance; `0/32` five-second holds | Rejected as the sole teacher; retain as a negative control |
| Longer rail plus damping | Rail/morphology training wheels at progress `0.01` | `0.00` deterministic success | Rail and damping alone are insufficient |
| Time/phase features | Fixed-stage policy with periodic cart-excitation features | No deterministic five-second successes | Do not rely on time features as the missing capability |
| Condensed upright MPC | Exact MuJoCo finite-difference model, first-action feedback, progress `0.0025` | Three independent `30/32` five-second sets; mean cart excursion below `0.012 m` | Keep as the current maintenance teacher stage |
| Maintenance frontier | Same MPC teacher at progress `0.005` | `16/32` | Treat as an unsolved next curriculum stage |
| CPU Torch gated maintenance, coarse step | Mixed system-Python Torch with native MuJoCo, `0.0025` curriculum step, 16 environments, 8-episode evaluations | Exact p0 gate passed; the best `p=0.0025` checkpoint scored `57/64 = 89.06%` on an independent evaluation, but the training run later decayed to `0/8`; the same checkpoint scored `37/64 = 57.81%` at `p=0.005` | Preserve the checkpoint as a development incumbent; replace the coarse morphology/reset step with a `0.0005` gated curriculum |
| Protected morphology-prefix PPO | Seven-link length/mass/damping/friction homotopy, `+/-12 m -> +/-3 m` rail schedule, p0 actor frozen through repeated gates, then `0.005` progress steps; follow-up used `1e-5` learning rate and hidden-layer freeze | p0 passed twice at `8/8`; the frozen policy advanced through `p=0.005` (`7/8`) and `p=0.010` (`2/8` to `4/8`), but p0-only performance fell at `p=0.015` (`5/8` then `1/8`); unfreezing or training only the action head at that point produced `0/8` | The curriculum protocol now preserves a real maintenance prefix, but PPO adaptation is still not a safe way to cross the first unsupported morphology; use an expert teacher/behavior-cloning or planner labels at the next stage |
| Real maintenance export | 128 deterministic source episodes, one qualifying state per episode | `122/128` successful source episodes and saved real `qpos/qvel` | Use as capture data; do not call it swing-up evidence |
| Teacher from real states | Seeded no-replacement sample of 32 saved states, measured velocity restored | `29/32` five-second holds at progress `0.0025` | Handoff plumbing works; capture expert remains open |
| Direct PPO continuation | PPO initialized from the teacher checkpoint on the real state list | Best observed checkpoint fell to `6/32` | Rejected; unconstrained updates overwrite the stabilizer |
| Residual capture expert | Keep MPC action active and train a zero-initialized PPO residual | Unbounded and bounded PPO residual trials fell below the `29/32` incumbent | Keep the branch as a constrained experiment; do not promote it |
| Energy-homotopy swing expert | Fade the local teacher while adding energy-progress/energy-level reward and moving hanging start toward `pi` | First seven-link probe stopped at update 70: `0/8` at the first two evaluations, maximum upright streak about `0.72 s`, no curriculum advance | Failed first-stage probe; retain the idea, redesign the interface before spending a long run |
| Global force CEM, gradient morphology | Exact MuJoCo batched search from hanging start on a long rail with base-heavy/damped morphology | 14 s search found a transient with `0.846 rad` maximum absolute angle and `1.152 rad/s` hinge RMS, but no low-momentum handoff or capture-ready state | Useful diagnostic only; smooth open-loop force knots are insufficient |
| Exact force CEM, full base-heavy gradient | Serial exact-MuJoCo 7-link search with base-heavy length/mass/damping/friction, `+/-12 m` rail, and 24 knots | The 40-iteration search converged to the zero-force hanging trajectory: `3.142 rad` angle, `0.0014 rad/s` hinge RMS, and `0.0002 m` rail use; no swing-up event | This profile is over-conservative for the current zero-centered force search; keep its gradient as a curriculum candidate, but seed it from a feedback/energy route rather than open-loop noise |
| Exact force CEM, length/mass-only gradient | Serial exact-MuJoCo search with base-heavy lengths and masses but uniform damping/friction, `+/-12 m` rail, and the same search budget | The 40-iteration search also stayed near hanging: `3.141 rad` angle, `0.105 rad/s` hinge RMS, and `0.005 m` rail use; no swing-up event | Length/mass grading alone did not make an unseeded open-loop search discoverable; test the packet's tip-heavy profile and feedback-conditioned continuation separately |
| Warm-started global CEM refinement | Reused the prior force-knot route with lower initial variance | Refinement worsened the best transient to about `1.256 rad` angle and `2.52 rad/s` hinge RMS | Reject this warm-start schedule; try a different parameterization and objective |
| Corrected exact-hanging global CEM | Zeroed every reset-noise scalar and schedule before a seven-link long-rail force search | Best late state reached only about `1.795 rad`; the prior `0.846 rad` result was invalidated because it inherited scheduled noise | Keep the correction; never count the older noisy artifact |
| Serial-angle boundary audit | Rechecked every batched trajectory scorer and feature map at the exact hanging pose | Several global-CEM and energy-feature paths wrapped each relative `+pi` joint to `-pi` before the serial sum, so the untouched hanging state could score as zero angle; all pre-fix global-CEM scores are invalidated | Accumulate relative coordinates first, wrap only the cumulative absolute angle, and keep the regression test in the native suite |
| Batched rail-contact audit | Replayed the saved corrected wide-rail global-CEM knots step-by-step in exact MuJoCo | The batch evaluator allowed candidates to reach the MuJoCo slide-joint limit because batched rollout does not execute the environment termination check; the saved `rail=12.06 m` route was a rail-clamped artifact and failed replay before the intended tail | Reject trajectories at `99.5%` of the configured rail and require step-by-step replay before any route can seed capture |
| Guarded uniform global force CEM | Exact uniform seven-link hanging start, temporary `+/-12 m` rail, 36 force knots, 80 iterations, boundary guard, soft `10 m` rail preference | Best replayable route reached about `1.08 rad` absolute angle, `1.55 rad/s` relative hinge RMS, `4.95 m` cart position, and `5.05 m` maximum excursion; no capture-ready state | Valid intermediate swing route only; use it as a prefix seed, not as a handoff or solution |
| Uniform tail refinement, point objective | Replayed the guarded route to `10 s`, then searched a two-second exact-MuJoCo force tail | Found a real near-upright crossing at `0.061 rad` maximum angle, but hinge RMS was `1.13 rad/s`, cart speed `1.33 m/s`, and cart position `1.72 m`; no feasible handoff | Angle crossing is still a false positive unless full-chain rate and cart state are low at the same timestamp |
| Uniform tail refinement, low-momentum objective | Warm-started the point tail and searched a four-second tail with stronger hinge/cart weights and a ten-step robust window | Settled near `0.285 rad`, `0.80 rad/s` hinge RMS, `0.07 m/s` cart speed, and about `0.70 m` cart position; no angle-qualified handoff | The tail can trade angle for momentum, but this source prefix does not yet enter the capture basin |
| Real capture from uniform tail crossings | Materialized exact `qpos/qvel` from the two-second and four-second tail outputs and ran separate nonlinear capture CEM searches | Near-upright crossing: `0/1`, `0.04 s` upright, `0` low-momentum; lower-momentum state: `0/1`, `0.06 s` upright, `0` low-momentum | Both are genuine component failures; train the capture expert on a measured state envelope and improve the swing objective against that envelope |
| Cart-PD swing plus hinge-heavy tail CEM | Replayed the real cart-trajectory source, then searched a five-second exact-MuJoCo force tail | Found a nominal handoff at `0.142 rad`, hinge RMS `0.524`, `x=0.739 m`, cart speed `0.436 m/s`; exact replay under online MPC and the existing maintenance teacher both held only `0.02 s` | Planner box is not the capture basin; optimize arrival margin or train a dedicated capture expert |
| Open-loop capture CEM from the real handoff | Searched an eight-second action spline from the saved `qpos/qvel` state | No upright event or hold; best pass was about `0.287 rad` and `0.533 rad/s`, with rail contact | Open-loop control cannot arrest this late handoff |
| Extended eight-second tail CEM | Warm-started a low-hinge five-second tail and required an eight-second terminal route | Converged to about `0.67-1.60 rad` angle with `1.12-1.52 rad/s` hinge RMS; no feasible handoff | Earlier warm start does not produce a robust endpoint |
| Tail start at `12.5 s` | Began the tail closer to the cart-PD route's intermediate crossing and searched five seconds | Best late score was about `0.365 rad` angle with `2.23 rad/s` hinge RMS; no declared feasible handoff | This source phase is still too energetic |
| Rail-matched tail iLQR | Refined the five-second tail with exact feedback and a `12 m` rail penalty, avoiding the prior canonical-rail mismatch | Optimizer converged numerically, but replay hit the rail and held `0.00 s` | Optimization convergence is not capture evidence |
| Fixed-state iLQR terminal feedback | Optimized a four-second feedback trajectory directly from the saved handoff and held the terminal local feedback | Replay hit the rail; maximum upright streak `0.02 s` | The saved state is outside the tested local feedback basin |
| Dedicated handoff PPO | Raw-action PPO from the one real handoff on the easier base-heavy/damped plant | Aborted at update 215 after no improvement beyond a `0.02 s` transient; evaluation success remained `0/1` | Single-state PPO needs a better teacher/rollback or a wider, staged state distribution |
| Rolling-window global force CEM, wide rail | Forty-eight force knots, `18 s`, `25`-step rolling window, rail `12 m`, base-heavy/damped morphology | Best endpoint was about `0.584 rad`, hinge RMS `1.290`, `x=7.82 m`, cart speed `0.956 m/s`; no handoff | Long rail enables partial synchronization but not a centered low-momentum route |
| Force-authority training wheel | Repeated rolling force CEM at `120 N` instead of canonical `80 N` | Best was about `0.432 rad`, hinge RMS `1.570`, `x=-3.15 m`; no capture-ready state | More force alone is not the missing capability |
| Force `200 N` numerical probe | Same global scorer at excessive force authority | MuJoCo produced `NaN/Inf/huge-state` warnings and zero-valued bogus metrics; run was stopped and not scored | Global search now rejects non-finite/huge batch rollouts before ranking |
| Robust handoff-window CEM | Added a final-window objective requiring the last `25` policy steps (`0.50 s`) to remain inside the angle, hinge, cart, and cart-speed box | The optimizer moved away from upright; best endpoint was about `2.67 rad`, hinge RMS `1.36`, with no robust or point handoff | The earlier `0.142 rad` state was a one-frame crossing, not a stable arrival; score the downstream capture basin directly |
| Capture-value tail CEM | Evaluated each exact-MuJoCo tail endpoint by running the actual maintenance checkpoint for `1.5 s` from its emitted `qpos/qvel` | No candidate produced an upright streak; best downstream cost improved while endpoint angle settled around `0.31-0.40 rad` and hinge RMS `1.54-1.68` | Existing maintenance teacher has no useful capture authority at the swing boundary; keep the evaluator and replace or train the capture expert |
| Capture teacher authority sweep | Re-materialized local MPC teachers at policy scales `10x` and `100x` and replayed the same handoff | `10x` still applied only about `-0.002` action; `100x` reached about `-0.044` by `0.16 s` but accelerated the cart in the wrong direction; both held only `0.02 s` | A stronger gain alone is not enough; capture needs a correctly phased nonlinear feedback policy and a wider training envelope |
| Closed-loop linear/tanh CEM | Searched a state-feedback policy directly on the easier base-heavy/damped 7-link plant with a `12 m` rail, then replayed the same actor for `15 s` | Found a genuine hanging-start upright crossing at `0.069 rad`, but only `0.04 s` of upright streak; the joint handoff score was `138.93` because hinge RMS was still `2.643 rad/s`, and the cart excursion was `3.75 m` | State feedback can discover a swing phase, but angle-only improvement is another false handoff unless all quantities are scored at the same timestamp |
| Joint handoff-scored linear CEM | Corrected the scorer to minimize angle, hinge velocity, cart position, and cart velocity from the same state, then increased the joint handoff weight to `15x` and `50x` | Both continuations stayed at the same `0.04 s` crossing and `138.93` joint score; the cold `50x` branch did not find a lower-momentum arrival | The linear policy family and current CEM update are not enough; use a separate capture learner or richer phase/energy policy with demonstrations |
| Phase-feature linear CEM | Added six explicit time harmonics (`0.5, 1, 2, 3, 4, 5 Hz`) to the closed-loop linear policy on the same gradient/long-rail plant | Reached `0.04 s` upright streak at `0.072 rad`, but the best joint handoff still had `3.305 rad/s` hinge RMS and joint score `206.75` | Phase features improve timing discovery but do not supply capture feedback; retain as a diagnostic, not a solution path |
| Two-expert linear chain | Held the state-feedback swing actor fixed and CEM-optimized a separate capture actor with a reset-free phase switch | Switch at `9.8 s` reached `0.06 s` upright streak and joint score `92.64` (`2.036 rad/s` hinge RMS); moving the switch to `9.6 s` improved the joint score to `82.54` (`1.975 rad/s`) but still no sustained hold; switch at `10.0 s` did not improve | The two-expert decomposition is directionally correct, but this linear capture family still cannot arrest the multi-link momentum; continue with a nonlinear or learned capture expert |
| Time-varying trajectory feedback | Reconstructed the exact source-plus-tail route, finite-differenced every nominal tail state, applied finite-horizon Riccati feedback, then used an upright takeover | Reproduced the `0.1421 rad` handoff exactly; upright spectral radius was about `1.0005`; gain sweeps either saturated into the rail or produced no hold | Literature-aligned feedback does not repair this arrival state; the feedforward handoff still needs a larger capture basin |
| Long smooth handoff capture | CEM from the saved real handoff, `41` knots, `8 s`, `30` iterations, no state reset | Best next state was about `0.1643 rad`; maximum upright streak stayed `0.02 s` | A longer open-loop sequence cannot arrest the saved handoff's immediate momentum |
| Uniform trajectory CEM | Exact hanging start, progress `1.0`, temporary `20 m` rail, `12` CEM iterations | Best late state was about `0.534 rad` with `1.768 rad/s` hinge RMS and no upright event | The base-heavy trajectory is not a canonical uniform route; keep morphology homotopy explicit |
| Mass-matrix energy shaping | Virtual cart acceleration from energy error, link horizontal momentum, phase, and cart state; force recovered from the MuJoCo mass matrix | Base-heavy/damped `20 s` CEM reached energy fraction near `0.995`, but best late angle stayed about `0.72 rad` and upright streak was `0` | Energy injection without internal phase synchronization is insufficient; add modal or trajectory coordination |
| Absolute-rate constrained tail shooting | Exact MuJoCo replay of the long-rail source route, with constrained final-tail optimization on absolute link angular velocity `cumsum(qvel[1:])` | Tightening the absolute-rate RMS bound from `0.75` to `0.004` produced increasingly quiet endpoints: `0.389`, `0.226`, `0.149`, `0.077`, and `0.040 rad/s`; the corresponding downstream hold tests were only `0.28`, `0.48`, `1.12`, `4.56` (rail-invalid), and `6.78 s` on a single exact state | Relative hinge RMS is not enough; absolute-rate constraints improve the handoff, but the route remains a p0/long-rail discovery artifact and the capture basin is still narrow |
| Exact-state nonlinear capture CEM | Actor searched from the `absolute004` constrained handoff with whole-chain rate, centered-position, and cart-travel penalties | One reset-free `8 s` diagnostic reached `6.78 s` maximum upright and centered streak, `6.76 s` whole-chain low-momentum streak, and `1.141 m` maximum cart excursion; it later fell before the time limit | Genuine component evidence, not an integrated result; train against a state set rather than fitting one handoff |
| State-set nonlinear capture CEM | One actor optimized against the `absolute004`, `absolute008`, and `absolute015` measured handoffs using worst-case plus mean score | Only `1/3` states succeeded; worst-case upright/centered streak was `0.78 s` and maximum cart excursion was `1.919 m` | The exact-state actor does not generalize; expand the measured handoff set and use hard-negative curriculum |
| Capture stability-time objective | Added whole-horizon absolute-low-momentum time to the CEM score and seeded from the exact-state actor | `absolute004` held `8.02 s` upright, `8.00 s` whole-chain low momentum, with `0.289 m` maximum cart excursion and terminal cost below `1` | Keep the objective; the local stabilizer is now durable over the short component horizon |
| Shared stability-time capture actor | Optimized one linear actor over `absolute004`, `absolute008`, and `absolute015` with the same whole-horizon score | `1/3` success; best state held `5.78 s`, but the other states held only `0.88 s` and `0.74 s`, with up to `6.02 m` cart excursion | The state-set jump remains too wide; add hard negatives gradually and preserve the incumbent |
| Two-mode capture/stabilize CEM | Smooth momentum gate between separately searched high-momentum and low-momentum actors over three states | `0/3` successes; best worst-case centered streak `0.70 s`, and all three candidates reached about `12.08 m` rail excursion | Gate alone is not enough; enforce a hard cart barrier and improve the local policy representation |
| Rich interaction-feature CEM | Added angle-rate, angle-cart, cart-rate, and squared-rate features to a seeded linear actor | `0/3` successes; worst-case centered streak `0.90 s`, maximum cart excursion `8.19 m` | More features without a robust objective did not produce a capture basin |
| Two-state curriculum rung | Re-optimized a shared actor on only `absolute004` and `absolute008` with stability-time scoring | `1/2` successes; one state held `5.08 s`, the neighboring state held `0.88 s`, with `4.13 m` maximum cart excursion | The curriculum must expand through a measured basin, not jump directly to the full envelope |
| Reset-free planner-to-capture chain | Replayed the hanging-start source, constrained prefix, and constrained suffix once, then switched to the stabilized capture actor without resetting | Endpoint reproduced the saved handoff with `0` position/velocity error; the chain held all seven links centered and whole-chain low-momentum for `8.00 s` at p0/`+/-12 m` | First genuine integrated discovery chain; not canonical evidence, and the `30 s` replay failed at `9.08 s` on the rail |
| Long-horizon local capture search | Re-ran the stability-time CEM from `absolute004` with a `30 s` horizon, then tested simple cart PD corrections after capture | The best local actor still reached only `9.10-9.16 s`; the integrated chain reached `9.08 s` before rail violation | Local cart centering is not sufficient; the next stabilizer needs a stronger model-based or learned maintenance interface |
| Morphology transfer of integrated chain | Replayed the saved p0 planner controls and p0 capture actor at fixed `plant_progress` values | At `0.0025` the suffix hit the `12 m` rail before handoff; at `0.0001` endpoint error was `0.901` in `qpos` and `3.757` in `qvel`, with no centered hold | The p0 route cannot be annealed by replay; regenerate the planner and capture labels at every homotopy step |
| Finite-difference LQR residual curriculum | Uniform seven-link morphology, LQR teacher at scale `1.18`, CPU Torch residual PPO, actor frozen through the mastered frontier, `0.005` progress gates, rollback incumbent preservation, and reset state scales ending at `0.002` | Fine continuation advanced from `p=0.900` through `p=0.995`; the saved `p=0.999` frontier appeared to pass the scaled handoff, but the exact unscaled audit failed | Fine-grained gated continuation is useful, but reset qpos/qvel scaling was a hidden training wheel; the frontier is not a capture expert |
| Fine endpoint continuation | Started from the `p=0.995` frontier and advanced `0.001` progress at a time on the same uniform plant | Passed `p=0.996`, `0.997`, `0.998`, and `0.999`; the `p=1.0` stage repeatedly failed under the training residual cap of `0.5` | The final endpoint is sensitive to both curriculum step size and residual authority; preserve the last successful frontier rather than overwriting it |
| Residual-authority endpoint probe | Evaluated the saved `p=0.999` frontier at `p=1.0` with residual action limit `1.0`, but the checkpoint still used `init_qpos_scale=init_qvel_scale=0.002` | The scaled-state evaluation succeeded for `30 s`; the exact unscaled replay of the same saved handoff was `0/1`, hit the `13 m` rail after `1.32 s`, and held upright for only `0.02 s` | Residual authority was not the only training wheel; state scaling invalidated the apparent endpoint result and must be removed before capture claims resume |
| Scaled-state seven-link capture artifact audit | Replayed `runs/swingup7_capture_lqr118_final` with the recorded scaled reset and then with `init_qpos_scale=init_qvel_scale=1.0` | Scaled reset: `1/1` success for `30 s`; exact unscaled reset: `0/1` success, `0.02 s` maximum upright streak, `13.063 m` maximum cart excursion, `rail_violation` | This is a curriculum checkpoint and a negative control for the real handoff; it is not admissible full-scale capture evidence |
| Final-morphology long-rail trajectory CEM | Exact serial seven-link hanging start, canonical uniform plant, `+/-20 m` rail, `30 s` horizon, late capture-ready objective | Best late point reached `0.130 rad` maximum angle with `1.198 rad/s` relative hinge RMS and `0.065 rad/s` cart speed, but only `0.04 s` upright streak; exact replay through the existing capture expert hit the `20 m` rail | Strongest final-plant reachability candidate so far, but not a low-momentum handoff; preserve the controller as a swing-search seed only |
| Chain-aware swing plus LQR capture CEM | Exact serial swing controller and LQR switch, canonical uniform plant, `+/-20 m` rail, switch selected by angle and time | Best route entered capture at `8.96 s`, reached `0.129 rad`, `1.733 rad/s` relative hinge RMS, and `-0.947 m` cart position; LQR saturated and the route held only `0.06 s` before rail loss | Two-expert structure is directionally right; the swing expert must reduce joint rates before capture rather than handing a saturated LQR an energetic state |
| Receding-horizon capture from measured chain handoff | Random-shooting MPC from the exact chain-generated `qpos/qvel`, `8 s` replay, `+/-20 m` rail | Briefly reached `0.039 rad`, but the best hinge RMS was `1.928 rad/s`, with `0.06 s` upright streak and no hold | Short-horizon action search cannot rescue the current handoff; keep it as a diagnostic, not as the capture expert |
| Unit-scale capture PPO from chain handoff | CPU Torch PPO, exact saved chain state, `plant_progress=1`, reset scales `1.0`, residual teacher disabled | After about `0.94M` environment steps, evaluations remained `0/1`; returns degraded toward `-1,700` and no hold emerged | Do not train capture first from a high-momentum boundary; improve the swing handoff envelope before another learner run |
| Final-plant modal and mass-matrix energy searches | Interpretable phase/energy CEMs on the canonical uniform plant with a `+/-20 m` rail | Modal search stopped near `0.932 rad` with no upright event; mass-matrix search reached energy fraction `1.000` but only `1.058 rad` angle and `3.615 rad/s` hinge RMS | Total energy is not the bottleneck; internal phase synchronization and damping remain the active discovery problem |
| Exact hybrid replay trace | Replayed the recorded chain controller with its angle-gated capture switch, original seed, and configured cart-velocity noise | Reproduced the source handoff exactly at `8.96 s`: `0.125514 rad`, `2.578728 rad/s` relative hinge RMS; the next capture step matched `0.128716 rad`, `1.732562 rad/s`, then rail loss at `10.48 s` | Preserve the full hybrid trace; a swing-only controller JSON is not a valid warm start for a chain result |
| Exact-state tail CEM from hybrid handoff | Seeded an open-loop action tail from the measured `8.96 s` state and the actual downstream capture actions | No rail-safe tail; sampled candidates became unstable or reached the `+/-20 m` rail | Open-loop tail refinement is rejected for this handoff; keep feedback in the optimization loop |
| Final-plant local capture family | Unit-scale affine, rich-interaction, momentum-mixture, two-phase, and fixed-state iLQR searches from exact measured handoffs | Chain-LQR state reached at most `0.06 s` centered upright; the cleaner trajectory handoff also reached `0.06 s`, but `0.0 s` absolute-low-momentum hold; iLQR hit the rail | The upstream handoff remains outside the tested stabilizer basin |
| Fine-scale LQR equilibrium sweep | Replayed the exact quiet seven-link state with finite-difference discrete LQR, normalized control scales down to `0.0001`, tanh/clip squashing, and action slew limits | Best candidate reached `0.32 s` upright with `0.001512 rad` minimum angle, then violated the canonical `+/-3 m` rail; no candidate held | The original LQR amplitude was too aggressive, but reducing it alone does not create a capture basin; retain this as a negative control and use constrained nonlinear feedback or a learned local controller |
| Rail-width sensitivity of local capture | Replayed the same exact quiet state and fine LQR sweep at `+/-3`, `+/-6`, `+/-12`, and `+/-20 m` | The best upright streak stayed `0.32 s`; the cart reached each wider boundary in turn, while the `+/-20 m` run reached `19.7 m` at the time limit without holding | A longer rail is a valid discovery/training wheel and may be needed for the swing phase, but width alone does not create capture; measure the minimum rail for each successful controller and return to `+/-3 m` for final evidence |
| Absolute-rate swing objective | Added cumulative absolute link-rate RMS to the uniform-plant trajectory CEM and raised its weight in a follow-up run | Angle-constrained candidate: `0.149208 rad`, `1.064008 rad/s` relative hinge RMS, `2.495875 rad/s` absolute RMS, `x=0.148620 m`, `xd=0.043985 m/s`; absolute-priority Pareto point: `1.4546 rad/s` absolute RMS but `0.526 rad` angle | Optimize a constrained Pareto handoff; raw energy and single angle minima are insufficient |

The exact handoff evaluator records the selected state indices in
`runs/swingup7_capture_mpc0025_handoff_eval32_indexed.json`. The source data
are in `runs/swingup7_policy_handoff/maintenance_mpc0025_verified_states.json`.

## Experiment Ledger

### Upright maintenance

- **Exact upright PPO.** Started from the upright equilibrium with zero actor
  output and progressively introduced reset noise. The learned actor drifted
  away from the equilibrium before mastering the first nontrivial stage.
  Conclusion: ordinary PPO exploration is too destructive at this boundary;
  preserve a known stabilizer or use a constrained residual parameterization.
- **Static LQR.** Linearized the seven-link MuJoCo plant and materialized a
  policy checkpoint. The linearization is poorly conditioned/deficient for
  the full state, and nonlinear replay drove the cart out of the usable rail.
  Conclusion: use LQR as a diagnostic and negative control, not as the global
  seven-link controller.
- **Condensed linear MPC.** Used the exact finite-difference dynamics and a
  short horizon with high control cost to compute the first action. This is
  the first repeatable local maintenance teacher at `p=0.0025`. It is not a
  swing-up policy and does not satisfy the canonical result.

### Capture and recovery

- **Synthetic upright resets.** Useful for checking observation and reward
  plumbing, but they do not test the real handoff. They are not acceptable as
  capture evidence.
- **Real state-list handoff.** Exported post-step MuJoCo states after policy
  control, preserving both positions and velocities. A direct teacher
  continuation succeeds on most, but not all, states. The remaining failures
  are the hard-negative recovery set.
- **Velocity restoration curriculum.** The environment can blend saved
  velocities from zero to their measured values. Use this only during
  training; the final handoff audit must use the actual saved velocities.
- **Direct PPO fine-tune.** Initialized the policy from the direct teacher and
  trained on the real list. It lost the teacher's stability. This is a failure
  of the optimization parameterization, not proof that the plant is
  uncontrollable.
- **Teacher residual.** The capture config now keeps the condensed-MPC action
  in the environment and starts PPO with a zero residual. This protects the
  already-working equilibrium behavior while allowing recovery corrections.
  The first unbounded PPO trial still destabilized the teacher (`0/32` at its
  first evaluations). Trust-region retries with normalized residual caps of
  `+/-0.10` and `+/-0.005` also failed to preserve the `29/32` teacher baseline;
  the tiny-cap trial briefly reached `7/32` and then fell to `0/32`. Ordinary
  PPO is therefore not yet a safe optimizer for this handoff, even when the
  teacher remains active. The next capture experiment must use rollback,
  supervised teacher matching, hard-negative weighting, or a non-PPO optimizer.
- **CPU Torch maintenance curriculum.** The mixed system-Python Torch runtime
  successfully trained and evaluated the maintenance policy with native
  MuJoCo. The exact upright p0 stage passed its repeated gate. With a coarse
  `0.0025` progress step, the best checkpoint reached `57/64` successes at
  `p=0.0025`, but continued on-policy updates destroyed that basin and the
  run ended at `0/8`; the preserved checkpoint fell to `37/64` at `p=0.005`.
  This is a real maintenance frontier, not a canonical result. The next run
  uses `0.0005` gated steps and keeps the best checkpoint as the rollback
  incumbent.

### Swing-up discovery

- **Low-momentum handoff objective.** Low hinge, cumulative absolute-link, and
  cart momentum are valuable shaping signals because they often enlarge the
  capture basin, but they are not a standalone acceptance gate. The swing
  expert should be ranked by downstream capture value and the actual capture
  expert must be tested from real saved `qpos/qvel` states, including states
  that still carry substantial momentum. Every candidate route must be replayed
  in exact MuJoCo.
- **Handoff gate interpretation.** The benchmark gate is uninterrupted
  swing-up, capture, and sustained hold from hanging start. Angle, relative
  hinge-rate RMS, cumulative absolute-rate RMS/max, cart state, rail margin,
  and time-to-capture remain mandatory telemetry and useful Pareto objectives;
  a nominal low-momentum threshold is a curriculum target, not a reason to
  discard a state before closed-loop capture has tried it.
- **Longer rail.** Development schedules use rails such as `+/-9 m` or
  `+/-12 m` before annealing to the canonical `+/-3 m` rail. This is a training
  wheel for the windup, not final evidence. We must measure the minimum rail
  required as a function of total chain length instead of assuming that one
  rail works for every `n`.
- **Morphology gradients.** Base-heavy, length-graded, damping, friction, and
  mass schedules are permitted discovery homotopies. They may reveal a route,
  but the final policy must return to uniform length/mass/damping and the
  canonical force and rail limits.
- **Global search and trajectory optimization.** CEM, action-spline,
  feedback-MPC, DDP/iLQR, and ProxDDP probes have produced useful diagnostics
  and isolated local capture funnels, but no canonical seven-link hanging-start
  route. A planner result is only a teacher proposal until its uninterrupted
  MuJoCo rollout succeeds.
- **Rate-weighted direct-force refinement.** A canonical-rail exact force search
  warm-started from the best replayable trace reached a brief crossing near
  `0.0998 rad` maximum angle, `2.084 rad/s` cumulative absolute-rate RMS,
  `1.309 rad/s` relative hinge RMS, `x=0.610 m`, `xd=-0.909 m/s`, and `2.722 m`
  maximum rail use. Direct feedback, open-loop sequence, richer interaction,
  two-phase, and fixed-state iLQR capture probes from that real state produced
  at most `0.02 s` upright streak and no hold. A `0.50 s` rolling-window search
  reduced cart speed but moved the best angle to about `0.564 rad`. This is a
  measured Pareto tradeoff, not evidence that low momentum should remain a hard
  gate.
- **Global force CEM.** A 40-iteration, 28-knot exact-MuJoCo search on a
  long-rail, base-heavy/damped seven-link development plant found a transient
  with all absolute link angles below `0.846 rad`, but hinge RMS was still
  `1.152 rad/s` and the state was not accepted by the capture criteria. A
  warm-start refinement made the best recorded transient worse. This rules out
  treating smooth open-loop force knots as a sufficient discovery mechanism;
  the next search must include phase feedback, energy shaping, or an explicit
  capture-expert value.
- **Serial-angle boundary audit.** The first uniform-plant global CEM rerun
  exposed an evaluator bug: the batch scorer wrapped each relative joint
  before accumulating the serial chain. At the exact hanging boundary, `+pi`
  became `-pi`, and a second wrap turned the chain into an apparent upright
  state. The same pattern existed in the online-MPC and tail-CEM batch metrics
  and in several energy/capture feature maps. The shared
  `serial_absolute_angles` helper now accumulates first, the native regression
  test asserts that exact hanging is not upright, and every pre-fix global-CEM
  score is treated as invalid until rerun.
- **Batched rail-contact audit.** A saved wide-rail force route reported a
  `12.06 m` excursion but could not be replayed to its intended tail because
  the batched MuJoCo path does not execute the environment's rail termination
  check. The cart had reached the slide-joint boundary and the batch route was
  therefore a rail-clamped artifact. The global scorer now rejects candidates
  at `99.5%` of the configured rail; every route must also pass step-by-step
  replay before it can generate a capture state.
- **Guarded uniform force route.** With the corrected metric and rail guard, a
  uniform seven-link hanging-start CEM on a temporary `+/-12 m` rail produced
  a replayable intermediate route, but only `1.08 rad` angle, `1.55 rad/s`
  relative hinge RMS, and `5.05 m` maximum cart excursion. It is useful as a
  prefix seed, not a solution.
- **Uniform tail and real capture tests.** A two-second tail from that prefix
  found a real `0.061 rad` crossing, but with `1.13 rad/s` hinge RMS and
  `1.33 m/s` cart speed. A four-second low-momentum refinement traded the
  angle for a `0.285 rad` endpoint with `0.80 rad/s` hinge RMS and `0.07 m/s`
  cart speed. Capture CEM from both exact emitted states succeeded `0/1` and
  held upright only `0.04 s` and `0.06 s`, respectively. These are genuine
  component failures and should drive the next capture-basin curriculum.
- **Robust handoff window.** Added a terminal-window objective to the tail CEM
  so the final `25` policy states, not only the best state, had to remain inside
  the low-momentum box. The search left the upright region entirely, with its
  best endpoint around `2.67 rad`. This is a useful failure: the previous
  `0.142 rad` handoff was a late crossing with no arrival margin.
- **Downstream capture-value search.** Added
  `scripts/search_swingup_tail_capture_value.py`. It replays every candidate
  tail in batched MuJoCo, then runs the incumbent feedback checkpoint from the
  candidate's emitted terminal state for `1.5 s`. No candidate produced an
  upright streak. The best physical endpoint remained roughly `0.31-0.40 rad`
  with `1.5-1.7 rad/s` hinge RMS, showing that the current maintenance policy
  cannot serve as the capture expert.
- **Capture-authority sweep.** Generated `10x` and `100x` scaled versions of
  the local MPC checkpoint. The `10x` policy still produced only milliaction
  corrections; the `100x` policy became visibly active but drove the cart in
  the wrong phase. Both held only the first `0.02 s` crossing. Gain magnitude
  is therefore not the missing ingredient; action timing and a learned or
  nonlinear capture basin are.
- **Closed-loop linear/tanh policy search.** A CPU-only CEM over direct state
  feedback discovered a genuine hanging-start upright crossing on the
  base-heavy/damped plant with the `12 m` rail. The actor reached `0.069 rad`
  but only held for `0.04 s`; the same state still had `2.643 rad/s` hinge RMS
  and the route used `3.75 m` of cart excursion. The first implementation
  incorrectly combined minima from different timestamps, so it was repaired
  to score a joint handoff state. Weighted continuations then failed to improve
  the `138.93` joint score. Keep the actor as a phase-discovery diagnostic, not
  as a capture label.
- **Explicit two-expert linear chain.** Held the discovered swing actor fixed
  and optimized a separate linear/tanh capture actor after a reset-free phase
  switch. A `9.8 s` switch reached `0.06 s` upright streak and joint score
  `92.64`; a `9.6 s` switch reduced the score to `82.54` and hinge RMS to
  `1.975 rad/s`, while `10.0 s` returned to the original crossing. This is
  the first evidence that separating the experts helps, but it remains a
  diagnostic because no switch produced a sustained hold.
- **Time-varying trajectory feedback.** Added
  `scripts/evaluate_time_varying_trajectory_feedback.py`, which reconstructs
  the exact source-plus-tail nominal route, finite-differences MuJoCo at every
  tail state, applies a finite-horizon Riccati feedback law, and then switches
  to an upright gain without resetting state. The evaluator reproduced the
  saved handoff exactly, including the `0.1421 rad` endpoint. The upright
  takeover had spectral radius about `1.0005`; low-penalty gains saturated into
  the rail, while high-penalty gains stayed bounded but still produced no
  sustained hold. This transfers the two-degree-of-freedom feedforward plus
  time-varying-feedback idea into the project, but does not repair the current
  arrival basin.
- **Long smooth open-loop capture sequence.** Re-ran CEM directly from the
  real saved handoff with `41` action knots, an `8 s` horizon, and `30` search
  iterations. The optimizer could not keep the chain inside the upright box
  after the first step; the best next state was about `0.1643 rad` and the
  maximum upright streak remained `0.02 s`. A longer action sequence alone is
  not enough when the saved handoff has no admissible one-step capture action.
- **Uniform-morphology trajectory probe.** Repeated the trajectory CEM at
  progress `1.0` with exact hanging start and a temporary `20 m` rail. The
  best late candidate reached only about `0.534 rad` with `1.768 rad/s` hinge
  RMS and no upright event. The earlier base-heavy result must therefore stay
  labeled as a discovery homotopy; it does not transfer automatically to the
  canonical uniform plant.
- **Mass-matrix energy shaping.** Added
  `scripts/search_energy_shaping_feedback.py`. It computes relative pendulum
  energy and horizontal link momentum, selects a virtual cart acceleration,
  and uses the MuJoCo mass matrix to map that acceleration to force. On the
  base-heavy/damped plant, a `20 s` CEM reached an energy fraction near `0.995`
  but its best late angle was still about `0.72 rad`, with zero upright
  streak. The controller can inject energy but does not synchronize the seven
  internal phases; this lever needs a modal/trajectory layer before it is a
  credible swing expert.
- **Relative versus absolute angular velocity.** The MuJoCo hinge velocities
  in `qvel[1:1+n]` are relative joint rates. The physical angular rate of each
  link is their cumulative sum, so a chain can have a small relative hinge RMS
  while the links still carry substantial coordinated motion. The deep
  constrained tail endpoint measured `0.358 rad/s` relative RMS but `1.377
  rad/s` absolute RMS. The environment now exposes both absolute RMS and
  maximum absolute link rate, and optional reward/switch constraints can use
  the absolute metric. Future handoff labels must record both quantities.
- **Absolute-rate constrained tail series.** Added absolute-rate constraints
  to `src/gcartpole/constrained_shooting.py` and replayed the source route with
  progressively tighter final-tail bounds. The `absolute004` endpoint reached
  `0.0015 rad` maximum angle, `0.040 rad/s` absolute-rate RMS, `0.060 m/s`
  cart-speed bound, and `0.289 m` cart position on the discovery rail. It is a
  much better physical handoff than the old `0.142 rad` crossing, but it still
  requires a learned capture policy and is not canonical evidence.
- **Exact-state nonlinear capture.** Added whole-chain absolute-rate and
  centered-cart terms to `scripts/search_capture_feedback_cem.py`. From the
  `absolute004` state, the best actor reached `6.78 s` centered upright and
  `6.76 s` whole-chain low-momentum streak before falling later in the `8 s`
  diagnostic. Applying that same actor to the `absolute008` and `absolute015`
  states produced only `0.78 s` and `0.88 s`, respectively. The actor is a
  useful local controller but is overfit to one handoff.
- **State-set capture search.** Added
  `scripts/search_capture_feedback_cem_state_set.py` to optimize one actor
  over multiple saved handoffs. The first three-state run succeeded on only
  `1/3` states, with a worst-case `0.78 s` centered upright streak. This
  converts the current bottleneck from "find any quiet endpoint" to "expand
  and learn the capture basin," which is the correct next interface for the
  swing expert.
- **Whole-horizon stability score.** Added a configurable
  `stable_time_weight` to the capture CEM so a controller is rewarded for
  every timestep inside the whole-chain low-momentum box, not only its best
  streak. Seeded from the exact-state actor, this improved the
  `absolute004` component to `8.02 s` upright and `8.00 s` absolute-low
  momentum with `0.289 m` cart excursion. A 30-second local search did not
  extend it beyond roughly `9.10 s` before rail loss.
- **Staged state-set curriculum.** Re-optimizing on the first two measured
  handoffs produced `1/2` successes: `absolute004` held `5.08 s`, while
  `absolute008` held `0.88 s`. Adding `absolute015` returned `1/3`, with
  worst-case `0.74 s`. The evidence supports incremental hard-negative
  expansion, but the current local feature policy does not yet enlarge the
  basin.
- **Two-mode and rich-feature searches.** A smooth momentum gate between
  separate high- and low-momentum actors reached `0/3` and drove all three
  states to the `12 m` rail. A richer interaction map improved the worst
  transient to `0.90 s` but still reached `8.19 m`. These variants are
  retained as failed architecture probes, not as capture candidates.
- **Reset-free planner-to-capture integration.** Added
  `scripts/evaluate_planner_capture_chain.py`. It replays the source route,
  constrained prefix, and constrained suffix, verifies the emitted handoff
  against the saved `qpos/qvel`, and switches to the capture actor without a
  reset. The `absolute004` chain had exactly `0` endpoint error and held
  centered whole-chain low momentum for `8.00 s`. Extending the same chain to
  `30 s` failed at `9.08 s` after reaching the `+/-12 m` rail. This is the
  first real seven-link swing-to-capture integration artifact, but it remains
  p0/long-rail discovery evidence.
- **Morphology transfer replay.** Replayed the same source/tail controls and
  capture actor at `plant_progress=0.0001` and `0.0025`. The smaller change
  produced `0.901` maximum position error and `3.757` maximum velocity error
  at the expected handoff with no capture; the `0.0025` run hit the `12 m`
  rail before handoff. The p0 route is not a transferable teacher by direct
  replay. Each homotopy step needs fresh planner labels and a capture policy
  trained on that step's actual states.
- **Finite-difference LQR residual curriculum.** Mapped the exact finite-
  difference LQR teacher into a CPU Torch residual policy and advanced the
  uniform seven-link plant through small, gated progress steps. The fine
  continuation passed each checkpoint from `p=0.905` through `p=0.995`, and
  the endpoint continuation passed `p=0.996` through `p=0.999`. The coarse
  `0.005` jump and the direct `p=1.0` stage failed, so the saved frontier is
  the `p=0.999` checkpoint rather than the last attempted update.
- **Residual-authority endpoint probe.** The `p=0.999` checkpoint failed at
  exact `p=1.0` with a `0.5` residual cap, and appeared to succeed with a
  `1.0` cap only because its reset still scaled both saved positions and
  velocities to `0.002`. Replaying the identical checkpoint from the actual
  saved `qpos/qvel` with both scales forced to `1.0` produced `0/1` success,
  a `0.02 s` upright streak, and a `13.063 m` rail violation. The apparent
  full-scale capture result is therefore rejected. The next capture run must
  train and evaluate with explicit unit state scales.
- **Discovery morphology conditioning.** The p0 base-heavy/damped plant is a
  useful route-discovery homotopy but a poor local capture teacher: its finite
  difference controllability test was numerically rank `10/16`, and tested
  DARE closures remained unstable or saturated. Do not infer canonical
  uniform-plant capture authority from p0 handoff success.
- **Energy-homotopy PPO.** A seven-link probe that combined the local residual
  teacher with energy-progress and energy-level rewards failed its first
  maintenance interface: `0/8` at the first two evaluations, maximum upright
  streak about `0.72 s`, and no stage advancement. This is a failed
  parameterization/interface test, not evidence that energy shaping is useless.
  It must be retried only after the swing objective can hand a controlled state
  to the maintenance expert.
- **Terminal-rate-aware exact handoff shooting.** Added
  `scripts/search_exact_handoff_multiple_shooting.py` and extended the sparse
  shooting Jacobian to support extra terminal residual rows for cumulative
  absolute link rates. From the real final-plant `0.149 rad` handoff, a
  canonical-rail run improved the best transient absolute-rate RMS only to
  about `1.84 rad/s` before a `3 m` rail violation. A `13 m` rail diagnostic
  reached about `1.83 rad/s` but also rail-clamped; heavier absolute-rate
  weighting did not produce a quiet endpoint. The tail optimizer is now a
  valid diagnostic, but this handoff remains outside the tested capture basin.
- **Serial exact global CEM with absolute-rate scoring.** Warm-started from a
  replayable long-rail force waveform and ran a 12-second, 48-knot exact
  MuJoCo search with cumulative absolute-rate weight `55`. The best late
  candidate reached only `1.46 rad` angle, `2.28 rad/s` relative hinge RMS,
  and `4.98 rad/s` absolute-rate RMS. A separate 20-second attempt was
  rejected immediately because its warm start hit the temporary rail before
  the requested horizon. This rules out counting the old force-knot route as
  a quiet handoff and keeps the next lever on phase-aware feedback or a
  stronger trajectory optimizer.
- **Six-link calibration.** Near-upright six-link stabilization and local
  capture work are useful for debugging. They do not establish a seven-link
  swing-up and must not be used to inflate the seven-link score.

## Active Levers

1. **Residual capture training:** train only the correction around the MPC
   teacher; retain the teacher checkpoint as the rollback baseline.
2. **Hard-negative recovery:** oversample the exact saved state indices that
   lose the five-second hold, then validate on untouched state indices.
3. **Nested reset curriculum:** exact upright, tiny real handoff states,
   measured-velocity handoffs, synthetic angle/velocity perturbations, and
   only then the full capture envelope. Advance only on repeated gates.
4. **Capture-value swing objective:** score swing trajectories by uninterrupted
   replay under the capture expert, with penalties for hinge momentum, cart
   excursion, force saturation, and rail use.
5. **Rail-length homotopy:** sweep rail length against total chain length and
   record the least rail that permits a low-momentum one-motion swing. Keep
   shorter-rail back-and-forth excitation as a separate mechanism, not a hidden
   benchmark change.
6. **Gradient-to-uniform discovery:** use morphology gradients only to discover
   trajectories or initialize teachers; anneal each gradient independently and
   log where the route collapses.
7. **Planner/learner loop:** collect learner-visited failures, ask a bounded
   exact-MuJoCo planner for a reset-free recovery or swing prefix, retain only
   successful labels, and retrain on the aggregated state distribution.
8. **Link-count transfer:** after a seven-link route is real, test split-link
   homotopies and an `n`-conditioned policy to pursue eight or more links.
9. **Measured capture-basin expansion:** train the nonlinear capture expert on
   the exact absolute-rate-constrained handoff set, beginning with the quiet
   `absolute004` state and adding neighboring and hard-negative states without
   overwriting the incumbent. Rank by worst-case centered, whole-chain
   low-momentum hold, not by the best single state.
10. **Durable post-capture stabilization:** treat the first five seconds after
    handoff as a separate maintenance interface. Use the measured state at the
    end of the capture transient to train or synthesize a long-horizon
    stabilizer, and validate its 30-second rail-safe hold before using it as
    the swing planner's value function.

## Pitfalls That Invalidate A Claim

- Starting upright and holding is maintenance evidence, never swing-up
  evidence.
- Starting from a saved handoff state is a component test, never final
  hanging-start evidence.
- A state-list reset with `init_qpos_scale` or `init_qvel_scale` below `1.0`
  is a training-wheel evaluation. Final handoff evidence must replay the
  recorded `qpos` and `qvel` at unit scale with zero reset noise.
- A curriculum progress value, widened rail, altered morphology, extra
  damping, or selected seed must be labeled as development conditions.
- A reset, state overwrite, simulator restart, or hidden state replacement
  during the swing-to-capture switch invalidates reset-free evidence.
- A first upright crossing is not capture; require the configured low-momentum
  sustained hold.
- Relative hinge-rate RMS is not a sufficient low-momentum test for a serial
  chain. Record cumulative absolute link rates and require the declared
  whole-chain bound for handoff labels.
- A single-state capture actor that holds one exact state can be an informative
  local funnel, but it is not a capture expert until it survives a held-out
  neighborhood or state-set evaluation.
- A planner objective, optimizer convergence, or shaped return is not a
  successful trajectory until exact MuJoCo replay passes.
- A batched candidate that reaches the rail is not a valid route merely because
  its rollout state later returns toward the center; batched MuJoCo can bypass
  the environment termination check and let joint-limit dynamics fabricate
  that return. Reject rail contact and replay the candidate step by step.
- A saved tail handoff is only reproducible when source controller, morphology,
  rail, force limit, and every reset-noise schedule match exactly. A mismatch
  can turn the same JSON tail into a different physical state while appearing
  numerically plausible.
- Batched MuJoCo search must reject non-finite and huge states before sorting
  costs. Excessive force authority can otherwise create zero/NaN metrics that
  look like perfect candidates.
- Directly fine-tuning a stabilizing teacher can destroy the basin. Prefer a
  residual, trust-region, distillation-with-stability-check, or rollback-based
  update and compare every checkpoint to the incumbent.
- State-list evaluators must record the actual state indices, preserve the
  measured `qpos/qvel`, and use held-out indices for validation. Random
  replacement sampling can hide hard states.
- A short evaluator must explicitly override `env.episode_seconds`; otherwise
  source-state selection and success-window measurements can silently use a
  different horizon.
- A late near-upright state is not a useful handoff when any joint-rate mode
  remains energetic. The final-plant chain probe reached `0.129 rad` but still
  had `1.733 rad/s` relative hinge RMS; an LQR that saturated from that state
  was not a capture failure of the stabilizer alone, it was a failed swing
  handoff.
- A hybrid chain replay must preserve the capture switch, original seed, and
  reset-noise semantics. Replaying only the saved swing-stage cart trajectory
  produced a different state and invalidated the first warm-start comparison.
- A lower relative hinge RMS does not imply a quiet physical chain. The best
  angle-constrained final-plant handoff had `1.064 rad/s` relative hinge RMS
  but `2.496 rad/s` cumulative absolute link-rate RMS and yielded no
  absolute-low-momentum hold under the tested capture actors.
- The native MuJoCo/verifier test subset passes in the bundled CPU runtime,
  but the full `make roadmap-p0` discovery currently aborts when the
  MLX-dependent capture-envelope tests request a Metal device in this
  headless session. Keep that environment blocker separate from controller
  evidence and rerun the full suite on the configured MLX host.
- Final artifacts require resolved config, policy manifest, checkpoint hashes,
  per-episode metrics, reset-free video metadata, and fresh-clone replay.

Latest P5 two-expert follow-up (2026-09-11): a new discovery controller in
`scripts/search_pfl_capture_controller.py` combines the existing mass-matrix
cart-acceleration energy shaper with the exact seven-link LQR, gated by angle,
chain-rate, energy, cart-position, and cart-speed conditions. On a `+/-12 m`
development rail it reached no upright event and hit the rail; on a `+/-20 m`
diagnostic it improved the best transient angle to `0.592 rad` but still had
`0.0 s` upright streak and no capture-gate activation at the best point. A
separate CEM adaptation of the quiet `absolute004` maintenance actor, started
from the real `0.149 rad / 2.496 rad/s` cumulative-rate handoff, reached only
`0.02 s` upright streak and `0.0 s` low-momentum hold. This is the expected
failure mode for a maintenance basin used as an arrest expert: the project
still needs a capture policy trained on high-rate approach states, not another
quiet-state stabilizer.

Latest curriculum diagnostic (2026-09-11):
`runs/swingup7_swing_power2_trust_region` used the real upright maintenance
checkpoint as its initialization and a small residual action standard
deviation. It passed consecutive gates through `progress=0.040`, reached
`progress=0.050`, and then alternated between `1/1` and `0/1` eight-episode
evaluations without advancing. Because the quadratic schedule maps
`progress=0.050` to only `0.00785 rad` initial angle, the apparent success is
not a swing-up result. The earlier linear-angle continuation stalled at
`progress=0.005`. This experiment worked as a trust-region maintenance
diagnostic but failed as a route to swing-up: the next branch must expose
larger-angle initial states while retaining enough stochastic exploration or
an explicit swing/energy teacher. Do not count these curriculum gates in the
seven-link scoreboard.

Latest capture-interface diagnostic (2026-09-11):
`runs/swingup7_high_rate_capture_state_set/` contains 16 measured MuJoCo
states sampled from the absolute-rate approach route. They are development
states from a widened `+/-20 m` rail, not canonical evidence. A shared
linear/tanh actor optimized across the full set reduced the CEM score from
about `61,653` to `14,045`, but held `0/16` states, reached `0.0 s`
absolute-low-momentum streak, and repeatedly hit the canonical `+/-3 m` rail.
The failure is informative: a single low-capacity law cannot yet cover the
mixed phase/rate distribution, and the quiet maintenance actor is not an
arrest expert. The next test isolates the closest measured handoff and
compares open-loop feasibility with a richer rate/cart-interaction actor.

Latest exploration diagnostic (2026-09-11):
`runs/swingup7_swing_exploration_linear` started from a maintenance checkpoint
with action standard deviation `0.35` and linear curriculum progress `0.005`.
After roughly `0.67M` environment steps, deterministic evaluation remained
`0/8` even though shaped reward improved. More PPO noise without a swing or
energy teacher did not leave the maintenance basin. Do not interpret the
training reward or the near-upright curriculum work as swing-up evidence.

Latest capture feasibility probes (2026-09-11): the closest measured
high-rate state was tested with four local controllers. An exact open-loop
8-second CEM reached only a `0.234 rad` transient for `0.02 s` before a rail
violation. A fixed arrest-to-stabilize affine pair also stayed at `0.02 s`.
The richer angle-rate/cart-interaction actor stayed within `0.45 m` and
reduced relative hinge RMS to `2.75 rad/s`, but cumulative absolute-rate RMS
was still `6.31 rad/s` and the low-momentum streak was `0.0 s`. Exact iLQR
ran 50 iterations and replayed into a rail violation. This rules out the
current handoff as a useful capture starting point, rather than proving that
the final plant cannot be captured.

The new `absolute_capture_ready` mode in
`scripts/search_swingup_action_sequence.py` scores cumulative absolute link
rate directly. The first 20-second direct-force CEM branch
(`runs/swingup7_action_cem_absolute_capture_ready.json`) produced no saved
handoff states and had a best late angle of about `0.84 rad`. Preserve the
objective for later trajectory searches, but reject this force-knot route.

The attempted 20-second serial global-CEM warm start was invalid because the
saved cart-target controller was not an equivalent force waveform for this
plant; replay hit the widened rail in about `1.5 s`, so all candidates were
`cost=inf`. The first rerun with a longer episode horizon exposed the same
controller/plant mismatch. This is a replayability pitfall, not evidence of
physical infeasibility.

Latest live-planner and rail-length diagnostic (2026-09-11): added
`scripts/search_swingup_online_mpc.py`, a low-dimensional receding-horizon
exact-MuJoCo CEM that replans from the state actually emitted by the previous
action. On the canonical `+/-3 m` rail it reached only `0.379` normalized
height before rail termination at `1.42 s`. With a `+/-12 m` discovery rail,
the unweighted objective reached `0.995` height proxy but used `6.875 m` of
cart travel and had `22.16 rad/s` cumulative absolute-rate RMS at its best
geometric crossing. A rate-limited continuation kept the cart excursion at
`2.943 m` and reached `0.984` height proxy, but the best angle was still
`0.465 rad` with `21.30 rad/s` absolute-rate RMS. Increasing capture weight
traded height for instability rather than producing a low-rate top state.
This supports a longer rail as a discovery aid, while showing that rail length
alone does not solve phase synchronization or capture. These runs are
development diagnostics only; no state was promoted to the canonical evidence
set.

Latest captureability-frontier diagnostic (2026-09-11): a batched full-state
CEM produced an apparently strong terminal proposal (`0.083 rad` maximum
angle, `0.096 rad` relative-angle RMS, `2.06 rad/s` cumulative absolute-rate
RMS), but exact serial replay diverged and invalidated it. This is a concrete
warning that batched seven-link proposals must always be replayed in the
authoritative MuJoCo evaluator before they enter the handoff set. Exact serial
CEM from the iLQR terminal improved the replayable terminal metrics to about
`0.091 rad` angle, `0.105 rad` relative-angle RMS, `0.60 rad/s` relative hinge
RMS, `0.60 rad/s` cumulative absolute-rate RMS, and `0.65 m` cart position,
but fixed-state iLQR capture, zero/LQR control, and receding linear MPC still
failed after roughly `0.08-0.10 s`.

Centered iLQR refinements then generated real, progressively quieter states.
The best ultraquiet state had approximately `0.0061 rad` maximum angle,
`0.0127 rad/s` relative hinge RMS, `0.0135 rad/s` cumulative absolute-rate
RMS, `0.132 m` cart position, and `0.015 m/s` cart speed. It still failed
zero-control, LQR/MPC, and focused PPO capture probes. An exact serial
8-second open-loop CEM from that state reached only a `0.24 s` upright streak.
The conclusion is sharper than the earlier low-momentum note: low momentum is
useful shaping and ranking information, but it is not a sufficient gate and
must not be used as a universal rejection rule. The authoritative gate is
closed-loop captureability from the exact saved state, followed by sustained
hold. Until a controller can capture a state, even a very quiet endpoint is a
diagnostic rather than a valid handoff.

Latest local-basin probe (2026-09-11): the exact ultraquiet state in
`runs/swingup7_ilqr_ultraquiet_a200_r500_state.json` was tested with the
linear-MPC teacher while independently scaling the saved cart/angle/velocity
displacement and the applied action. Exact upright held for the full `4.02 s`
probe. At `2%` state displacement and `2%` action authority, the best upright
streak was only `0.26 s`; most larger-action trials hit the `+/-3 m` rail and
the remaining low-authority trials fell. This establishes a narrow measured
local basin for the current teacher, not a low-momentum acceptance threshold.
The result says the capture expert needs phase/rate-aware nonlinear authority
and rail-aware basin expansion; it does not justify rejecting future swing
states solely for exceeding a nominal momentum bound.

Latest deliberately-absurd controller probe (2026-09-11): tested a
low-dimensional blind resonant family as the "stupid option." The chirped
square-wave metronome reached no upright interval and hit the canonical rail
at `0.42 s`; its best observed angle was `2.253 rad`. The follow-up blind
sinusoidal swing-set driver, with only cart-position/velocity centering and a
fixed late LQR switch, saw a `1.35 rad` search crossing but the saved
best-score replay only reached `1.612 rad`, `1.825 rad/s` hinge RMS, and
`3.002 m` cart excursion before rail termination at `3.34 s`. Giving it a
`+/-10 m` rail caused the cart to walk to `9.15 m` while the chain remained at
`1.776 rad`; no upright interval appeared. The result rejects blind periodic
forcing and rail length alone as solutions, while preserving them as useful
negative diagnostics. The exact serial target-tail CEM that followed from the
three-second iLQR prefix and a `0.01`-scaled quiet target ended at `1.068 rad`
maximum angle, `2.664 rad/s` absolute-rate RMS, and `1.104 m` cart position;
it also produced no upright interval. Artifacts:
`runs/swingup7_resonant_bangbang_cem.json`,
`runs/swingup7_swing_set_driver_cem.json`,
`runs/swingup7_swing_set_driver_rail10_cem.json`, and
`runs/swingup7_tail_target_cem_start3_scale001.json`.

Latest exact force-trace handoff audit (2026-09-11): the saved
`runs/swingup7_global_force_trace400_fullstate_refinement.json` candidate was
replayed through serial MuJoCo with `scripts/replay_global_force_trace.py`.
The replay matched its saved 4.56-second state to `9.4e-8 m` in qpos and
`4.6e-7 rad/s` in qvel, reaching `0.083 rad` maximum angle, but it still had
`2.06 rad/s` cumulative absolute-rate RMS and about `-0.99 m/s` cart speed.
It is therefore a real high-rate crossing, not a stable handoff.

The first downstream capture-value search was a false positive. Its capture
config contained `init_qpos_scale=0.002` and `init_qvel_scale=0.002` at
progress 1.0; `reset(options={qpos, qvel})` silently evaluated a nearly
upright, quiet state instead of the actual tail endpoint. The exact replay
then hit the canonical rail at `6.44 s` with only `0.06 s` upright streak.
`search_swingup_tail_capture_value.py` now restores qpos/qvel after reset so
future value searches score the physical state actually emitted by the tail.
The corrected 1.5-second tail search produced no five-second hold: it
converged toward the rail with roughly `0.44 rad` endpoint angle and `2.1
rad/s` hinge RMS. This is a useful evaluator fix and a negative result for
the current high-rate capture expert, not evidence against the two-expert
architecture.

Latest explicit morphology-gradient screen (2026-09-11): added `--progress`
to `scripts/search_swingup_global_cem.py` and
`scripts/search_swingup_global_exact_cem.py`, and saved the realized
morphology vectors in each proposal artifact. A serial exact energy/modal CEM
was then run from the true hanging state at p0, p0.25, p0.50, p0.75, and
p1.00. The rail was held at `+/-12 m` for this discovery-only comparison;
the horizon was `12 s`, with one search episode, two evaluation episodes,
12 iterations, and 24 candidates per iteration. Best evaluation angles were
`0.572`, `0.902`, `0.775`, `0.778`, and `0.931 rad`, respectively. Every
profile produced `0/2` upright evaluation episodes, `0 s` maximum sustained
upright streak, and no capture-quality handoff. The full p0 gradient remains
the best small-screen discovery profile, but the response is not monotonic:
the p0.25 and p0.75 stages are worse than neighboring stages. Treat this as
a curriculum seed and a morphology-conditioning result, not evidence for a
canonical swing-up.

Artifacts: `runs/swingup7_energy_modal_clean_cem24.json`,
`runs/swingup7_morphology_energy_p0250_rail12.json`,
`runs/swingup7_morphology_energy_p0500_rail12.json`,
`runs/swingup7_morphology_energy_p0750_rail12.json`, and
`runs/swingup7_morphology_energy_p1000_rail12.json`.

The first batched profile proposals are deliberately rejected: for p0, p0.25,
and p0.50, serial replay of the saved best waveform returned to near hanging
and disagreed with the batched saved state by up to roughly `25 m` in qpos.
Nonuniform batched MuJoCo proposals must not enter the handoff set without
serial replay. The next morphology experiment should generate fresh exact
planner labels at the p0 or p0.50 profile, train a capture policy on those
measured states, and anneal one morphology axis at a time toward uniform.

Latest fine gated curriculum diagnostic (2026-09-11): starting from the
resolved-config p0.020 frontier, the CPU Torch run with `0.0025` progress
steps passed repeated eight-episode gates through p0.0275. It then failed
five consecutive evaluations at p0.030 and was stopped at update `506`.
This is not swing-up evidence: `hanging_curriculum_power=2` makes the p0.0275
initial angle only `pi * 0.0275^2 = 0.00238 rad`, and `swingup_slow` still
uses the full p0 length/mass/damping/friction profile at that progress. The
run is retained as a narrow maintenance-prefix diagnostic; future morphology
claims must use true hanging starts and report the realized profile vectors.

Latest isolated-gradient and live-planner follow-up (2026-09-11): four
single-axis profiles were evaluated from the exact hanging state on a fixed
`+/-12 m` rail with the same energy/modal CEM. Length-only reached `0.849 rad`,
mass-only `1.106 rad`, damping-only `1.177 rad`, and friction-only `0.959 rad`;
all had `0/2` upright evaluation episodes and `0 s` sustained hold. A
receding-horizon exact-MuJoCo planner on the length-only profile reached a
`0.165 rad` crossing, but the state carried `25.95 rad/s` relative hinge RMS,
`21.98 rad/s` cumulative absolute-rate RMS, and `25.92 m/s` cart velocity.
The coupled p0 profile produced repeatable near-full-height crossings at
`0.373` and `0.423 rad` across two planner seeds, with no upright interval.
These results support morphology gradients as discovery/training conditions,
not as a substitute for the canonical uniform plant or a capture gate.

The live planner initially inherited the curriculum's maintenance residual
and was rejected because that teacher changed the applied action. The planner
now disables both residual and switch teachers internally. A subsequent
length-only tail CEM from the `0.165 rad` crossing was replayed exactly and
did not produce a quiet handoff. The full-gradient tail CEM from the `0.373
rad` crossing proposed `0.573 rad`, but exact replay measured `0.613 rad`,
`1.68 rad/s` relative hinge RMS, and no hold. Batched tail endpoints remain
proposals until `scripts/replay_swingup_tail.py` accepts their serial replay.

The direct Torch PPO hanging-start probe
`runs/swingup7_p0000_hanging_torch_probe_20260911` was stopped at update 84
after `688,128` environment steps. Its first evaluation at update 50 was
`0/8` success, `0/8` upright, and `0/8` capture, with the chain still at the
hanging angle. The configuration used the full p0 gradient, a `+/-12 m` rail,
true `init_mode=hanging`, and no maintenance teacher. Preserve the checkpoint
and log as a negative control; do not interpret the zero-motion PPO basin as
evidence that morphology conditioning cannot help.

Artifacts for this follow-up include
`runs/swingup7_morphology_length_only_energy_modal.json`,
`runs/swingup7_morphology_mass_only_energy_modal.json`,
`runs/swingup7_morphology_damping_only_energy_modal.json`,
`runs/swingup7_morphology_friction_only_energy_modal.json`,
`runs/swingup7_morphology_length_only_online_mpc_rail12.json`,
`runs/swingup7_morphology_p0000_online_mpc_seed20260920_rail12.json`,
`runs/swingup7_morphology_p0000_online_mpc_seed20260922_rail12.json`,
`runs/swingup7_morphology_p0000_tail_cem_from_seed20260920.replay.json`, and
`runs/swingup7_morphology_p0000_ilqr_seed20260920.json`.

An exact iLQR refinement initialized from the strongest p0 planner waveform
was also tested on a four-second `+/-12 m` horizon. It did not converge to a
useful terminal state: the serial endpoint was `1.809 rad` maximum angle,
`3.745 rad/s` relative hinge RMS, `2.927 rad/s` cumulative absolute-rate RMS,
and `1.018 m/s` cart speed. The new fixed-progress and top-level trajectory
input support is retained, but this local trajectory optimizer is rejected as
the current synchronization method.

Latest real-state capture handoff probe (2026-09-11): the p0 planner trace
`runs/swingup7_morphology_p0000_online_mpc_seed20260920_rail12.json` was
filtered by `scripts/extract_trajectory_states.py` into
`runs/swingup7_p0000_capture_states_from_real_swing.json`. This produced only
three actual visited states satisfying the development capture envelope:
`0.373`, `0.537`, and `0.600 rad` maximum angle, with cumulative
absolute-rate RMS values `11.69`, `12.42`, and `16.32 rad/s`. No synthetic
states were added. A fixed-p=0 Torch PPO capture probe used exactly those three
states, no residual teacher, no reset noise, a `+/-12 m` rail, and 250 gated
updates (`512,000` environment steps). Its final evaluation was `0/3` success,
`0/3` ever upright, and `0 s` maximum upright streak. The checkpoint is
`runs/swingup7_p0000_capture_from_real_swing_ppo` and remains a component
negative control.

A shared linear full-state CEM actor over the same three real states was then
searched for 21 serial iterations (4-second rollouts, population 48, 8
elites). It achieved `0/3` captures and no upright streak. The best individual
state-specific searches also produced no upright streak; their best transient
angles were `0.288`, `0.537`, and `0.289 rad`, still with `3.20-4.08 rad/s`
cumulative absolute-rate RMS and no five-second hold. Artifacts are
`runs/swingup7_p0000_capture_actor_from_real_swing_cem.json` and the three
`runs/swingup7_p0000_capture_actor_state*_cem.json` files. The conclusion is
that the current p0 swing expert is arriving too fast and too sparsely for the
capture expert, even on the easier morphology. The next lever is a measured
closed-loop capture-basin curriculum: first teach the capture expert from
nearby, lower-rate states, then train the swing expert against that expert's
value rather than accepting height or low momentum alone.

Two basin-bridge variants were then tested. A cold-start policy ramped the
three real states from `2%` to `100%` qpos/qvel displacement over a gated
curriculum, with p0 morphology fixed and a `+/-12 m` rail. After 300 updates
and `614,400` steps it remained at the `2%` gate; its best saved evaluation
was `0/3` success with `0 s` final upright hold, despite brief upright
intervals. A second run initialized from the preserved p0 maintenance
checkpoint and retained its residual teacher at the small-scale stages, then
faded that teacher toward zero. It still scored `0/3` at the first exact-state
evaluation and was stopped at update 223 after repeated failed gates. These
artifacts are `runs/swingup7_p0000_capture_basin_curriculum_ppo` and
`runs/swingup7_p0000_capture_basin_teacher_bridge_ppo`. The simple scale
curriculum and maintenance warm start are therefore insufficient; the next
capture design needs an explicit nonlinear phase/rate feedback representation
or a closed-loop value teacher that can shape the basin before the real
high-rate handoff is restored.

Latest terminal-state morphology curriculum probe (2026-09-12): the exact
quiet iLQR endpoint in `runs/swingup7_ilqr_terminal_state.json` was used as
the sole state-list handoff. The first continuation used the checked-in
`mass_last` schedule, so state scale and morphology changed together; its
best one-episode checkpoint reached `p=0.055` with an 8.02-second low-momentum
hold, but the run finished at `p=0.060` and never reached the full handoff.
The controlled follow-up held the complete p0 length/mass/damping/friction
profile fixed (`alpha_length=0.65`, `alpha_mass=5`, `alpha_damping=2.5`,
`total_damping=0.120`, `alpha_frictionloss=3`, `total_frictionloss=0.060`,
`+/-9 m` rail) while restoring the measured qpos/qvel from zero to one.
It advanced to `p=0.175`; its best checkpoint at `p=0.15` passed the single
training-seed diagnostic (`1/1`, 8.02 seconds, maximum angle `0.036 rad`,
capture quality `0.9976`, and maximum cart excursion `0.170 m`). A fresh
32-episode replay at the same stage was `0/32` success with a maximum streak
of `2.94 s`; a 32-episode zero-noise replay was also `0/32` and drifted to
the `+/-9 m` rail after a `2.38 s` streak. The apparent pass was therefore a
seed-specific narrow basin, not robust evidence that the gradient widens
capture. It did not reach the full state (`p=1.0`) or the uniform plant.
The gradient remains a valid training-wheel hypothesis, but this result
does not support the high-rate handoff yet.

The checked-in capture config now declares the qpos homotopy and enables
`rollback_on_eval_regression` so a stochastic PPO update cannot silently
replace the best policy at a mastered stage. The fixed-gradient artifact is
`runs/swingup7_capture_ilqr_terminal_fixed_gradient_state_probe`; the coupled
continuation is `runs/swingup7_capture_ilqr_terminal_gated_continuation`.
Both remain training diagnostics only. The next morphology experiment should
resume from the fixed-gradient frontier, finish the state homotopy at p0,
then anneal length, mass, damping, and friction one axis at a time toward the
uniform canonical plant, with exact unscaled replay after every stage.

The p0 local-teacher check was also negative. Finite-difference LQR on the
packet's normal p0 profile had closed-loop spectral radius `1.000535`; its
exact terminal replay reached the `+/-9 m` rail after `0.72 s` of transient
upright behavior. Increasing only the p0 total damping and friction-loss
training wheels to `0.5` each moved the spectral radius only to `1.000402`.
This is not evidence that friction is useless for the nonlinear swing search,
but it rules out the simple explanation that the current capture failure is
just missing linear damping. Artifacts are
`runs/swingup7_capture_ilqr_terminal_p0_lqr` and
`runs/swingup7_capture_ilqr_terminal_high_damping_lqr`.

## Entry Format For Future Runs

Every nontrivial run should add a row to the scoreboard or a dated entry below
with:

```text
Date / run directory:
Hypothesis:
Exact config and overrides:
Plant / rail / force / horizon:
Initial-state distribution:
Controller or teacher:
Episodes / seeds / state indices:
Result metrics:
Worked, partially worked, or failed:
Decision and next lever:
```

The most important rule is to preserve failed branches. A negative result is
useful only if the next experiment can tell what was changed and why.

## Exact-Start Seven-Link Route And Robustness Audit (2026-09-12)

The first complete seven-link swing-up and hold now exists as an exact-start,
reset-free MuJoCo artifact. The successful controller is
`runs/swingup7_fddp_full_hanging_ilqr_terminal100k_deferred_lqr1.json`.
It executes a 228-step Box-FDDP feedback route from the exact hanging state,
then switches to terminal LQR at the route horizon. The decisive fixes were:

- rebuild the iLQR warm-start qpos/qvel states into the FDDP evaluator's
  dimensionless absolute-coordinate convention;
- use a `100000` terminal state weight and a `2.80 m` internal rail soft limit
  so the route ends nearly quiet instead of merely crossing upright;
- defer the capture handoff until the 4.56 s route horizon;
- use LQR scale `1.0`; scale `0.5` failed on the same terminal route.

Exact replay metrics: first upright `4.54 s`, centered upright hold `25.48 s`,
low-momentum hold `25.46 s`, maximum absolute angle during the final hold
`4.21e-7 rad`, final hinge-velocity RMS `2.45e-9 rad/s`, maximum cart
excursion `2.3703 m`, zero resets, and time-limit termination with
`success=true`. The state-faithful video and metadata are
`runs/swingup7_exact_start_swingup_hold_2d.mp4` and
`runs/swingup7_exact_start_swingup_hold_2d.video.json`.

This is not canonical noisy-start evidence. The same controller's 20-episode
canonical hanging-start replay in
`runs/eval_swingup7_fddp_two_expert_canonical20.json` scored `0/20` success,
`0.05` ever-upright rate, and a maximum upright streak of `0.04 s`. The exact
route is therefore a valid proof that the seven-link problem is dynamically
reachable under the declared plant, and a useful teacher trajectory, but it is
not yet a competition-standard solution. The paper
`docs/seven_link_swingup_paper.md` records the result and the remaining
feedback-basin work without promoting the exact-start video to a record claim.

## Settled-Launch Canonical Seven-Link Gate (2026-09-12)

The wait-then-launch hypothesis produced the current strongest result. A
finite-difference LQR linearized at the exact hanging equilibrium runs for
`10.0 s` with a centered-cart reference. It damps the initial angle and
velocity noise while returning cart position and velocity toward zero. The
measured settled cart position then translates only the route's nominal cart
coordinate; qpos, qvel, and simulator time are never replaced.

The final chain uses:

- hanging-equilibrium LQR scale `1.0`, control cost `1000`, gain hash
  `48b0b5d607587774891f5d8b806924a43701a9127b5fe5487c80405a957ff083`;
- the 228-step, 4.56-second Box-FDDP route from
  `runs/swingup7_fddp_full_hanging_ilqr_terminal100k_deferred_lqr1.json`;
- swing-route tracking gain scale `2.0`;
- route nominal cart translation to the measured settled cart position;
- terminal upright LQR scale `1.0` after the route horizon.

With the canonical uniform 7-link config and noisy hanging starts, the final
frozen chain passes `20/20` and `100/100` episodes. Every episode reaches
upright at `14.54 s`, holds for `15.48 s`, terminates at the 30-second time
limit, and stays inside the `+/-3 m` rail; the worst 100-episode cart
excursion is `2.3708 m`. The final evidence is in
`runs/swingup7_uniform/eval_swingup7_20.json`,
`runs/swingup7_uniform/eval_swingup7_100.json`, and
`runs/swingup7_uniform/seven_link_swingup_manifest.json`.

The public held-out video uses seed `50732`, reports zero resets, and now fits
all seven links inside the frame. Its metadata is
`runs/swingup7_uniform/seven_link_swingup_success.video.json`; the video is
`runs/swingup7_uniform/seven_link_swingup_success.mp4`. This supersedes the
earlier exact-start-only video as the primary public artifact. The remaining
open items are fresh-clone reproduction and the earlier roadmap calibration
phases, not the canonical 7-link 20/100 control gates.

## Escalating Eight-Link Discovery (2026-09-12)

The roadmap now promotes the active frontier from seven to eight links after
the canonical seven-link control gate. The following exact-MuJoCo discovery
branches have been run and remain non-solution evidence:

| Branch | Setup | Best result | Decision |
|---|---|---|---|
| Padded FDDP, 4.56 s | Uniform 8 links, 7-link route padded with a duplicated terminal coordinate, 160 Box-FDDP iterations | `min_v=1040.82`, best angle about `0.854 rad`, rail violation at `3.022 m` | Failed; the 7-link timing does not transfer directly |
| Padded FDDP, 6.56 s | Uniform 8 links, 100 zero-control tail steps, 300 iterations | `min_v=1424.83`, no upright interval, rail `3.020 m` | Failed; extra tail alone is insufficient |
| Padded FDDP, 12.00 s | Uniform 8 links, 372 zero-control tail steps, 220 iterations | `min_v=226.19`, best angle `0.380 rad` at `2.06 s`, hinge RMS `8.83 rad/s`, rail violation at `3.005 m` | Best partial route so far, but not a quiet handoff |
| Exact global CEM | Uniform 8 links, 6.56 s, widened `+/-12 m` discovery rail, 40 knots, 32 candidates, 60 iterations | best late angle `1.364 rad`, hinge RMS `7.75 rad/s`, cart speed `2.98 m/s` | Failed; low-dimensional open-loop CEM did not reach the capture envelope |
| Exact global CEM | Uniform 8 links, 12 s, native 3 m rail | no valid complete candidates because the rail terminated the exploratory waveforms | Failed search setup; retained as a rail-budget warning |
| Energy/modal CEM | Uniform 8 links, widened `+/-12 m` rail, 20 s, 32-parameter controller | best angle `1.268 rad`, no upright event | Failed; modal family plateaued before capture |
| Energy-shaping feedback | Uniform 8 links, widened `+/-12 m` rail, 20 s, mass-matrix acceleration conversion | best angle `1.356 rad`, cart reached `12.01 m` and terminated | Failed; it transfers energy by spending the rail |

The gradient packet's idea is now represented by two named 8-link conditions.
`configs/swingup8_gradient_discovery.yaml` uses the packet's base-heavy,
base-damped, frictional morphology and a scheduled long rail for discovery.
The first run showed that directly reusing the uniform 7-link waveform is
invalid for that morphology: the warm start hit the widened rail in about
`1.06 s` while zero force remained stable.

`configs/swingup8_ghost_continuation.yaml` adds an explicit profile homotopy.
At progress 0, seven links have length `0.425714... m` and mass
`0.142842... kg`, while the eighth link retains length `0.02 m` and mass
`0.0001 kg`; at progress 1, all eight links are uniform. The new profile
support in `src/gcartpole/morphology.py` checks link count, positivity, and
exact total length/mass at every stage. The first ghost-profile FDDP attempt
still failed (`min_v=1109.55`, rail `3.022 m`), so smaller continuation steps
and a route that rephases the extra mode are still required.

These branches establish the current 8-link bottleneck: the system can reach
the upper neighborhood transiently, but the extra internal mode arrives with
too much hinge energy and consumes the rail before capture. The next allowed
lever is staged feedback homotopy from the explicit ghost profile, with exact
serial replay after every morphology step. No 8-link paper, video, or public
claim is created until the canonical 20/100 and reset-free video gates pass.

## Inherited Seven-Link Method Continuation (2026-09-12)

The latest campaign was narrowed to the method that actually passed seven
links. The new `scripts/force_proposal_to_fddp.py` adapter replays a saved
force route through the target-chain MuJoCo model and records dynamically
consistent target-chain states before Box-FDDP starts. This removes the
duplicated-coordinate warm-start artifact from the earlier padded runs.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| 7-route replay | Released 228-step seven-link controls replayed on uniform 8 links for 4.56 s | Best absolute-link angle about `0.688 rad`; replay cart excursion `2.124 m`; subsequent upright LQR hit the `3 m` rail | Failed; the predecessor waveform is not a target-chain route |
| State-consistent Box-FDDP | Exact 8-link replay warm start, 4.56 s, same terminal-state objective and time-varying feedback | Solver/live route remained outside capture (`min_v` about `1.89e6` after the unstable pass) | Failed; direct link increment is too large |
| Ghost-profile continuation | Seven-link route continued into the explicit eight-body ghost plant at progress `0.0` | FDDP reached terminal Lyapunov about `0.03`; terminal LQR still left the intermediate plant through the rail | Partial only; the ghost endpoint is not a usable capture expert |
| Fine ghost step | Progress `0.02`, gentle terminal-state homotopy, exact replay between stages | Terminal Lyapunov about `6.55`; progress `0.04` and `0.10` did not retain the capture neighborhood | Failed continuation; smaller steps or a better capture handoff are required |
| Base-heavy gradient FDDP | Predeclared gradient morphology at progress `0.0`, widened scheduled rail, exact seven-link route warm start, 120 Box-FDDP iterations | Nominal terminal Lyapunov about `63.98`; live feedback reached the `9.125 m` rail without capture | Failed; the gradient is a conditioning aid, not canonical evidence |
| Upright capture probe | Uniform 8 links, exact upright plus a `0.001 rad` link-3 perturbation, ordinary LQR and finite-horizon FDDP feedback | Link-3-and-later internal-mode perturbations reached the rail; exact upright alone is misleadingly successful | Capture basin is the current blocker |
| Direct inherited FDDP from zero | Uniform 8 links, exact hanging state, 4.56 s route, scale-1 upright LQR, same terminal objective as the seven-link route, zero-force initialization | FDDP performed `0` iterations; live replay reached `3.051 m` and held `0 s` | A zero-force eight-link seed is not a useful replacement for the seven-link route |
| Stronger inherited route tracking | Uniform 8 links, dynamically consistent replay of the released seven-link controls, 4.56 s Box-FDDP, route-tracking gain `2.0`, scale-1 upright LQR | Solver stopped after `22` iterations; `min_v=2.25e6`, rail at `3.075 m`, hold `0 s` | More route feedback does not repair the extra mode |
| Softened inherited terminal objective | Uniform 8 links, same replay seed, route-tracking gain `0.5`, scale-1 upright LQR, terminal-state weight `1000`, cart terminal weights `50`, rail soft weight `1e7` | Solver stopped after `19` iterations; `min_v=2.06e6`, rail at `3.026 m`, hold `0 s` | Reducing terminal pressure only delays the rail exit |
| Better-conditioned ghost mass | Ghost continuation with the eighth link at `0.01 kg` instead of `0.0001 kg`, exact seven-link replay seed, 300 Box-FDDP iterations, scale-1 LQR | Replay stayed within `2.378 m`, but the optimized route diverged to `min_v=7.78e7` and rail `3.008 m` | The singular tiny-mass ghost was not the only failure; this continuation stage still does not transfer |
| Better-conditioned ghost with soft terminal | Same `0.01 kg` ghost and exact replay seed, terminal-state weight `1000`, cart terminal weights `50`, rail soft weight `1e7` | Solver stopped after `73` iterations; `min_v=7.73e7`, rail `3.026 m`, hold `0 s` | Objective relaxation does not recover the route; stop tuning this branch and redesign the real low-momentum handoff |

The modal probe explains why the seven-link recipe cannot simply be copied:
the uniform eight-link plant is substantially more ill-conditioned in its
single-cart controllability directions. A zero-noise upright origin is not a
capture result. The next permitted work is therefore still the same two-expert
chain: save real low-momentum terminal states from the eight-link FDDP swing
route, train or optimize capture on those states and their measured internal
modes, then replay the complete chain on the canonical `+/-3 m` rail. Broad
PPO/global-CEM exploration is deliberately paused; it does not answer the
current inherited-method question.

## P1 Full-Envelope Capture Boundary (2026-09-12)

The first unmet roadmap gate was re-audited against the frozen six-link
synthetic envelope rather than scaled curriculum checkpoints. These are
development diagnostics only; none is P1 evidence.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| Exact-state LQR scale sweep | First 64 frozen P1 test states, progress `1.0`, residual and switch disabled, scales `0.25` through `3.0` | Every scale: `0/64` success, `64/64` rail, median hold `0.04 s` | Full envelope is outside direct upright-LQR nonlinear basin |
| Exact-state LQR control-cost sweep | Same states and plant, control costs `0.001` through `1000` | Every cost: `0/64`, `64/64` rail; the highest-cost case still rails in about `0.04 s` | Gain tuning alone is not the fix |
| Full linear PPO curriculum | Existing capture supervisor checkpoint, same state features and LQR switch, linear progress toward `1.0` | Best saved checkpoint at progress `0.05956`: `56/64 = 87.5%`; later checks fell to `28%` at `0.122` and `9%` at `0.185` | Linear exposure is too aggressive |
| Gated continuation | Supervisor checkpoint, `0.01` progress increments, held-out check every 10 updates | `0.070`: `50/64 = 78.125%`, advanced to `0.080`; `0.080`: `59%` at updates 20 and 40, no next frontier | Reproducible boundary is between `0.07` and `0.08` |

Artifacts: `runs/p1_capture_full_curriculum_probe/` and
`runs/p1_capture_gated_continuation_probe/`. The next P1 experiment must
address high internal-rate modes with a reusable state-feedback recovery
teacher and then rerun the same full 1,000-state gate; no rail or benchmark
threshold is relaxed.

## Inherited Eight-Link Route Boundary (2026-09-12)

The eight-link campaign stayed on the seven-link recipe: an exact-MuJoCo
Box-FDDP swing route followed by a terminal capture expert. These runs are
all genuine uniform-eight diagnostics or measured handoff states; none is
canonical eight-link evidence.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| Wider-rail time stretch | Seven-link force route retimed to `5.0 s`, replayed and optimized on a `+/-12 m` discovery rail | FDDP stopped at `min_v=1613.96` and `12.073 m`; stronger rail shaping stopped at `min_v=416.87` and `12.046 m`; canonical replays hit `3.014 m` and `3.098 m` with no hold | A longer rail alone does not create a capture route |
| Exact canonical route, feasible FDDP | Dynamically consistent 8-link replay of the 4.56-second released force route, feasible Box-FDDP, canonical rail | `min_v=374.37`, live rail `3.069 m`, hold `0 s` | Direct link-count continuation remains outside the basin |
| Open-loop route | Same exact replay, zero trajectory feedback, handoff deferred to the route horizon | Best transient angle `0.243 rad` at `4.44 s`; at `4.56 s`, angle `1.122 rad`, hinge RMS `2.25 rad/s`, `x=2.807 m`; live rail `3.027 m` | Feedback amplification is not the only blocker; the cart arrives too close to the rail |
| Repeated windup | Two and three repetitions of the released seven-link force cycle, then FDDP on canonical rail | Live routes reached `3.050 m` and `3.148 m` with `0 s` hold | Repeating the proven swing does not rephase the eighth mode |
| Measured two-expert handoffs | Capture-only Box-FDDP from measured upper-neighborhood states: `(0.671 rad, 18.23 rad/s)`, `(0.898 rad, 6.53 rad/s)`, and `(0.243 rad, 8.45 rad/s)` | Canonical or discovery-rail tails all failed; the near-top tail reached `4.589 m` on a `4.5 m` rail | Capture must receive a substantially quieter and more centered state |
| Terminal momentum homotopy | Added configurable terminal cart/hinge velocity factors to `scripts/search_fddp_capture.py`; continued factors `20 -> 5 -> 1` from the exact route | Handoff moved from `(1.672 rad, 1.63 rad/s)` to `(1.498 rad, 1.35 rad/s)` and then `(1.506 rad, 1.41 rad/s)`, but never entered capture; an exact tail extension also failed at `3.011 m` | Low momentum is necessary but not sufficient; the remaining angle must be removed without spending the rail |

The new terminal velocity factors default to `1.0`, preserving prior behavior.
The focused FDDP and roadmap tests pass (`11 passed`). The reproducible
frontier is therefore a low-momentum but still non-upright terminal state,
not an eight-link solution. The next inherited-method experiment should
continue from this route family with a longer swing-and-brake trajectory or a
capture teacher trained on these measured internal modes; broad policy or
global-CEM exploration remains out of scope.

## Inherited Eight-Link Capture-Tail Continuation (2026-09-12)

The next campaign kept the seven-link decomposition intact: replay the
measured eight-link swing prefix, optimize only a downstream arrest tail, then
hand the saved state to a separate capture expert. The prefix was fixed at
`3.50 s`; the tail used exact serial MuJoCo and a canonical `3.0 m` rail as a
soft objective while a `12.0 m` rail prevented premature discovery
termination. None of these runs is canonical eight-link evidence.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| Late crossing tail | Prefix `4.44 s`, zero-force CEM tail, `2.0 s`, widened rail | All candidates hit the temporary rail; no finite population was available to rank | A late upright crossing has too little braking authority |
| Target-directed tail | Prefix `3.50 s`, 32 knots, 64 candidates, 45 iterations, canonical-rail soft penalty | Best terminal angle `0.534 rad`, cart position `1.396 m`, terminal hinge RMS `3.54 rad/s`, maximum cart excursion `2.16 m` | Best inherited 8-link warm start, but not a capture handoff |
| Target-tail rate homotopy | Same warm start, rate weight increased to `1,000,000` | Rate RMS reached `1.37 rad/s` only while terminal angle degraded to `2.55 rad` | Lower momentum alone does not recover the upright neighborhood |
| Long tail | Same prefix, `10.0 s` tail, same rail penalty | Best terminal angle `1.26 rad`, rate RMS `1.97 rad/s` | More time without a capture-aware objective does not solve the mode |
| Constrained tail shooting | Seeded by the target-directed tail, explicit angle/rate/cart constraints | SLSQP stopped at the initial feasible warm start with `evaluations=1`; no refinement was made | Retain as a solver/Jacobian negative control |
| iLQR tail refinement | Seeded by the canonical-rail tail, exact MuJoCo, `3.0 m` rail | Numerical instability; terminal angle `2.76 rad`, no hold | Local iLQR refinement is not robust for this 8-link tail |
| Full stitched Box-FDDP | Dynamically consistent 8-link replay of the rail-biased tail, then FDDP | Diverged to terminal value `2.8e10` and a `12.03 m` discovery-rail exit | Do not optimize the entire route from this seed again without a new capture-aware terminal metric |
| Separate capture expert | FDDP from the measured best tail state, canonical rail | Diverged to a `3.013 m` rail exit with `0 s` hold | The current handoff is still outside the capture basin |
| Padded predecessor feedback | Seven-link released feedback gains padded to eight links and replayed | `0/1` success, `3.045 m` maximum cart excursion | Extra-mode feedback cannot be copied unchanged |

The best artifact is
`runs/swingup8_inherited_early_tail_cem_canonicalrail.json`; its dynamically
consistent FDDP warm-start replay is
`runs/swingup8_inherited_early_tail_fddp_warmstart_v2.json`. The proposal
adapter now accepts stitched controls and records the terminal physical state,
and the tail CEM can seed a new run from a prior best knot vector. The next
permitted inherited-method test is to score candidate tails by an actual
post-tail capture/LQR replay, using this rail-safe warm start; no new global
policy or morphology search is justified by the current evidence.

## Inherited Eight-Link Capture-Value Boundary (2026-09-12)

The requested continuation stayed on the seven-link method: inherited swing
prefix, exact-MuJoCo arrest tail, then a separate capture expert. The new
capture-value path was tested with the same upright LQR used by the released
seven-link controller, first at the tail endpoint and then at several real
states along each candidate tail. None of these artifacts is eight-link
evidence.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| LQR-valued endpoint tail | Uniform 8 links, fixed 3.50 s inherited prefix, 5 s tail, 5 s downstream LQR, canonical rail penalty | Initial CEM had no hold; best endpoint was `1.489 rad`, `4.10 rad/s` hinge RMS, `2.47 m` cart position, and `4.18 m/s` cart velocity | Capture value improves ranking but does not create a handoff |
| LQR-valued local refinement | Same route, seeded from the prior tail, tighter CEM and heavier downstream value | Best endpoint angle `0.481 rad`, hinge RMS `7.27 rad/s`, cart `2.66 m`, cart velocity `6.33 m/s`; downstream LQR streak `0.00 s` | Low angle is a high-momentum false positive |
| Real-state LQR window | Existing multi-state tail evaluator, LQR replay from actual serialized states along each tail | Twelve diagnostic iterations produced no upright streak; best logged state was about `0.735 rad` at `3.12 s` and reached the canonical rail | Keep the window evaluator, but replace the capture expert before spending a long budget |
| Long capture FDDP | Exact measured best-tail state, 10 s Box-FDDP capture horizon, canonical rail | `min_v=388.15`, terminal value `9846.94`, live cart `3.006 m`, hold `0.00 s` | More capture time without a better basin does not solve the extra mode |
| Switch/scale sweep | Inherited FDDP route, LQR switch times `2.0-4.56 s`, scales `0.25-2.0`, rails `3-12 m` | Every case had no upright interval and terminated at its rail | Timing and rail width are not the missing capture mechanism |
| Protected-capture prerequisite probe | 8-link upright maintenance PPO from the seven-link maintenance recipe, uniform plant, p0 gated stage | Repeated p0 checks stayed at `0/8`; the stopped run reached update `199` and its best deterministic replay drove to the `9 m` training rail | Reject unconstrained PPO; preserve a stabilizing teacher before widening internal-mode disturbances |

The multi-state evaluator initially used the old `score_batch` call
signature. It now supplies the explicit absolute-rate, rail, and endpoint
weights introduced by the newer tail scorer. The Torch runtime helper also
resolves the repository's handoff layout (`gradient_cartpole_handoff/.conda-aligator`)
before importing the system PyTorch, so the documented CPU training path loads
Torch, MuJoCo, and Gymnasium together. These are infrastructure fixes, not
control evidence.

The next inherited-method experiment is a protected capture expert trained on
real eight-link states and measured internal modes, with the stabilizing
teacher retained until each envelope gate passes. Reuse the existing
capture-value tail search only after that expert demonstrates recovery from
held-out states; do not create an 8-link paper, video, or record claim from
these endpoint or upright-origin diagnostics.

## Corrected Eight-Link Exact-State Boundary (2026-09-12)

The eight-link continuation stayed on the seven-link decomposition: inherited
swing route, capture expert, then upright hold. A validity audit found that
`scripts/search_capture_sequence.py::fixed_state_cfg()` cleared only the base
reset-noise fields. When a curriculum config supplied `*_start`/`*_end` fields,
the supposedly fixed state was replaced with random cart, angle, and velocity
noise. The helper now clears every reset-noise schedule and forces qpos/qvel
reset scales to one; `tests/test_fddp.py` locks this contract down.

| Attempt | Exact setup | Result | Decision |
| --- | --- | --- | --- |
| Corrected FDDP teacher | Uniform 8 links, exact `0.005 rad` link-3 or link-7 state, canonical rail, 5 s Box-FDDP | `0.10-0.12 s` transient upright streak, then rail violation | Not a capture teacher |
| Corrected capture CEM | Same exact states, LQR-seeded 51-knot action sequence, 12 CEM iterations | Best `0.06 s` upright streak; no five-second hold | Action-sequence search is insufficient |
| Corrected feedback MPC | Same exact link-3 state, 100-step horizon, nine replans, exact MuJoCo | `0.04 s` upright streak and rail violation | Short-horizon MPC does not recover the tail mode |
| Rich capture feedback CEM | Exact link-3 state, 66-feature angle/rate/cart interaction actor, seeded from the bounded actor, 15 iterations | `0/1` success, best `0.20 s` upright streak, `3.079 m` rail exit | Feature interactions do not yet supply the missing capture basin |
| Release-seeded ghost continuation | Frozen seven-link release route padded to eight, exact replay-built states, ghost profile at p0, `+/-9 m` discovery rail | `min_v=229.73`, no handoff, rail at `9.026 m` | Morphology continuation needs a nonlinear capture-compatible seed |
| Full-authority residual probe | Eight-link protected PPO, LQR residual authority `1.0`, action std `0.05`, p0 maintenance stage | p0 briefly passed at updates 25/50, then fell to `0/8` by update 75; stopped at update 125 | More residual authority destabilizes the teacher |

The older protected-capture artifacts are therefore not used as eight-link
evidence. The next run must first demonstrate held-out recovery from a measured
internal-mode envelope, then reuse the existing capture-valued tail search. No
canonical eight-link claim, video, weights, or record statement is justified by
these probes.

## Inherited Eight-Link Method Focus Boundary (2026-09-12)

The current eight-link work is deliberately staying on the released seven-link
recipe: hanging LQR conditioning, target-chain Box-FDDP swing-up, a real-state
capture expert, and terminal hold. These continuation checks changed only the
route or capture component inside that chain; they did not reopen the complete
controller search space.

| Attempt | How it was tried | Result | What it means |
|---|---|---|---|
| Shared internal-mode feedback CEM | One bounded nonlinear actor optimized over four exact single-link `0.005 rad` states, 8 s, 20 CEM iterations | `0/4` success; minimum upright streak `0.22-0.24 s`; maximum cart positions `9.02-9.11 m` | A single static local actor is not yet a reusable eight-link capture expert |
| Padded released route | Seven-link release controller padded to eight, exact uniform-eight replay, 10 s hanging LQR prelude | `0/1` success, no upright event, `3.031 m` rail exit at `13.42 s` | The seven-link waveform itself does not produce an eight-link handoff |
| Route timing/amplitude continuation | Same seven-link controls, retimed `0.60-1.80x` and scaled `0.60-1.40x`, `+/-12 m` diagnostic rail | No upright candidate; best late composite retained about `3.04 rad` maximum angle | Simple waveform retiming is not enough |
| Inherited terminal iLQR | Padded route refined on exact uniform 8 links toward upright terminal state, 6.56 s, canonical rail penalty | Terminal angle `2.902 rad`, hinge RMS `22.70 rad/s`, cart `3.059 m`; no handoff | The local trajectory optimizer needs a better capture-aware route seed |

Artifacts are local diagnostics: `runs/swingup8_capture_state_set_feedback_cem_exact.json`,
`runs/swingup8_release_padded100_direct_eval.json`, and
`runs/swingup8_inherited_ilqr_terminal_exact.json`. They are not eight-link
evidence. The next permitted experiment is a held-out nonlinear capture teacher
from measured target-chain handoffs, followed by the existing tail search scored
by that teacher. Broad PPO, global CEM, and unrelated controller families remain
out of scope until this inherited chain has been falsified.

## Split Eight-Link Continuation Boundary (2026-09-12)

Audit qualification: the `p=0.80` custom-gain hold needs a saved executable
controller/result bundle. Only hinge 8 is constrained, and deleting the finite
constraint at `p=1` introduces a dynamics jump. Progress values do not measure
physical unlocking or task completion. The statement below that every tail
reached the rail is incorrect: the zero-action and online-MPC probes lost
upright but reached time limits inside the rail. Current findings and repaired
code are documented in [the transfer review](transfer_review_2026_09_12.md).

The next experiments kept the seven-link decomposition intact: settle the
cart, execute a target-chain swing expert, hand off from a measured state, and
hold with a separate terminal expert. They used the split-link continuation
branch as a diagnostic only; none of these results is a uniform-eight claim.

| Attempt | Exact setup | Result | Decision |
| --- | --- | --- | --- |
| Locked split route | Eight-link split geometry, hinge 8 equality retained, exact MuJoCo Box-FDDP route plus terminal hold | `25.48 s` uninterrupted hold | Intermediate baseline with a constrained extra hinge |
| Unlock homotopy | Same route, equality locks and split parameters gradually released | Standard replay passed through `p=0.60`; a composite route with inherited feedback and reweighted terminal Riccati feedback held about `24.1-24.3 s` through `p=0.80` | The method transfers deep into the continuation, but this is still split geometry and a diagnostic handoff schedule |
| Terminal-state capture at `p=0.90` | Exact quiet endpoint; zero action, ordinary LQR, inherited FDDP feedback, custom Riccati feedback, and local iLQR tails | Tested tails lost upright; some hit the rail; local iLQR held `0.06 s` | Failure of these tested controllers; does not establish a universal capture boundary |
| Nonlinear receding-horizon tail | Exact `p=0.90` endpoint, MuJoCo rollout MPC, 50-step horizon, 75 replans | `0.06 s` upright streak, `0.04 s` low-momentum streak, then rail pressure | Bounded MPC did not repair the missing capture mode |
| Continuation FDDP | `p=0.80` warm start continued into `p=0.90`, 300 Box-FDDP iterations | `0.38 s` transient, cart `3.042 m`, rail violation | Warm-start continuation alone is insufficient |

The saved continuation artifacts are `runs/swingup8_split_unlock_p060_from058_fddp.json`,
`runs/swingup8_split_unlock_p090_composite_warmstart_v2.json`,
`runs/swingup8_split_unlock_p090_terminal_online_mpc.json`, and
`runs/swingup8_split_unlock_p090_from080_fddp.json`. The state loader now
accepts `--state-index terminal` for artifacts that serialize a real terminal
state, which makes the negative capture tests replayable without reconstructing
the preceding route. The next narrow step is corrected Box-FDDP transfer and
capture from measured handoffs under the reviewed method inheritance rule;
the final uniform-eight benchmark, video, weights, and paper remain blocked.

## Transfer Implementation Audit (2026-09-12)

See [the detailed review](transfer_review_2026_09_12.md) for the repaired
coordinate mapping, interval-action clock, inherited-success metadata, release
identity check, and MPC policy-step contracts. These bugs make affected older
negative experiments unsuitable for ruling out the inherited method.

| Attempt | How it was tried | Result | Decision |
| --- | --- | --- | --- |
| Seven-link reference regression | Frozen controller, full ten-second conditioning and scale-2 feedback, original 20 seeds | 20/20; 15.48 s mean hold; 2.371 m maximum excursion | Reference still reproduces; repeated seeds are regression evidence |
| Corrected direct transfer | Correctly padded gains and nominal states, uniform 8, identical phases, three development seeds | 0/3; 0 s hold; 3.039 m maximum excursion | Copying the route remains insufficient |
| Corrected clock and target-state FDDP | Exact 228 predecessor actions replayed on uniform 8, rebuilt states, 100-iteration cap | Warm start within 2.124 m; solver stopped at iteration 18 without convergence; live hold 0 s and cart 3.081 m | Further route optimization is required within the same architecture |
| Repaired nonlinear capture diagnostic | Same saved split `p=0.90` endpoint and seed, 50 policy-step horizon, 256 candidates, four iterations, three-second episode | 0.20 s upright streak, 0.16 s low-momentum streak, time-limit termination | Improves the old 0.06 s transient, still fails hold; does not rule out all MPC |

New artifacts use `runs/swingup7_review_*`, `runs/swingup8_review_*`, and
`runs/transfer_review_numerics.json`. No new canonical eight-link claim.

## Adaptive Eight-Link Curriculum And Warm-Start Boundary (2026-09-13)

The latest continuation stayed on the released seven-link architecture:
exact MuJoCo rollout, Box-FDDP trajectory feedback, a deferred capture/LQR
handoff, and sustained upright verification. The search implementation now
supports `--rebuild-initial-feedback`, which uses the predecessor controller's
saved feedback gains while rebuilding its warm-start states on the target
plant. This matters because open-loop control replay made tiny morphology
changes look like hard physical failures. The optional
`--initial-feedback-scale` is recorded in each artifact and is only a
warm-start diagnostic.

| Attempt | Exact setup | Result | Decision |
| --- | --- | --- | --- |
| Open-loop geometry continuation | Split-mass, locked-joint branch, 1.00% to 1.01% length progress, rebuilt controls only | Rail exit at about `3.06 m`; no hold | Open-loop warm starts are not reliable enough for morphology continuation |
| Feedback-preserving geometry continuation | Same branch, saved predecessor feedback used during rebuild | 1.01%, 1.02%, 1.025%, 1.035%, 1.04%, and 1.045% checkpoints held `25.54 s`; the ordinary objective reached a local boundary near 1.046% | Keep feedback-tracked rebuilds |
| Rail-aware locked continuation | Canonical rail, soft limit `2.5 m`, rail weight `5e6`, cart terminal weights `1000` | True 1.047%, 1.048%, and 1.049% geometry checkpoints held `25.56 s` at about `2.509 m`; 1.0495% failed | Rail margin is a useful search objective, but this remains split-mass/locked evidence |
| Rail-aware unlock continuation | Geometry fixed at 1.046%, split masses, final joint unlocked from 3.75% through 4.00% | 3.8%, 3.9%, and 4.0% held `25.54 s` at `2.509-2.513 m`; 4.2% failed; 4.1% held but emitted a transient optimizer QACC warning | Treat 4.0% as the clean current checkpoint until independent replay of 4.1% |
| Mass-only continuation | Split lengths and locked joint, uniform-mass interpolation | Existing 0.5% checkpoint passes; 0.51%, 0.55%, 0.625%, and 0.75% attempts failed or railed, including feedback-warm and 2x-warm-start probes | Mass is a sharper continuation axis than length in this branch |
| Geometry with joint already unlocked | Fixed 3.75% or 4.1% unlock while moving lengths further toward uniform | Failed near the first tested geometry step, including the stronger rail objective | Do not treat alternating one-axis curricula as sufficient; improve the capture-aware terminal objective |

Representative positive artifacts:
`runs/swingup8_locked_geometry_p01049_true_railtarget25.json`,
`runs/swingup8_geom_p01046_unlock_u004_from00395_railtarget25.json`, and
`runs/swingup8_geom_p01046_unlock_u0039_railtarget25.json`.
Representative negative controls include
`runs/swingup8_locked_geometry_p010495_true_railtarget25.json`,
`runs/swingup8_geom_p01046_unlock_u0042_railtarget25.json`, and
`runs/swingup8_uniform_locked_mass_only_p0051_feedbackwarm.json`.

All of these are one-seed curriculum diagnostics. They use nonuniform masses
and/or lengths and, for the unlock branch, a fractional constraint on the
eighth joint. None is a canonical uniform eight-link result. The required
20/100 held-out gates, reset-free video, public manifest, hashes, and paper
remain unearned.

## Adaptive Unlock Refinement (2026-09-13)

The same exact-MuJoCo two-expert route was continued from the nearest accepted
predecessor rather than replaying a coarse jump. The new warm-start rebuild
applied the predecessor feedback gains while rolling the route through the
target plant. A centered objective used a `2.4 m` soft rail limit, rail weight
`1e7`, and terminal cart and cart-velocity weights of `5000`.

| Attempt | Exact setup | Result | Decision |
| --- | --- | --- | --- |
| Unlock continuation `0.59` | Split lengths/masses, same route, feedback rebuild, rail-aware objective | `25.04 s` hold; max cart `2.363 m` | Accepted |
| Unlock continuation `0.60` | Same setup from the accepted `0.59` controller | `23.90 s` hold; max cart `2.362 m` | Accepted |
| Centered unlock continuation `0.603` | Same setup, tighter cart barrier and terminal centering | `25.02 s` hold; max cart `2.360 m` | Accepted |
| Centered unlock continuation `0.604` | Same setup from the nearest accepted predecessor | `25.02 s` hold; max cart `2.360 m` | Accepted |
| Soft-warm unlock continuation `0.6045` | Half-scale inherited feedback and final tracking, same centered objective | `25.02 s` hold; max cart `2.360 m` | Strongest accepted checkpoint |
| Boundary probes `0.60475` and `0.605` | Same soft-warm setup, next unlock steps | `0.42 s` transient, then `3.063 m` and `3.008 m` rail exits | Rejected |

The strongest artifact is
`runs/swingup8_split_unlock_p06045_softwarm_centered_railtarget24.json`; the
full-strength 60.45% probe
`runs/swingup8_split_unlock_p06045_centered_railtarget24.json` remains a
negative control, as do the rejected probes
`runs/swingup8_split_unlock_p060475_softwarm_centered_railtarget24.json` and
`runs/swingup8_split_unlock_p0605_softwarm_centered_railtarget24.json`. These are
curriculum evidence only: the final morphology remains nonuniform and the
eighth joint is not fully free. They do not support an eight-link record,
paper, video, or public release claim.

## Phase-Adaptive Unlock Replay (2026-09-13)

Fixed-step route replay became brittle at the first release of the eighth
joint. The evaluator now searches a bounded forward window of the inherited
nominal route in dimensionless state space, advances that phase monotonically,
and retains the saved time-varying feedback gains. The switch to the terminal
hold controller remains deferred until the inherited route horizon, so this is
still the same settled-launch two-expert interface used for seven links.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| Phase-adaptive split unlock `0.60475`, `0.605`, `0.610` | Source `p=0.6045`, exact hanging start, 20-step forward phase window | `25.02 s` holds, peak cart `2.360 m` | Accepted curriculum continuation |
| Phase-adaptive split unlock `0.615`, `0.6175` | Same route family, saved feedback replay | `25.48 s` holds, peak cart `2.360 m` | Strongest accepted replay |
| Phase-adaptive split unlock `0.61875`, `0.619375` | Same route family, saved feedback replay | `24.38 s` and `24.36 s` holds, peak cart `2.360 m` | Current accepted frontier |
| Phase-adaptive split unlock `0.620`, `0.630`, `0.650` | Nearest accepted route, exact target plant | `0.02 s` transient and rail exit near `3.1 m` | Rejected boundary probes |
| Canonical uniform target from `p=0.619375` | `configs/swingup8_uniform.yaml`, progress `1.0`, same saved route/gains | `0.02 s` transient, `3.085 m` peak cart, no latch | Canonical transfer failed |

The accepted artifacts are
`runs/swingup8_split_unlock_p060475_phaseadaptive_replay.json`,
`runs/swingup8_split_unlock_p0615_phaseadaptive_replay.json`,
`runs/swingup8_split_unlock_p0619375_phaseadaptive_replay.json`, and their
neighboring continuation probes. The `p=0.620` feedback-refinement probes
also failed: the zero-feedback rebuild reached only `0.42 s` upright streak
before a `3.031 m` rail exit, while the full-feedback rebuild failed earlier.
The current evidence therefore supports a sharp newly-freed-joint swing-up
boundary, not a capture/stabilization failure and not an eight-link claim.

The phase tracker includes a defensive end-of-route fallback so a morphology
jump cannot crash evaluation with an empty search window. This is an
implementation repair, not performance evidence.

## Prefix-Corrected Swing And Retuned Capture (2026-09-13)

The next continuation separated the two controller responsibilities at the
observed boundary. A short feedforward correction was applied only to the
opening swing prefix, while the saved phase-aware feedback route and deferred
handoff remained unchanged. The terminal capture LQR was then tuned from the
actual route handoff.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| Unlock `0.6200` | Original `p=0.619375` route, first `0.20 s` feedforward scaled `1.10x` | `23.44 s` hold; max cart `2.359 m` | Accepted |
| Unlock `0.6210` | Original route, first `0.18 s` scaled `1.02x` | `24.32 s` hold; max cart `2.360 m` | Accepted |
| Unlock `0.6220`-`0.6300` | Chained saved controllers with phase-adaptive replay | `24.22-24.30 s` holds; max cart `2.360 m` | Accepted curriculum chain |
| Unlock `0.6350` | Chained route, unchanged prefix | `23.48 s` hold; max cart `2.360 m` | Accepted |
| Unlock `0.6360` | Same route, LQR scale `1.0`, control cost `10000` | `24.06 s` hold; max cart `2.917 m` | Accepted |
| Unlock `0.6375` | LQR scale `0.6`, control cost `5000` | `24.06 s` hold; max cart `2.706 m` | Accepted |
| Unlock `0.6380` | LQR scale `1.0`, control cost `5000` | `24.06 s` hold; max cart `2.902 m` | Current accepted frontier |
| Unlock `0.6390` | Prefix/LQR grid, real-handoff MPC, and dedicated capture FDDP | Best transient `2.44 s`; MPC `0.10 s`; FDDP `0.18 s`; all rail-failed or unstable | Rejected boundary |

Representative artifacts include
`runs/swingup8_split_unlock_p062_prefix_s11_t02.json`,
`runs/swingup8_split_unlock_p0621_prefix_s102t018.json`,
`runs/swingup8_split_unlock_p0635_prefix_s1t018.json`,
`runs/swingup8_split_unlock_p0636_lqr_l1r10000.json`, and
`runs/swingup8_split_unlock_p0638_lqr_l1r5000.json`. The exact `p=0.639`
handoff was materialized in
`runs/swingup8_split_unlock_p0639_handoff_state.json` before the independent
MPC and Box-FDDP capture probes. This is the first direct test of a capture
expert trained from a real failed swing handoff at the current frontier; it
did not produce a reusable basin.

## Modal Capture And Micro-Step Morphology Continuation (2026-09-13)

### What worked

- Persisting the full LQR state-cost weights in each controller artifact made
  capture tuning reproducible. Moderate internal-mode penalties
  (`relative_angle=100`, `relative_angular_velocity=100`,
  `absolute_angular_velocity=10`, `control_cost=5000`) outperformed the
  earlier `relative_angle=1000` choice at a locked-morphology handoff.
- Reusing the successful seven-link pattern continued to work: saved
  time-varying swing feedback, phase-adaptive forward route matching, a real
  target-plant Box-FDDP refinement when replay failed, and a separate capture
  expert. This reached split/unlock progress `0.9000` with `25.50 s` hold and
  `2.413 m` peak cart excursion.
- The length/mass homotopy can move only in very small steps near its locked
  endpoint. Exact replay checkpoints through `0.0036` were successful after
  target-specific refinement; the `0.00365` checkpoint required changing the
  capture weights rather than increasing them.

### What failed or remains unsafe

- Removing the final split-joint lock at split/unlock progress `1.0` remains a
  failure even after a 300-iteration exact Box-FDDP attempt.
- Directly jumping the locked length/mass homotopy to `0.02` or `0.005`
  failed; the `0.02` 300-iteration refinement also failed and emitted MuJoCo
  instability warnings. These are not reasons to widen the rail or relax the
  canonical endpoint.
- A high relative-angle capture penalty was counterproductive at the
  `0.00365` handoff: the LQR saturated and exited the rail. A medium modal
  penalty held the same measured handoff for `27.66 s` in an isolated exact
  capture test and produced `23.10 s` in the full reset-free replay.
- The next micro-step, `0.00366`, is currently a measured boundary. Medium
  modal replay held only `0.72 s`; reusing the predecessor LQR linearization
  at several nearby `lqr_progress` values also railed, and a 200-iteration
  target-plant Box-FDDP refinement diverged before capture. This is evidence
  that the route and capture basin must be co-refined at the boundary.
- The corrected replay-only implementation now applies the requested
  inherited-feedback scale. Earlier artifacts whose metadata says scale zero
  but whose route used inherited gains are historical and must not be used as
  fresh reproducibility evidence.

### Current boundary

The strongest eight-link work remains curriculum evidence only. The current
uniform target has not passed a canonical hanging-start evaluation, and no
eight-link claim, video, or record comparison should be published until the
final morphology is uniform, the joint is free, and independent 20/100
episodes plus a no-reset video pass.

## Protected PPO Continuation Boundary (2026-09-13)

The existing six-link capture frontier was used to test the same
incumbent-preserving curriculum discipline applied to the seven-link release.
The inherited best checkpoint was retained separately from each training
stream, and later regressions were not promoted.

| Attempt | Setup | Result | Lesson |
| --- | --- | --- | --- |
| Standard PPO continuation | Resume `runs/swingup6_capture_envelope_allstates_gated_probe_75/checkpoints/frontier.safetensors`; 250 updates; progress step `0.0025` | `92.97%` at `p=0.0500`, `90.63%` at `p=0.0525`, then `75-83%` at `p=0.0550`; best artifact is the early protected checkpoint | A successful incumbent is destroyed when the next morphology stage is too hard |
| Low-rate protected continuation | Resume the protected `p=0.0525` checkpoint; 200 updates; learning rate `1e-5`; progress step `0.00125`; entropy `1e-4` | `90.63%` at `p=0.0525`; the next stage reached `88.28%` at best and ended at `85.16%`; no advance | Slower PPO is less destructive but does not cross the boundary |

Artifacts are `runs/swingup6_capture_envelope_resume_p005/` and
`runs/swingup6_capture_envelope_lowrate_p00525/`. These are diagnostic
six-link learner experiments, not P1 evidence: neither has the required
1,000-state held-out success, no-rail successful episodes, or complete
reusable-policy gate. They do establish a clean negative result against
continuing with more scalar PPO alone.

The next permitted continuation must preserve the existing capture expert as
a hard fallback on every rollout and train only a bounded residual, a
teacher-labeled correction, or an explicitly verified model-based recovery
policy. Any candidate must first beat the incumbent on held-out states, then
be replayed on the same frozen gate before the curriculum advances. This is
also the control rule for extending the 7-link method toward uniform 8 links.

## Eight-Link Micro-Step Boundary Probe (2026-09-13)

The accepted `p=0.00365` locked split-to-uniform checkpoint was replayed on
the same target plant at progressively smaller steps. The next point,
`p=0.003651`, failed before capture and reached the canonical rail at
`3.030 m`. The predecessor route failed identically. Additional probes found:

| Probe | Result | Lesson |
| --- | --- | --- |
| `p=0.00365025`, `0.00365050`, `0.00365075` | All replayed to a `3.030 m` rail failure | The boundary is real at sub-micro-step resolution |
| Prefix scale `0.98x`, `0.99x`, `1.01x`, `1.02x` for `0.20 s` | All failed before capture; `3.012-3.027 m` peak cart | Short launch retiming alone does not restore the route |
| Phase-adaptive matching disabled | `3.056 m` peak cart and no handoff | The phase matcher is not the sole cause |
| Target-plant Box-FDDP at `p=0.00365025` | `47` iterations, no upright interval, `3.067 m` peak cart | One-shot trajectory refinement cannot cross the cliff |
| Diagnostic rail widened to `+/-6 m` | Still no handoff; exit at `6.069 m` | The failure is not only the canonical rail limit |

Artifacts include
`runs/swingup8_uniform_locked_morphology_p0003651_modal_medium_capture_feedback.json`,
`runs/swingup8_uniform_locked_morphology_p000365025_modal_medium_fddp_refine.json`,
and the named prefix, no-adaptive, predecessor, and rail-6 probes. The
7-link method is therefore being applied faithfully, but its next transfer
requires co-refining the route and the capture basin with a hard incumbent
fallback; further isolated LQR gain sweeps would not address the measured
failure.

## Action Precision Is Part of the Controller Contract (2026-09-13)

A six-link tail search exposed a numerical trap that is negligible on easier
plants but decisive in a chaotic high-link trajectory. Batched MuJoCo rollout
was receiving float64 interpolated actions, while the public `env.step` path
casts each policy action to float32. A saved tail could therefore look useful
in the batch and follow a completely different serial trajectory.

The tail CEM now quantizes candidates to float32 before batched scoring, then
replays its winner independently through `env.step` and records termination
and physical metrics. The corrected n=6 rerun found no feasible handoff and
terminated at the 12 m diagnostic rail. General rule: action dtype,
interpolation grid, frame skip, and termination semantics belong in the
evidence contract; a batched optimizer result is only a proposal until the
ordinary controller path reproduces it.

## Uniform Capture Endpoint Audit (2026-09-13)

The protected six-link capture checkpoint was checked at the actual endpoint,
not only at its easier curriculum stage. On the frozen 1,000-state test split
it reached `974/1000` at `p=0.05`, with a `15.020 s` median hold and `26` rail
exits. The same checkpoint at exact uniform `p=1.0` reached `0/1000`, had a
`0.040 s` median hold, and exited the rail in all 1,000 episodes. This is a
negative P1 endpoint audit, not a six-link result.

The existing model-based/distillation candidates do not close that gap yet:

| Candidate | Exact endpoint probe | Result | Interpretation |
| --- | --- | --- | --- |
| Source-grouped supervisor distillation | 64 held-out states, uniform `p=1.0` | `0/64`, `0.040 s` median, `64` rail exits | Curriculum teacher does not transfer |
| Supervised LQR distillation | 64 held-out states, uniform `p=1.0` | `0/64`, `0.040 s` median, `64` rail exits | Static LQR imitation is insufficient |
| Trajectory-conditioned policy | 64 held-out states, uniform `p=1.0` | `0/64`, `0.040 s` median, `64` rail exits | Time/initial-state features do not create a basin |
| Time-conditioned policy | 64 held-out states, uniform `p=1.0` | `0/64`, `0.040 s` median, `64` rail exits | Same endpoint failure |
| Scheduled-target exact-MuJoCo planner | 16 uniform endpoint states | `0/16`, no recovery | Planner probe is diagnostic only |

The lesson is specific: the seven-link architecture is being reused, but the
capture expert must be learned from state-specific model-based teachers with
an incumbent fallback and then audited on held-out endpoint states. A good
`p=0.05` score, an isolated teacher, a widened rail, or a partial morphology
hold is not evidence for the canonical six-link gate or the eight-link claim.

## Bottom-Up Five-Link Promotion (2026-09-13)

The generalized track reached a five-link `20/20` noisy hanging-start gate.
The successful chain was deliberately mostly deterministic: a fixed-size
13-parameter PFL proposal, exact MuJoCo tail search, tail Box-FDDP, one final
full-horizon Box-FDDP refinement, Riccati capture, and the exact planar mirror.
The only per-launch selection was an exact-model comparison between those two
symmetry-related routes.

The decisive continuation variable was rail length. Searches constrained to a
3 m half-rail found near-upright states but no accepted robust capture. At a
4.5 m half-rail, the refined route passed all 20 uninterrupted episodes and
used at most 3.33175 m of cart-center travel. Including the 0.18 m cart
half-length gives a measured required ratio of 1.17058 for the 3 m chain. This
is a passing upper bound, not yet a minimum-rail certificate.

Phase-adaptive route skipping was actively harmful at five links. With a
six-step forward window, one of five initial probes failed; strict time-order
tracking passed all five over feedback scales from 0.5 through 1.25, then
passed the independent 20-seed gate at scale 1.0. Higher-link work should keep
the optimized route clock deterministic unless a phase change is proven safe
by a separate gate.

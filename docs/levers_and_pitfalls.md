# Levers And Pitfalls

This is the living experiment ledger and working playbook for the seven-link
and arbitrary-`n` program. It records what was tried, the exact mechanism, the
observed result, and the decision that follows. It is development evidence, not
benchmark completion evidence. Canonical claims still require the verifier and
artifacts listed in `ROADMAP.md`.

## Operating Goal

Solve the canonical uniform cart-pole at the active frontier, currently eleven
links, from the hanging initial distribution, swing it up, capture it, and
hold it upright for the required horizon. Promote one link at a time only
after the full evidence contract passes. The dependency order is:

1. Maintain an upright seven-link chain from exact upright and progressively
   larger disturbances.
2. Capture and recover from real `qpos/qvel` states emitted by the mastered
   maintenance policy.
3. Swing from the true hanging start into the measured capture basin and hold
   without resetting or overwriting simulator state.

The public six-link result is a calibration target, not the project endpoint.
The same evidence contract applies at every rung: hanging-start launch, saved
real-state feedback route, reset-free capture, noisy 20/100 gates, exact
replay, and inspectable video. The seven-link result remains the reference
release; ten links is the highest verified internal result, and eleven links is
the active experiment.

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

## Continuous Free-Joint Homotopy Boundary (2026-09-13)

The split-unlock curriculum contained a discrete mechanics change: for every
`p<1` the final equality constraint remains in the MuJoCo XML, while at
`p=1` it is removed. The accepted phase-adaptive replay therefore held
`25.50 s` at `p=0.999` but failed immediately at exact `p=1.0`. That result
cannot be described as an almost-solved uniform eight-link plant.

As a controlled bridge, the final joint was given an explicit spring and the
spring was reduced by incumbent-preserving exact Box-FDDP continuation. The
successful chain retained the 7-link method: settled launch, saved
time-varying swing feedback, phase-adaptive matching, separate capture/hold,
zero-scale inherited-feedback rebuild, and rollback on failed holds.

| Final-joint spring | Result | Status |
| ---: | --- | --- |
| `100` direct replay | `25.50 s` hold, `2.413 m` peak cart | Diagnostic bridge |
| `60`, `50`, `40`, `30`, `25` | `25.48-25.50 s` holds, `2.406-2.411 m` peak cart | Accepted continuation checkpoints |
| `22.5`, `22.4`, `22.35`, `22.3`, `22.25`, `22.2`, `22.0` | `25.48-25.52 s` holds, about `2.408-2.416 m` peak cart | Accepted fine continuation |
| `21.5`, `21.0`, `20.5`, `20.0`, `19.5`, `19.0`, `18.0`, `17.0`, `16.9` | `25.48 s` holds, `2.405-2.414 m` peak cart | Strongest current diagnostics |
| `16.8` | Immediate rail failure; no upright interval | Rejected boundary probe |

The decisive endpoint controls were negative. Direct replay of the best
spring-`16.9` controller under `configs/swingup8_uniform.yaml` held `0.00 s`
and exited at `3.010 m`; canonical zero-spring Box-FDDP refinement from the
same controller held `0.00 s` and exited at `3.081 m`. The nonzero spring is
causally supporting the current route.

The spring is nonzero in every success above, so these are not canonical
eight-link evidence. The next step is a free-joint capture-aware controller
that can cross the `16.8-16.9` basin boundary and continue to exactly zero
stiffness under `configs/swingup8_uniform.yaml`. Until then, the project is
closer to 8 than the earlier locked-joint results, but it has not solved 8.

### Fine Free-Joint and Rail Controls (2026-09-13)

The interpolated spring-`16.85` checkpoint did not cross the current free-joint
boundary. Direct replay of the spring-`16.9` incumbent failed, and a
target-specific exact Box-FDDP refinement stopped after 271 iterations with a
large terminal cost and `3.056 m` peak cart excursion. The same spring-`16.9`
controller replayed on the exact zero-spring uniform-eight plant with a `+/-6
m` diagnostic rail still produced no upright interval and walked to `6.019 m`.
The longer rail is therefore a diagnostic control, not a repair: the missing
piece is a free-joint capture transition.

The predecessor-transfer recipe was then re-run explicitly. A dimensionally
correct seven-link route was lifted with the existing
`scripts/pad_fddp_controller.py` contract, replayed on exact uniform eight,
and refined with target-plant Box-FDDP. The lifted replay exited at `3.077 m`
with no upright interval; 100 refinement iterations stopped after 31 solver
iterations and still exited at `3.037 m`. Repeating the same refinement on a
`+/-6 m` diagnostic rail exited at `6.005 m` with no upright interval. A
longer 8-second zero-control FDDP attempt from the exact symmetric hanging
state also produced no search direction and exited at `3.011 m`. These tests
show that the 7-link timing and padded feedback are not sufficient to discover
the extra mode, even when the rail is widened or the route horizon is longer.

The generalized six-link route was separately replayed on the frozen `+/-3 m`
rail. It scored `0/20`; all episodes exited the rail before capture, with
`3.0045 m` minimum peak cart-center travel and a maximum body-aware required
rail ratio of `1.0615`. The published generalized six-link promotion remains
valid only as a `+/-4 m` development-rail gate, not as the unfinished
canonical six-link calibration result.

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

The follow-on whole-route search exposed a second objective-design pitfall.
A smooth weighted sum lowered total cost but could exchange upright coherence
for lower cart or terminal velocity. The exact serial CEM now optionally uses
a capture-envelope barrier: it first minimizes explicit violations of angle,
hinge-rate, absolute-rate, cart-position, and cart-velocity limits at one
candidate handoff phase, then uses the smooth score as a tie-breaker. On n=6
this reduced violation from `8.7709` to `4.1654` and isolated `98.05%` of the
remaining modal energy in the first collective mode. Hard acceptance geometry
should be represented as constraints, not merely another weighted reward.

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

## Bottom-Up Six-Link Promotion (2026-09-13)

The morphology-derived n=6 modal route is now promoted through the generalized
development gate. The capture-barrier CEM alone remained outside the handoff
envelope, but it provided a useful nonlinear route on which the exact endpoint
map had full rank: all 14 terminal cart/link position and velocity coordinates
were locally controllable through 48 smooth action-correction knots.

A staged damped Gauss--Newton solve through the ordinary float32 `env.step`
path reduced the 4.18-second endpoint to `0.09335 rad` maximum angle,
`0.38612 rad/s` hinge-rate RMS, `0.73249 rad/s` absolute-rate RMS, `-0.52950 m`
cart position, and `0.30060 m/s` cart speed. This passes all five declared
handoff limits. A 60-iteration exact-MuJoCo Box-FDDP pass was feasible but hit
its iteration limit rather than reporting formal convergence; its resulting
time-varying feedback nevertheless replayed successfully and handed off to
upright LQR.

The packaged route and its exact planar mirror passed `20/20` independent,
uninterrupted noisy hanging-start episodes at scale-1 route feedback and no
phase skips. All 20 reached the time limit, each held upright for `26.06 s`,
maximum cart-center travel was `3.09881 m`, and the maximum body-aware required
rail ratio was `1.09294`. This is an upper bound for the tested controller on a
`+/-4.0 m` center rail, not a minimum-rail certificate. The n=1 through n=5
gates remain the regression set; the frozen n=7 route also passes the shared
evaluator `20/20` with its published 10-second conditioning and scale-2 route
feedback. The remaining research boundary is synthesis for arbitrary unequal
morphologies and n>=8, not whether the shared runtime can execute n=1...7.

## Eight-Link Settled-Launch Transfer Probe (2026-09-13)

The seven-link release's full three-stage architecture was tested on the exact
uniform eight-link plant: 10 seconds of hanging-equilibrium LQR conditioning,
the saved eight-link time-varying route, and terminal upright LQR capture. The
route was the dimensionally correct seven-to-eight padded/refined artifact, so
this was a transfer test rather than a shape-mismatched replay.

| Variant | Exact noisy replay | Lesson |
| --- | --- | --- |
| No conditioning prelude | `0/5`, `0.00 s` hold, `3.105 m` rail exit | Direct transfer remains a failure |
| 10-second hanging LQR prelude | `0/5`, `0.00 s` hold, `3.039 m` rail exit | Settling improves margin but does not create an upright interval |
| Prelude plus settled-cart nominal shift | `0/5`, `0.00 s` hold, `3.039 m` rail exit | Cart-reference alignment is not causal |

The exact evaluator artifacts are
`runs/swingup8_prelude_probe_baseline.json`,
`runs/swingup8_prelude_probe_10s.json`, and
`runs/swingup8_prelude_probe_10s_shift.json`. The seven-link conditioning
technique is therefore applied and verified as a negative transfer control;
the unresolved problem is the eighth free-joint swing/capture transition.

## Eight-Link Near-Upright Tail Probe (2026-09-13)

The best exact uniform-eight global-CEM approach was followed by an exact
closed-loop iLQR tail and then terminal LQR. The two tested tail placements
were:

| Tail | Replay result | Lesson |
| --- | --- | --- |
| `10.0 s` to `14.0 s` | `0.00 s` hold; `3.027 m` rail violation | A late capture tail cannot absorb the internal-link rates |
| `11.5 s` to `14.5 s` | `0.00 s` hold; `2.990 rad` terminal angle; rail violation | Delaying the tail makes the route less recoverable |

Artifacts:
`runs/swingup8_softcanonical_tail_ilqr_10to14.json` and
`runs/swingup8_softcanonical_tail_ilqr_11p5to14p5.json`. This supports a
specific design rule for eight links: the swing expert must deliver a
low-modal-velocity handoff; terminal LQR cannot be used as a substitute for
that handoff.

## Eight-Link Direct Modal Endpoint (2026-09-13)

The morphology-native route generator was also exercised directly at eight
links, independently of the seven-link transfer and spring-homotopy branches.
A deterministic normal-mode seed plus bounded exact-model residual search
reached the upright neighborhood. Staged endpoint Gauss--Newton then retained
rank 18 across every terminal coordinate and produced a route satisfying all
componentwise capture limits at `3.94 s`:

- maximum absolute angle `0.114742 rad`;
- hinge-rate RMS `0.243317 rad/s`;
- absolute-rate RMS `0.669792 rad/s`;
- cart position `-1.064930 m`;
- cart velocity `0.460546 m/s`;
- peak cart-center excursion `3.716165 m` on the diagnostic rail.

The route is preserved in
`runs/generalized_solver/n8_gn_stage13.json` with `not_solution: true`. It is a
stronger frontier than the earlier energy-only approaches because it passes
the declared componentwise handoff box through exact serial replay. It is not
a hold: several exact local-LQR weightings lasted only `0.08 s`, and a
Box-FDDP capture attempt reached `0.14 s` before rail violation.

Two endpoint objectives bracketed the final feasible tradeoff. Their exact
control sequences were convexly blended, then corrected with a very small
full-rank Gauss--Newton step. The refiner now implements this as
`--secondary-controller` and `--blend-alpha`, making the continuation
reproducible and useful for any link count. The larger lesson is that the old
componentwise handoff box does not approximate the shrinking nonlinear
invariant set closely enough at eight links. Future promotion should target a
morphology-conditioned terminal invariant-set metric, while retaining the
componentwise values as readable diagnostics.

## Exact Endpoint Feedback-Teacher Probe (2026-09-13)

The existing receding-horizon feedback-MPC teacher was moved to the actual
uniform six-link endpoint. Eight training states that had already failed at the
`p=0.07` curriculum stage were reset at `progress=1.0` and evaluated with the
canonical rail and terminal LQR fallback. The planner used 75-step horizons,
5-step replanning, four CEM iterations, and 128 candidates per iteration.

It produced `0/8` successful teachers. Each state reached only `0.02--0.06 s`
of upright streak before a rail violation, so the run yielded zero successful
feedback-MPC labels. The artifact is
`runs/p1_capture_feedback_mpc/train_teachers_endpoint8.json`; it is training
diagnostic evidence, not P1 evidence. This path cannot be rescued by larger
distillation cohorts: there are no exact-endpoint successes to imitate. The
next controller must change the reachable-set objective or feedback structure,
then be checked on held-out endpoint states.

## Eight-Link Online-MPC Endpoint Probe (2026-09-13)

The new reset-free online-MPC controller was tested from the exact endpoint of
the morphology-native eight-link route. It replanned every three policy steps
with a 1.2-second horizon, 128 candidates, and three CEM iterations, applying
each selected action through the ordinary serial `env.step` path. Its best
planned intermediate state reached `0.08297 rad` angle but retained
`3.1467 rad/s` absolute-rate RMS. The six-second live replay achieved only
`0.04 s` maximum upright streak, with no capture and `2.7935 m` maximum cart
excursion.

Artifact: `runs/swingup8_online_mpc_endpoint_probe.json`. This is stronger
negative evidence than a static LQR retry: feedback can rank a near endpoint,
but it does not keep the internal modes inside the nonlinear invariant set.
The next 8-link capture controller must optimize a terminal invariant-set
condition, not only endpoint distance.

A stricter online-MPC run on a `+/-12 m` diagnostic rail used a 2.0-second
horizon, 256 candidates, five CEM iterations, and two-step replanning. It
avoided rail termination but reached only `0.10 s` of upright streak before
drifting to `x=-4.451 m`, with final absolute-rate RMS `1.904 rad/s`. A
continuation using the upright discrete-LQR value matrix as the iLQR terminal
cost was worse: `0.251 rad` terminal angle, `3.426 rad/s` hinge RMS, and
`4.562 m/s` cart speed. Both branches remain diagnostics and do not count as
canonical evidence.

## Six-Link Wide-Rail Capture Curriculum Probe (2026-09-13)

To separate rail starvation from policy failure, a gated PPO curriculum used a
rail scheduled from `+/-12 m` at the initial stage to the canonical `+/-3 m` at
the endpoint. The state envelope, LQR residual, observation contract, and
strict evaluation gates were unchanged. It passed `p=0.025` at `128/128` and
`p=0.05` at `127/128` (`99.22%`), then advanced to `p=0.075`.

At `p=0.075`, with the rail still `+/-11.325 m`, the first evaluation scored
`95/128` (`74.22%`), and later evaluations stabilized at `94/128` (`73.44%`)
through update 175. It never advanced. The wide rail removes early rail
termination as the dominant explanation; the remaining failure is reusable
internal-mode recovery. The training log and checkpoints are in
`runs/swingup6_capture_rail_curriculum_probe/`; this run is diagnostic only
and does not satisfy P1.

## Saturation-Aware Terminal Geometry (2026-09-13)

The shared componentwise handoff box hid a severe link-count effect: the
eight-link endpoint inside that box still requested `-309.54` normalized
action from the exact LQR. The solver now constructs a dimensionless discrete
Lyapunov matrix and computes the largest sublevel whose linear feedback stays
inside the actuator limit. This set is certified only for the linearization;
exact nonlinear replay may shrink it and remains mandatory.

Direct endpoint continuation reduced the normalized invariant value from
`1.2138e8` to `16.30` and the raw handoff action to `0.134`, while retaining
rank 18 endpoint sensitivity. The exact nonlinear controller still diverged.
A short exact-feedback rollout was therefore added to the optimization
residual; its first probe reduced mean rollout cost 57% but did not yet change
the endpoint enough to hold. The practical lesson is to optimize the feedback
trajectory, not merely a readable terminal box or a linear quadratic value.

## Eight-Link Parked-Route Feedback Promotion (2026-09-13)

The eight-link result was obtained by applying the seven-link stack in a more
careful form, not by replacing the project with a broad search. The route uses
the exact uniform eight-link plant and the canonical `+/-3 m` rail.

### What was tried

| Lever | Result | Interpretation |
|---|---|---|
| Seven-link route transfer with 10 s hanging LQR | `0/5`; no hold; rail exit | The eighth internal mode breaks direct transfer |
| Open-loop eight-link endpoint route after cart parking | Exact replay passed; noisy replay `0/20` | Cart translation alone is not robust |
| Box-FDDP route with saved time-varying feedback | Noisy `20/20` and `100/100`; exact `20/20` | Positive controller ingredient |
| Parking at `-0.10 m` for `17.5 s` | Noisy `100/100`; max cart `2.9868 m` | Robust but wastes hold time |
| Parking at `-0.15 m` for `12.0 s` | Noisy `19/20` | Too close to the rail on the larger cohort |
| Parking at `-0.20 m` for `12.0 s` | Noisy `95/100` | More negative target does not replace settling time |
| Parking at `-0.15 m` for `14.0 s` | Noisy `100/100`; max cart `2.9300 m` | Promoted launch contract |

### Promoted controller

The runtime sequence is:

1. Hanging-equilibrium LQR parks the cart at `-0.15 m` for `14.0 s` while
   damping the noisy hanging chain.
2. A 197-step (`3.94 s`) saved Box-FDDP route applies dimensionless
   time-varying feedback at scale `0.75`.
3. The nominal cart coordinate is translated by the measured parked position.
4. Upright LQR scale `1.0` captures and maintains the chain around the same
   parked cart target, with no state reset.

The positive evidence is in:

- `runs/generalized_solver/n8_fddp_parked_target015_14s_noisy20.json`
- `runs/generalized_solver/n8_fddp_parked_target015_14s_noisy100.json`
- `runs/generalized_solver/n8_fddp_parked_target015_14s_exact20.json`
- `runs/generalized_solver/eight_link_swingup_success.video.json`
- `runs/generalized_solver/eight_link_swingup_manifest.json`

The 100 noisy episodes all ended at the time limit with zero rail failures.
The held-out video uses seed `90901`, reaches upright at `17.80 s`, holds for
`12.22 s`, and reports zero resets. The zoomed-out renderer fits the complete
chain in both hanging and upright poses so the visual evidence is inspectable.

### Why this counts and what it does not count

This is a canonical internal eight-link swing-up-and-hold promotion because
the final gate uses the uniform plant, canonical rail, noisy hanging start,
uninterrupted simulator state, and the same 20/100 evidence contract used for
seven links. The optimizer's wide-rail setting is not used as final evidence;
the full feedback controller is replayed on the canonical rail.

It is not an external world-record claim. The result is model-specific,
timing-specific, and not yet tested against link variation, actuator delay,
sensor noise, or a competition's exact rules. At the time of this eight-link
entry, the next active frontier was ten links; that historical handoff is now
complete. The current active frontier is eleven links.

## Nine-Link Continuation Diagnostics (2026-09-13)

The nine-link campaign began with the same architecture that promoted eight
links: quiet the hanging state, park the cart, execute one saved time-varying
feedback route, and use a terminal capture controller. The early transfer and
tail branches below were retained as negative controls before the successful
terminal-angle refinement.

| Lever | Result | Interpretation |
| --- | --- | --- |
| Direct eight-to-nine route transfer | `0/5`; route reached the rail near `3.00 m` | Adding one link without retiming does not preserve the handoff |
| Longer transferred route with a settling tail | No hold; terminal mode remained high | Extra time after the inherited route does not remove injected internal energy |
| Split-link continuation with a tiny locked tip | Negative; hard equality made FDDP ill-conditioned | A locked topology is not a dynamically identical lower-count plant in MuJoCo |
| Equal half-link split with exact kinematic embedding | Stayed synchronized through most of the swing, then missed late capture | Capsule inertia changes matter at the terminal handoff |
| Finite-spring tip-link curriculum | Tip stayed aligned, but the inherited first-eight route still missed capture | The missing ingredient is a low-modal-velocity nine-link handoff, not only tip angle |
| Exact serial modal CEM, `3.94 s` | Best last-link angle about `0.65 rad`; hinge RMS about `1.61 rad/s` | Modal shaping is a useful warm start, not a solution |
| Exact serial modal CEM, `6.0 s` | Best last-link angle about `0.61 rad`; hinge RMS about `2.24 rad/s`; diagnostic rail demand above `5.3 m` | A longer wind-up alone does not solve the nine-link terminal mode |

The diagnostic artifacts are preserved under
`runs/generalized_solver/n9_*` and the split adapter is
`scripts/embed_locked_split_route.py`. They remain explicitly marked
`not_solution`. During this continuation, the adapter was found to copy the
source distal absolute rate into the inserted split joint incorrectly; the
state, feedback, and physical-state embedding were corrected and covered by
`tests/test_embed_locked_split_route.py`. The repaired locked-link branch is
still negative and is not part of the promotion.

## Nine-Link Parked-Route Promotion (2026-09-13)

The successful nine-link controller keeps the eight-link runtime contract and
changes only the target-plant route objective and parked target:

| Lever | Result | Interpretation |
| --- | --- | --- |
| Park at `-0.15 m`, 14 s | `4/5` in an initial probe | Too close to the rail for the noisy nine-link route |
| Park at `-0.05 m`, 14 s | `20/20` noisy probe | Accepted launch target |
| Balanced Box-FDDP terminal objective | Endpoint angle about `0.314 rad`; LQR failed | Internal modes remain outside the capture basin |
| Terminal angle factor `20`, hinge-rate factor `4` | Exact handoff `0.00004 rad`, `0.00068 rad/s` hinge RMS | Successful swing-route refinement |
| Strict 3 m tail CEM | No valid replay | Tail-only repair was insufficient |
| Wide-rail tail CEM | Lower angle but hinge RMS about `1.8 rad/s` | More rail does not remove internal momentum |
| Parked route, feedback scale `1.0` | `20/20` and `100/100` noisy; exact `20/20` | Promoted nine-link controller |

The frozen evidence is:

- `runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_20.json`
- `runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_100.json`
- `runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_exact20.json`
- `runs/generalized_solver/nine_link_swingup_success.video.json`
- `runs/generalized_solver/nine_link_swingup_manifest.json`

All noisy 100 episodes ended at the time limit, with maximum cart excursion
`2.9800 m`; the held-out video uses seed `91141`, reaches upright at `17.84 s`,
holds for `12.18 s`, and reports zero resets. This is an internal canonical
benchmark result, not an external record claim. The next ledger entry is the
ten-link extension using this exact parked-launch, feedback, and capture
contract.

## Six-Link Canonical Rail Homotopy Checkpoint (2026-09-13)

The exact route that had previously succeeded only on a wide diagnostic rail
was re-run under the proper `configs/swingup6_uniform.yaml` metadata and
continued back to the canonical `+/-3 m` rail. This applies the promoted
seven-link method directly: saved time-varying Box-FDDP feedback for the
swing phase, a real uninterrupted terminal state, and LQR capture/hold. The
rail was reduced in bounded steps rather than changing the route and rail at
the same time.

| Rail limit | Exact result | Peak cart excursion | Artifact |
| ---: | --- | ---: | --- |
| `12.00 m` | hold `26.06 s` | `3.099 m` | `runs/generalized_solver/n6_wide12_replay_provenance_fixed.json` |
| `3.25 m` | hold `25.94 s` | `2.905 m` | `runs/generalized_solver/n6_rail325_homotopy.json` |
| `3.10 m` | hold `24.76 s` | `2.855 m` | `runs/generalized_solver/n6_rail310_homotopy.json` |
| `3.05 m` | hold `26.06 s` | `2.821 m` | `runs/generalized_solver/n6_rail305_homotopy.json` |
| `3.00 m` | hold `26.08 s` | `2.755 m` | `runs/generalized_solver/n6_rail300_homotopy.json` |

The canonical route was mirrored analytically and evaluated with exact
forward selection after a 15-second hanging-equilibrium conditioning phase:
`20/20` and `100/100` noisy episodes succeeded, both directions were used,
all episodes ended at the time limit, and the maximum required rail ratio was
`0.9724`. The evidence files are
`runs/generalized_solver/n6_rail300_noisy20.json` and
`runs/generalized_solver/n6_rail300_noisy100.json`; the mirrored route is
`runs/generalized_solver/n6_rail300_mirror.json`.

This is a strong six-link integrated-route checkpoint, not a claim that every
earlier six-link capture-basin or global-discovery gate is complete. The P1/P2
learner and real-handoff requirements remain separately tracked. The canonical
presentation and route manifest are now recorded in
`runs/generalized_solver/six_link_swingup_noisy_success.mp4`,
`runs/generalized_solver/six_link_swingup_noisy_success.video.json`, and
`runs/generalized_solver/six_link_route_manifest.json`; they do not close P1/P2
by themselves.

## Ten-Link Frontier Diagnostics Before Promotion (2026-09-13)

At the time of this entry the active target was the uniform ten-link plant.
The nine-link parked-launch route was the incumbent, and every result below
started from the hanging state or from an exact saved state generated by that
route. These are pre-promotion development diagnostics only; none is the
canonical ten-link release evidence recorded later in this ledger.

| Lever | Result | Interpretation |
| --- | --- | --- |
| Direct uniform-10 route transfer | Brief upright crossing, then cart momentum and rail loss | The nine-link swing route reaches the neighborhood but does not shape the tenth internal mode |
| Native ten-link LQR cost sweep | No sustained hold across tested control costs | Changing regulator aggressiveness alone does not repair the handoff |
| Source nine-link LQR/open-loop tail | About `0.46 s` upright at best, then failure | A copied post-route tail is not an adaptive ten-link capture expert |
| Exact online MPC from the clean `3.94 s` handoff | `0.10 s` upright, then `+/-6 m` diagnostic rail exit | Receding-horizon capture injected excessive cart velocity |
| Fixed-state Box-FDDP capture from the same handoff | `0.12 s` upright, then rail loss | Zero-control local warm start leaves the narrow nonlinear basin |
| Locked ghost continuation | Full `29.98 s` hold at release `1e-7`--`2e-7`; failure by `3e-7` | The inherited route survives only while the inserted mode is effectively constrained |
| Free tiny ghost, no support | Immediate instability | A `0.02 m`/`0.01 kg` free mode is a high-frequency numerical training trap |
| Free ghost with temporary damping/stiffness | `29.98 s` at the start of the schedule; 1% release accepted, 2.5% failed | Temporary support is useful for curriculum discovery but does causal work |
| Exact mass-matrix energy/modal controller | `0/20` tested uniform-10 episodes across tested modal gains | Energy reachability does not synchronize the internal modes |
| Full-length free light-tip gradient | Split nine-link route replay on a free `0.033 m`/`0.0001 kg` distal tip reached the `+/-6 m` rail with no upright streak; one angle-weighted FDDP repair reached only `0.18 s` upright before the same rail loss | A light, geometrically present mode is still dynamically disruptive; it is not a free continuation endpoint |
| Shortened nine-link backbone | Re-timing the nine-link route for a `2.7 m` nine-link backbone failed before capture and reached `6.07 m` on the diagnostic rail | The ten-link difficulty is not isolated to the last-link mass; backbone frequency changes matter too |
| 8-second residual CEM on `+/-12 m` rail | Completed 60 iterations; best intermediate point was about `0.577 rad` angle, `2.51 rad/s` hinge RMS, and `15.95` capture violation | No valid capture candidate; earlier force shaping, not a tail-only patch, is required |
| Cart-position CEM with smooth crossing objective | Uniform-10 `+/-12 m` pilot reached all ten absolute angles within `0.139 rad` at `4.12 s`, but hinge RMS was `2.555 rad/s`, cart speed `1.199 m/s`, and upright streak only `0.02 s` | Angle crossing is discoverable, but it is not a low-momentum handoff |
| Long-tail online MPC from the smooth crossing | Six-second exact replay reached no low-momentum interval, held upright only `0.02 s`, and ended at `1.558 rad` maximum angle; best planned point was `0.0955 rad` with `2.465 rad/s` absolute-rate RMS | A longer horizon and weaker cart-centering penalty did not make the measured crossing capturable |
| Exact force CEM seeded from the smooth cart trajectory | Thirty iterations, 64 knots, 96 candidates; best point remained `2.039 rad` angle, `4.668 rad/s` hinge RMS, and `224.5` capture violation | Converting the cart-position seed to force knots did not reveal a low-momentum route |
| Analytic minimum-energy modal seed | Ten-second exact replay ended at `3.108 rad` angle with `6.396 m` rail demand | The linear modal bridge is not a nonlinear swing-up route; use it only as a diagnostic warm start |
| Full-state nonlinear actor CEM from the exact handoff | Reached a brief upright crossing at `7.60 s`; best low-momentum streak was `0.22 s`, maximum cart excursion `6.58 m`, and terminal hinge RMS `3.72 rad/s` | Absolute-plus-relative state features can find the target region, but a single linear actor is not a capture expert |
| Exact n9-transfer replay and four-second Box-FDDP tail | The transferred route left the `+/-12 m` rail; the tail optimizer became unstable in eight iterations | The successful n9 route does not survive dimension lifting without a new n10 swing/capture interface |
| Energy/modal CEM on uniform n10 | Fifty iterations with 48 candidates improved the best angle to `0.926 rad` but produced no upright interval | Energy shaping reaches a closer configuration, but does not synchronize ten internal modes |
| MPC-seeded linear n10 maintenance PPO | Four hundred updates; best independent checkpoint held only `0.20 s` and reached `12.14 m` cart excursion | A linear maintenance learner cannot repair the n10 disturbed upright basin; a nonlinear capture/maintenance expert is still required |
| Chirped resonant square-wave CEM | Thirty iterations; best angle `1.089 rad`, zero upright/low-momentum streak, and about `12.1 m` rail demand | Coherent swing-set excitation alone is not enough; it needs state-dependent phase and mode feedback |
| Full-route residual CEM around the best exact waveform | Sixty iterations; best recorded point at `6.50 s` had `0.463 rad` angle, `2.425 rad/s` absolute-rate RMS, `1.682 rad/s` hinge RMS, and capture violation `11.24` | Refining the entire waveform lowers some rates but still does not reach the n10 capture box |
| Four-second residual correction on top of n10 LQR | Twenty iterations; best cart excursion fell to `3.87 m`, but upright and low-momentum streaks remained zero | A local correction around the current LQR is not enough; the missing controller must shape the arrival state earlier |
| Base-heavy curriculum continuation (`progress=0.0 -> 0.5 -> 1.0`) | Intermediate plant reached `0.388 rad` angle, but still had `max_hinge=6.07 rad/s` and capture violation `14.99`; uniform endpoint regressed to `1.193 rad`, `max_hinge=7.30 rad/s`, and violation `59.27` | The gradient makes the intermediate plant easier but does not transfer to the uniform ten-link endpoint |
| Full-state nonlinear capture actor from the strongest exact handoff | Forty CEM iterations on the `+/-12 m` diagnostic rail; best angle `0.599 rad`, absolute-rate RMS `2.226 rad/s`, maximum cart `3.46 m`, and zero upright/low-momentum streak | A nonlinear actor can reduce local cost but did not enter a capture basin even from a measured near-upright state |
| Eight-second robust tail from the exact `0.546 rad` global handoff | Wide-rail tail reached a point at `0.527 rad`, but the final two-second robust window still had `3.138 rad` maximum angle, `5.75 rad/s` hinge RMS, and `6.54 rad/s` absolute-rate RMS; serial replay hit `12.03 m` | More tail time and rail do not convert the handoff into stable capture |
| Best batched global force crossing | Proposal reached `0.173 rad` maximum angle at `4.96 s`, with hinge RMS `0.856 rad/s`, cart speed `0.939 m/s`, and batched rail `1.92 m`; no hold was present | This is the strongest geometric crossing found, but its internal rates are too high and exact prefix replay drifts to the rail |
| Eight-second exact tail from the batched crossing | Tail CEM remained at about `2.34 rad` angle and `2.12 rad/s` hinge RMS while the exact replay reached `3.03 m` rail | The attractive batched handoff is not reproducible as a canonical reset-free capture route |
| Receding-horizon nonlinear MPC from the `0.173 rad` measured state | Best forecast point had `0.063 rad` angle but `2.37 rad/s` absolute-rate RMS; the uninterrupted execution had zero upright streak and ended at `2.199 rad` angle | Planning quality is not enough; the first-action/replan loop must preserve the predicted modal state |
| Exact LQR gain, scale, and cart-target sweep from the same state | `378` combinations across control costs, scales, and cart targets; none produced an upright streak, and every representative run reached the `3.02 m` rail | The measured crossing is outside the linear regulator basin; tuning the existing stabilizer cannot finish the problem |
| Box-FDDP capture from the same near-upright state | Strict `+/-3 m` rail exited at `3.003 m` without hold; widening the diagnostic rail produced numerical instability and `12.055 m` cart excursion | FDDP needs a better arrival state and/or a state-dependent capture expert; rail widening alone is not a solution |
| Current-model exact CEM re-refinement from the archived near-crossing | After 80 iterations, the best reproducible crossing was `0.354 rad`, hinge RMS `1.266 rad/s`, absolute-rate RMS `2.003 rad/s`, and capture violation `7.72` | The older `0.173 rad` archive does not survive the current XML/model hash; current-model replay is the authority |
| Five-second exact Gauss-Newton endpoint refinement | Reduced endpoint cost from `63.94` to `20.07`; endpoint angle `0.340 rad`, hinge RMS `0.885 rad/s`, absolute-rate RMS `1.158 rad/s`, rail `2.904 m` | Deterministic local refinement improves the arrival state but still misses the capture box |
| Four-point-nine-second timing refinement | Endpoint angle `0.342 rad`, hinge RMS `0.802 rad/s`, absolute-rate RMS `1.247 rad/s`, rail `2.891 m` | Retiming around the first current-model crossing does not remove the remaining absolute-chain motion |
| Affine full-state capture actor from the current-model handoff | Forty iterations found only `0.04 s` maximum centered/upright streak, zero low-momentum streak, and `1.45 m` maximum cart excursion | A local feedback law can touch the upright set but does not stabilize the ten-link internal modes |
| LQR-terminal endpoint objective | Perturbed terminal feedback candidates immediately violated the canonical rail; no valid Jacobian population was available | The current handoff is not inside the existing LQR basin, even when the optimizer is asked to plan for the actual post-handoff rollout |
| Centered exact iLQR terminal objective | Re-optimized the current uniform-10 route with high terminal cart-position and cart-velocity weights | Exact 8-second replay reached `0.0082 rad` maximum angle, `0.123 rad/s` hinge RMS, `0.076 rad/s` absolute-rate RMS, `0.017 m` cart position, and `0.037 m/s` cart speed; this is the strongest arrival state, but zero action held only `0.14 s` |
| Real centered-endpoint maintenance actor | Trained a bounded full-state actor from exact visited handoff states and the centered terminal state | Best shared actor and endpoint actor reached only `0.02 s` and `0.14 s` upright streaks respectively; neither entered a sustained capture basin |
| Exact control-array replay audit | Compared the iLQR producer's direct action-index replay with the tail helper's inclusive-time interpolation | The helper replay was phase-shifted enough to destroy the route; it now uses the exact `t=i*dt` grid. The corrected tail still reached only `0.06 s` hold, so the bug fix did not create a solution |
| Centered-endpoint Box-FDDP and fixed-state iLQR | Started nonlinear capture directly from the centered endpoint, with rate-heavy terminal costs and continued feedback replay | Box-FDDP held `0.04 s`; fixed-state iLQR held `0.10 s`; both ultimately failed the canonical rail/5-second hold gate |
| Endpoint maintenance PPO | Trained raw-action Torch PPO from the exact centered handoff with absolute-rate observations | The best saved checkpoint improved transient return but remained `0/1` success; no public capture claim is justified |
| Centered-endpoint open-loop CEM | Searched a 41-knot corrective force sequence for 8 seconds from the exact handoff | It converged to the zero-action neighborhood at `0.14 s` hold; open-loop correction alone is not the missing stabilizer |
| Slow gated six-link capture continuation | Initialized from the retained `p=0.0725` frontier, used `0.00125` progress steps, `80` updates per stage, and a `0.70` promotion gate | The internal stage evaluation promoted `p=0.0725` to `p=0.07375` at `71.09%`; repeated evaluations at `p=0.07375` held at `67.19%` through update `320`. A frozen held-out check at `p=0.0725` scored `181/256 = 70.70%`, with `13.90 s` median hold among successful episodes and `75/256` rail exits overall; the strict `1,000`-episode/`90%` P1 gate remains closed |
| Normalized-observation six-link continuation | Restarted the gated curriculum with `obs_normalize_reset_scaling=true`, preserving physical reset scaling while mapping observations back to effective state coordinates; used `0.01` progress steps and `40` updates per stage | The run advanced through `p=0.07`. Independent frozen replay scored `187/256 = 73.0%` at `p=0.07` (`13.92 s` median hold, `69/256` rail exits) and `149/256 = 58.2%` at `p=0.08` (`13.81 s` median hold, `107/256` rail exits). The final checkpoint scored `0/256` at `p=1.0` with `0.04 s` median hold and `256/256` rail exits; normalization improves transfer but does not pass P1 or establish a 10-link result |
| Dual-observation linear-scale curriculum | Retained raw physical state, appended effective reset state plus reset scales, and used linear `qpos`/`qvel` reset scales with `0.05` progress steps | The trivial `p=0` stage passed, but the first real stage failed: internal validation was `1/128 = 0.78%`; independent replay of the saved checkpoint was `6/256 = 2.3%`, with `244/256` rail exits and `0.26 s` median hold. Stopped at the decision boundary; the representation and linear two-variable schedule do not pass P1 |
| Dual-observation angle-linear/velocity-cubic curriculum | Retained raw physical state, appended effective reset state plus reset scales, used a linear angle scale and cubic velocity scale, and stepped progress by `0.05` | The first real stage again failed: internal validation was `5/128 = 3.91%` with a `0.823 s` mean maximum upright streak; independent replay of the saved checkpoint was `8/256 = 3.1%`, with `0.240 s` median hold and `184/256` rail exits. Stopped at the decision boundary; this schedule does not pass P1 |
| Locked-split ten-link continuation | Embedded the solved nine-link route exactly in a split ten-link plant, passed the locked baseline for `26.18 s` on a `+/-6 m` diagnostic rail, then released the inserted joint with replay-first screening, bounded waypoint repair, and Box-FDDP | Release proposals `p=0.005`, `0.0025`, and `0.00125` all failed. Best repaired replay latched `0.68 s` at `p=0.0025` and reached `6.04 m`; the smallest tested step latched `0.26 s` and reached `6.04 m`. The ledger bisected the next step to `0.000625` without an accepted free-joint route | Exact route embedding is a valid curriculum baseline, but gradual lock release still crosses out of the route's basin immediately |
| Explicit inserted-joint feedback residual | Added a time-varying feedback term on the split's absolute-angle difference and inserted hinge rate; tested nine angle/rate combinations at `p=0.005` and on uniform free n10 | Every tested setting failed. The uniform endpoint had `0 s` upright hold and `3.013--3.081 m` maximum cart excursion; no residual entered capture | The new mode needs route-level re-planning or a broader feedback policy; a two-term local correction cannot repair the transferred route |

The strongest positive artifacts are
`runs/generalized_solver/n10_ghost_unlocked_d1k10_p0_cost1000.json`,
`runs/generalized_solver/n10_ghost_fine_p2e-7_cost2000.json`, and the associated
ghost continuation configs. They are intentionally marked as diagnostics: the
first uses temporary damping and stiffness, and the second releases only an
infinitesimal fraction of the inserted mode. The current next lever is an
incumbent-preserving route/capture continuation on the genuinely free uniform
plant, with any damping, stiffness, ghost geometry, or widened rail removed
before a ten-link claim is considered.

## Ten-Link Transfer And Bounded-Capture Follow-Up (2026-09-14)

The simplest 9-to-10 transfer remains the baseline. It was regenerated with the
current coordinate and feedback-transfer helpers, then replayed through the
canonical ten-link evaluator. The transferred route did not produce an upright
interval and reached the hard rail at about `3.02 m`; the transfer artifact's
count-changing feedback projection also has a nonzero invariance error because
interpolating a nine-link state into ten links is not invertible. This is a
dimension-lifting limitation, not a reason to claim that the predecessor route
was not used.

The successful n9 artifact records the same distinction: its controller was
`n9_fddp_terminal_angle20_widerail.json`, initialized from
`n9_gn_endpoint_fddp120.json`. In other words, the 8-to-9 transfer was a warm
start for exact n9 target refinement, not a literal copy that was already a
solved n9 controller. The n10 work should preserve that sequence.

| Follow-up | Result | Interpretation |
| --- | --- | --- |
| Corrected literal 9-to-10 transfer | `0/1` exact episode, zero upright streak, `3.022 m` maximum cart excursion, rail termination | The inherited route is the correct first warm start, but the added mode is not controlled by simple interpolation; it needs exact target-plant refinement |
| Actuator-aware feedback-horizon endpoint refinement | Endpoint angle about `0.00027 rad`; the modeled 28-step feedback tail predicted `27/28` saturated actions; exact replay held only `0.04 s` and reached `3.035 m` | A lower endpoint residual is not a bounded capture certificate |
| Nonlinear capture from that measured endpoint | Receding-horizon CEM held `0.06 s`; constrained iLQR held `0.30 s`; both ended at the rail | The current endpoint is outside the usable nonlinear capture basin even when capture is optimized directly |
| Tight velocity plus action-aware endpoint refinement | Endpoint action request reduced to `-0.189`; directional capture-ray radius improved to `0.00437`; exact LQR replay held `0.06 s` and reached `3.004 m` | The action-aware route is the best current local arrival, but it still misses the canonical rail/hold gate |
| Exact waypoint repair of the literal transfer, 24-step segments | First endpoint miss was `0.325`; final miss was `15.67`; maximum cart `3.011 m` | Short local corrections do not preserve the transferred route |
| Exact waypoint repair of the literal transfer, 12-step segments | First endpoint miss was `0.184`; final miss was `23.76`; maximum cart `3.013 m` | Halving the repair horizon does not fix the tenth-link drift |
| Box-FDDP initialized from the clean transfer on the canonical rail | `0/1` capture, no upright hold, `3.087 m` cart excursion | Target-plant re-optimization is necessary, but this direct initialization is not yet a viable route |
| Same transfer and Box-FDDP with a `12 m` diagnostic rail | `0.000 s` hold; the optimizer reached `12.10 m` | Extra runway is not sufficient; the missing ingredient is tenth-mode swing/capture shaping, not only rail width |
| Eight-second terminal iLQR from the clean transfer on a `12 m` diagnostic rail | Best replay point was `0.897 rad` angle with `5.23 rad/s` absolute-rate RMS; terminal state reached `12.04 m` | Removing the canonical rail constraint does not make the transferred route a useful arrival trajectory |
| Eight-second terminal iLQR from the clean transfer on the canonical rail | Terminal angle `2.612 rad`, hinge RMS `16.64 rad/s`, and cart position `0.197 m` | The route optimizer is outside the useful local basin; terminal-cost refinement must be seeded from a better target-consistent route or a staged continuation |

Artifacts: `runs/generalized_solver/n10_transfer_clean_corrected.json`,
`runs/generalized_solver/n10_transfer_clean_corrected_eval1.json`,
`runs/generalized_solver/n10_feedback_horizon_gn_w1e-6_geometry.json`,
`runs/generalized_solver/n10_feedback_horizon_gn_capture_mpc.json`,
`runs/generalized_solver/n10_feedback_horizon_gn_capture_ilqr.json`, and
`runs/generalized_solver/n10_zero_target_high_gn_tight_action100_rail295_geometry.json`.
The transfer-repair diagnostics are
`runs/generalized_solver/n10_transfer_clean_waypoint24.json`,
`runs/generalized_solver/n10_transfer_clean_waypoint12.json`,
`runs/generalized_solver/n10_transfer_clean_fddp80.json`, and
`runs/generalized_solver/n10_transfer_clean_rail12_fddp80.json`,
`runs/generalized_solver/n10_transfer_rail12_terminal_refine8.json`,
`runs/generalized_solver/n10_transfer_rail12_terminal_refine8_replay_canonical.json`,
and `runs/generalized_solver/n10_transfer_canonical_terminal_refine8.json`.
All are diagnostic artifacts and remain outside the canonical ten-link claim.

## Ten-Link Promotion: Bounded Route Feedback (2026-09-14)

The ten-link frontier is now promoted on the uniform target plant. The release
keeps the two-expert structure discussed throughout this ledger, with a
continuous hanging-equilibrium conditioning phase, a target-plant swing expert,
and a terminal upright capture expert. There is no simulator reset at either
phase boundary.

| Lever | Result | Lesson |
| --- | --- | --- |
| Exact endpoint SVD refinement only | Exact replay reached the upright manifold and held for `22.12 s`, but a `0.01 rad` perturbation destroyed the route | A quiet endpoint is not a noisy capture basin |
| Longer two-second arrival shaping | Removed the terminal impulse and produced the strongest exact handoff | The arrival must be shaped over time, not only at one sample |
| Target-plant Box-FDDP with saved time-varying feedback | Exact replay held for `22.12 s` after the `8.0 s` route | The route feedback must be re-optimized for the added tenth mode |
| Fourteen-second parked launch | `17/20` noisy at first, with marginal rail failures | Settling helps, but the launch residual was still too large for the ten-link route |
| Sixteen-second parked launch, cart target `0.0 m` | `100/100` noisy on the canonical rail, maximum cart excursion `2.0483 m` | Longer conditioning was the decisive robustness lever |
| Sixteen-second parked launch, cart target `-0.05 m` | `20/20` noisy, `100/100` noisy, and exact `20/20`; maximum noisy excursion `1.9976 m` | The original parked-target presentation is also canonical-gate safe |
| Held-out render | Seed `102221`, zero resets, all ten links visible, first upright at `23.90 s`, hold `6.12 s` | The release is externally inspectable instead of being a metrics-only artifact |

The frozen release files are:

- `runs/generalized_solver/n10_fddp_refined_route_feedback100.json`
- `runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005_20.json`
- `runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005_100.json`
- `runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005_exact20.json`
- `runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005.mp4`
- `runs/generalized_solver/ten_link_swingup_manifest.json`

The negative controls remain important: static LQR gain sweeps, bounded MPC,
wider rails, endpoint-only refinement, and ghost-link support did not solve the
free uniform plant. The promoted result is therefore specifically a
target-plant route-feedback result, not evidence that every earlier method
worked. It is an internal canonical benchmark, not an external competition
record.

The active next lever is eleven-link target-plant re-optimization. Start from
the ten-link parked launch and route as a warm start, preserve the `16.0 s`
conditioning and `-0.05 m` target initially, and require the same exact/noisy
gates before changing the contract. Dimension-lifted transfer, widened rail,
altered morphology, and temporary support remain diagnostics until replayed on
the uniform eleven-link plant.

## Eleven-Link Warm-Start Probe (2026-09-14)

The first eleven-link attempt used the same route-transfer discipline as the
successful ten-link promotion: a uniform `n=11` target config, a dimension-lifted
ten-link route, the unchanged `16.0 s` parked launch, and exact target-plant
Box-FDDP refinement before any noisy gate. The literal padded route failed the
zero-noise replay at `3.0985 m`. The generalized morphology transfer was then
replayed on five noisy seeds (`0/5`) and refined with Box-FDDP; the optimizer
stopped after `13` iterations with a `3.007 m` rail exit and no upright hold.

This is a warm-start failure, not an eleven-link impossibility result. The
ten-link release remains valid and unchanged. The next attempt needs an
eleven-link route seed whose inserted mode is shaped before the final target
plant, or a longer target-plant route optimization; simply padding or
interpolating the ten-link controller is insufficient.

The follow-up audit used the released ten-link controller itself rather than
the older n10 development route. Exact target-plant transfer replay still
exited the canonical `3 m` rail before capture (`3.0449 m`), and a 220-iteration
Box-FDDP refinement reduced its nominal terminal Lyapunov value to about
`1,391` but replay still exited at `3.058 m`. A locked split continuation was
then tested as a training wheel: target-plant Box-FDDP reached a short `0.62 s`
upright interval, but the LQR tail left both the `+/-3 m` and temporary `+/-6 m`
rails. Direct FDDP stabilization from an extracted quiet upright state also
failed on the wider rail. These are route/capture failures, not evidence
against the ten-link release or a proof that eleven links is impossible.
The next experiment must shape the newly inserted mode before it reaches the
capture phase, then remove every lock and wider-rail allowance before a claim.

The current-release re-run is recorded separately in
`runs/generalized_solver/n11_from_current_n10_pipeline.json`. It used the
published ten-link route, the same `16.0 s` parked launch, a dimension-lifted
11-link transfer, and an 80-iteration exact target-plant Box-FDDP refinement.
The transfer replay was `0/5` on held-out target-plant seeds. Refinement
stopped after `8` iterations with `min_v=7.666e6`, terminal
`v=5.787e15`, and a live `3.004 m` rail exit. The pipeline correctly stopped
before packaging a route or running a gate. This is the current eleven-link
frontier, not a promotion and not evidence that the ten-link release regressed.

## Independent Ten-Link Release Audit (2026-09-14)

The frozen ten-link release was replayed in a fresh process using the recorded
config, controller, phase timing, cart target, and three disjoint seed cohorts.
The corroborating artifacts are:

- `runs/generalized_solver/n10_independent_replay_20.json`: noisy `20/20`
- `runs/generalized_solver/n10_independent_replay_100.json`: noisy `100/100`
- `runs/generalized_solver/n10_independent_replay_exact20.json`: exact `20/20`

All three reached the time limit without a rail termination. The noisy 100-episode
replay had a maximum cart excursion of `1.9976 m`; the exact replay had
`1.9714 m`. These files corroborate the frozen release but are deliberately not
substituted into its manifest, checksum list, or claim boundary.

## Six-Link Exact Validation Fine Continuation (2026-09-14)

The exact capture curriculum was restarted from the stronger LQR-homotopy
checkpoint instead of the earlier small-probe frontier. The run used the frozen
validation state list, a `0.001` progress step, a `0.90` promotion gate, and
256 indexed episodes per evaluation. It passed four consecutive stages:
`p=0.055`, `0.056`, `0.057`, and `0.058`. The saved `p=0.058` frontier scored
`234/256 = 91.41%` with a `15.02 s` median hold on that curriculum subset.

The broader audit is the controlling result for generalization: the same
checkpoint scored `1773/2000 = 88.65%` across the complete frozen validation
split, with `227/2000` rail terminations. The 256-state score is therefore
useful for measuring curriculum advancement but is not evidence of a P1 pass.
The prior `p=0.05` full-validation checkpoint remains stronger on the complete
split at `1817/2000 = 90.85%`; the current work still needs a policy that clears
the full `1000`-episode test gate with the required rail-safety condition.

Artifacts:

- `runs/swingup6_capture_validation_lqr_fine_0001/checkpoints/frontier.safetensors`
- `runs/swingup6_capture_validation_lqr_fine_0001/train_log.csv`
- `runs/p1_capture_envelope/eval_lqr_fine_frontier_p0058_validation2000.json`
- `runs/p1_capture_envelope/eval_lqr_homotopy_best_p0055_validation256.json`
- `runs/p1_capture_envelope/eval_axis_homotopy_frontier_p006_validation256.json`

Lesson: a stronger local curriculum branch can advance the exact indexed gate,
but the current 256-state promotion subset is optimistic relative to the full
validation distribution. Future promotions must be checked against the complete
split before they are treated as a real capture-basin expansion.

## Six-Link Full-Width Frontier and Fine-Step Audit (2026-09-14)

The validation-sized gate was widened to `1000` indexed states before changing
the policy architecture. The LQR-homotopy checkpoint at `p=0.055` scored
`1846/2000 = 92.30%` on the complete frozen validation split, with a `15.02 s`
median hold and `153` rail terminations. A 1000-state continuation passed
`p=0.055` and `p=0.056`, then failed `p=0.057` twice at `89.4%`.

Increasing the LQR residual authority to `1.30` produced a real but limited
improvement: the full-width continuation passed `p=0.056` through `p=0.059`
on 1000 validation states. The saved `p=0.059` checkpoint scored
`1807/2000 = 90.35%` on the complete validation split, with a `15.02 s` median
hold and `193` rail terminations. A static `1.50` authority override at
`p=0.060` scored `0/2000` and terminated every episode at the rail, so larger
LQR scale is not a monotone fix.

A finer continuation from the `p=0.059` checkpoint used `0.00025` progress
steps and the same 1000-state gate. It passed `p=0.059` at `90.1%` and reached
the threshold at `p=0.05925` (`90.0%`), then failed `p=0.0595` twice at
`89.6%`. The saved `p=0.05925` checkpoint also scored `1803/2000 = 90.15%`
on the complete validation split, with `197` rail terminations.

A post-training residual-floor screen looked promising on the first 256
validation states (`92.2%` at floor `0.10`), but the complete 2000-state audit
fell to `1792/2000 = 89.60%`. Training with a floor of `0.05` likewise failed
the 1000-state promotion three times at `89.4%` on `p=0.0595`. These are
selection-bias and adaptation warnings, not improvements to the incumbent.

The `p=0.05925` checkpoint is the current full-width development frontier; it
is not a P1 pass because the policy has not reached `p=1.0`, the frozen 1000-
state test split has not passed, and successful rail safety is still not
established.

## Exact Six-Link Discovery Route (2026-09-14)

The saved mirrored six-link Box-FDDP route was replayed from the canonical
hanging distribution with an exact MuJoCo hanging-LQR conditioning phase. A
new evaluator, `scripts/evaluate_global_discovery_route.py`, keeps route
selection in isolated forward predictions but executes the selected route,
capture LQR, and stabilization in one live episode. The four held-out seeds
`90001` through `90004` all reached the five-second hold requirement on the
`+/-3 m` rail, with full `qpos`/`qvel`/action traces and `post_launch_reset_count
= 0` in `runs/goal_global_discovery/global_route_evidence.json`.

This closes the narrow global-discovery feasibility checkpoint. It does not
close P1: the route starts after a 15-second conditioning phase and is not a
reusable capture policy for the frozen synthetic envelope. It also does not
close P2 or P3. The existing negative local-capture results remain relevant:
launching the route before internal hinge rates are quiet causes rail loss,
and direct full-envelope LQR, linear MPC, nonlinear MPC, open-loop CEM, and
simple gain scaling did not recover the hard states.

Reproduction command:

```text
PYTHONPATH=src:scripts ./.conda-aligator/bin/python scripts/evaluate_global_discovery_route.py --config configs/swingup6_uniform.yaml --controller runs/generalized_solver/n6_rail300_route.json --controller runs/generalized_solver/n6_rail300_route_mirror.json --episodes 4 --seed 90001 --conditioning-seconds 15 --tracking-gain-scale 1.0 --phase-window 12 --hold-seconds 5 --out runs/goal_global_discovery/global_route_evidence.json --fail-on-gate
```

The same route was then replayed for 100 disjoint seeds with trace storage
disabled but measured handoffs retained in
`runs/goal_global_discovery/global_route_100_handoff_evidence.json`. It scored
`100/100` live successes, `100/100` five-second holds, and `100/100`
independent capture-expert replays. Every episode ended by the time limit and
the measured handoff relative-rate RMS stayed between `0.0017907` and
`0.0017911 rad/s`. This cohort is useful for the next capture/swing teacher
loop, but it is not a P1 pass: the launch includes the route's 15-second
hanging conditioning phase and the P1 envelope is still not solved at
`progress=1.0`.

The measured handoffs are exported as
`runs/swingup6_policy_handoff/route_handoffs_100.json` with disjoint `80/10/10`
train/validation/test membership. Each row preserves the source seed, route
choice, exact `qpos/qvel`, handoff time, absolute-link rate metrics, and the
independent capture replay result. The exporter records the artifact as
`not_solution`; it is a real-state training dataset, not a replacement for
the frozen P1 envelope or a claim that conditioning can be hidden inside a
capture expert.

**Reverse-time route diagnostic (2026-09-14).** The saved swing route was
reversed with velocities sign-flipped, controls reversed, and the feedback
coordinates transformed accordingly, then replayed from five full-scale
near-upright P1 states with phase windows `0/2/6/12`. All trials exited the
canonical rail before reaching a controlled hanging state. This rejects the
time-reversed route as a down-swing recovery method; it did not provide a
shorter alternative to passive settling.

**Shared endpoint feedback search (2026-09-14).** A common `tanh`-squashed
full-state law over cart state, absolute angles/rates, relative angles/rates
was optimized against eight full-scale held-out endpoint states for twelve
exact-MuJoCo CEM iterations. The best law achieved `0/8` sustained successes
and `5/8` rail exits in the ten-second diagnostic. The stationary linear
feature family is rejected as the capture expert; the artifact is
`runs/p1_capture_shared_feedback_endpoint8.json`.

**Shortened launch and recovery diagnostic (2026-09-14).** The saved exact
six-link route still passed `8/8` noisy hanging-start episodes after reducing
the hanging-LQR prelude from `15 s` to `4 s` (`runs/goal_global_discovery/
global_route_cond4_8.json`). This is useful for the eventual episode budget,
but it does not generalize to the P1 envelope. On the first 32 frozen P1 test
states, a reset-free four-second recovery followed by route selection produced
`0/32` successes and `32/32` rail exits. A passive cart-centering fall with
gains `(0.1, 0.05)` also produced `0/32`; higher gains only increased internal
rates and rail exposure. The detailed diagnostics are under
`runs/goal_global_discovery/p1_fall_route_cond4_test32.json` and
`p1_fall_route_pd_*_test32.json`. The shorter launch is accepted as a route
optimization, while passive fall-to-route is rejected as a P1 recovery expert.

Artifacts:

- `runs/swingup6_capture_validation_lqr_scale13_fullgate_0001/checkpoints/frontier.safetensors`
- `runs/swingup6_capture_validation_lqr_scale13_fullgate_0001/train_log.csv`
- `runs/p1_capture_envelope/eval_lqr_scale13_frontier_p0059_validation2000.json`
- `runs/p1_capture_envelope/eval_lqr_scale15_p006_validation1000.json`
- `runs/swingup6_capture_validation_lqr_scale13_finegate_0001/checkpoints/frontier.safetensors`
- `runs/swingup6_capture_validation_lqr_scale13_finegate_0001/train_log.csv`
- `runs/p1_capture_envelope/eval_lqr_scale13_frontier_p005925_validation2000.json`
- `runs/p1_capture_envelope/eval_lqr_residualfloor10_p00595_validation2000.json`
- `runs/swingup6_capture_validation_lqr_floor05_finegate_0001/train_log.csv`

Lesson: the extra LQR authority and finer curriculum step enlarge the measured
capture basin, but only incrementally. The next useful experiment must address
the remaining distribution shift or provide a stability-preserving recovery
representation; simply increasing force or continuing the same PPO updates is
not justified by the evidence.

## Six-Link Hard-Tail and Saturation-Aware Residual Probes (2026-09-14)

The `p=0.05925` incumbent was evaluated on `2048` frozen training states and
found `219` failures (`10.7%`). Repeating those failures three times produced
the training-only mixture `train_hard_005925.json`; the validation and test
state lists were unchanged. A fine-step PPO continuation passed its first
1000-state gate at `p=0.05925`, but reached only `896/1000 = 89.6%` at
`p=0.0595` on updates 40, 60, and 80. Hard-tail sampling therefore did not
enlarge the held-out basin and is retained as negative evidence.

The environment now contains an opt-in `policy_on_saturation` residual mode.
When the LQR residual bias is clipped, the actor can take full normalized-force
authority; ordinary additive residual behavior remains the default. On the
saved checkpoint this untrained mode scored `79.7%` at saturation threshold
`0.95` and `83.2%` at `0.999` on the 256-state diagnostic, versus `89.9%` for
the incumbent, so it is not promoted without retraining. The focused regression
tests pass, and the mode remains available for a future stability-preserving
architecture experiment.

Artifacts:

- `runs/p1_capture_envelope/train_hard_005925.json`
- `runs/swingup6_capture_validation_lqr_hardtail_finegate_0001/train_log.csv`
- `runs/p1_capture_envelope/eval_lqr_saturation_switch_p00595_validation256.json`
- `runs/p1_capture_envelope/eval_lqr_saturation_switch999_p00595_validation256.json`
- `runs/p1_capture_envelope/eval_lqr_tanh_p00595_validation256.json`

## Eleven-Link Count-Continuation Playbook (2026-09-15)

The 11-link campaign is now constrained to the same settled launch, saved
time-varying swing feedback, measured terminal capture, and sustained-hold
method used for the 7-10 link development releases. No 11-link claim is
permitted until the canonical hanging-start, exact/noisy episode gates,
reset-free video, hashes, and independent replay all pass.

The first locked-split route audit found two bookkeeping hazards. The custom
`scripts/embed_locked_split_route.py` diagnostic duplicated the distal source
hinge rate at both sides of the inserted joint; that is not the physical
locked split, whose new internal joint starts at zero rate. The algebraic lift
in `scripts/generalized_swingup_solver.py split` was therefore used for the
controlled run because it preserves the source route and feedback exactly.
The adaptive driver also screened its baseline and release replays with
`tracking_gain=0`, which silently converted the two-expert controller into a
feedforward waveform. `scripts/run_split_count_homotopy.py` now preserves the
configured tracking gain for those screens.

The corrected locked-start route is a real development success: the exact
11-coordinate locked split reproduces a `22.12 s` hold with a `2.023 m` peak
cart excursion on the `+/-3 m` rail. It is not an 11-link result because the
last internal joint remains locked and the two half-links are not yet the
uniform target chain.

The first release probe keeps the route unchanged while freezing the plant at
each morphology progress value. The inherited route passes through
`p=0.000017` with the same `22.12 s` hold and `2.023 m` peak cart excursion;
raw replay fails between `p=0.000017` and `p=0.000018`, then reaches the rail
after only `0.28-2.0 s` for `p=0.000018-0.0001`. Reconditioning from the
stable locked route with Box-FDDP recovered full `22.12 s` holds at
`p=0.000018`, `0.000020`, `0.000025`, and `0.000035-0.000039`, with peak cart
excursions near `2.070-2.071 m`. The same repair failed at `p=0.000040` and
`p=0.000050`. The current development frontier is therefore the accepted
`p=0.000039` checkpoint, not a canonical 11-link solve.

Current decision rules:

- Keep the last complete-hold controller as the incumbent at every release
  step; never promote a terminally quiet but rail-unsafe trajectory.
- Use logarithmic or bisected release steps near the measured `4.0e-5`
  boundary. Directly transferring a repaired controller between nearby
  frozen plants was unstable, so controller transfer and morphology transfer
  must be reconditioned together from the stable locked seed.
- Treat a locked split, wider rail, ghost link, altered mass/length gradient,
  or diagnostic hold as a curriculum instrument only. None can update the
  public link count.
- When an unlocked uniform 11-link controller finally passes exact replay,
  run the full 20/100 noisy gates, held-out video, manifest/hash audit, paper,
  README/About update, and public push before starting 12 links.

Primary artifacts:

- `runs/generalized_solver/n11_locked_split_warm_start.json`
- `runs/generalized_solver/n11_locked_split_algebraic_replay.json`
- `runs/generalized_solver/n11_algebraic_split_release_probe/`
- `runs/generalized_solver/n11_algebraic_split_release_probe/p3.90e-05_fddp_from_locked.json`

The current campaign has not solved 11 links. The playbook remains the
controlling record for what was tried, what passed, what failed, and which
intermediate states are admissible as the next incumbent.

### Follow-up continuation diagnostics

The next bridge experiments narrowed the failure modes further:

- The linear lock-impedance path was not a bridge. Replay-only release failed
  at `p=0.01` and `p=0.05`; Box-FDDP repairs from the stable locked seed also
  failed immediately with MuJoCo instability warnings. Do not interpret a
  linear spring/damper schedule as a smoother version of the logarithmic
  locked split.
- A geometry-first path kept the final split locked while changing only the
  embedded lengths and masses toward uniform 11-link geometry. Its first
  nonzero trial at `p=0.0125` failed replay and both local repairs, with peak
  cart motion at the rail and numerical-instability warnings. This means the
  current issue is not only the equality release; even the rigid-split
  geometry perturbation has a very narrow basin.
- The supported-unlock ramp using unit-scale stiffness and `0.01` damping
  produced an invalid all-zero trace after MuJoCo instability warnings. Its
  apparent 30-second holds are rejected as a numerical-collapse false
  positive, and the support curriculum is not evidence for 11 links.

The current defensible development frontier remains the logarithmic rigid-split
route at `p=3.90e-05`, which holds for 22.12 seconds but is not uniform 11-link
evidence. Any next curriculum must preserve a nonzero, physically evolving
trace, retain the exact rail constraint, and pass an independent replay before
it can replace that incumbent.

### Forced split-position and numerical-integrity follow-up

To test whether the location of the inserted split was the limiting factor, the
corrected adapter was forced to split the locked route at links `1`, `3`, `5`,
`7`, `9`, and `10`. Positions `7`, `9`, and `10` reproduce the real locked
`22.12 s` development hold at `p=0`; positions `1`, `3`, and `5` do not. This
only maps the locked-route basin. Release attempts from the successful
positions still failed at the first useful nonzero values (`p=0.001` and then
successively bisected values), so there is no evidence that choosing a better
split position has widened the uniform-chain basin.

The adapter itself is now physically conservative: an inserted internal hinge
starts with zero rate, and the source split angle/rate feedback is projected
onto the proximal target coordinate instead of being duplicated onto both sides
of the split. Older forced-split artifacts made with the duplicating adapter
must not be used as evidence.

Two additional checks closed a verification gap. A `rail=12` ghost-link route
still became unstable at `p=0.00003`, and a damping-only support curriculum
reported apparent holds only because MuJoCo had already collapsed the simulated
state to zeros after instability warnings. `scripts/run_generalized_homotopy.py`
now requires finite, dimensionally correct, physically evolving traces that
start near the declared selected state and rejects long zero-collapse suffixes.
The previously reported damping-only holds are therefore explicitly rejected.

The playbook rule is now: a continuation result is admissible only when the
numerical-integrity gate, the rail/hold gate, and independent replay all pass.
This prevents a solver warning from being promoted as a new link count.

Additional artifacts:

- `runs/generalized_solver/n11_forced_split_link7_homotopy/`
- `runs/generalized_solver/n11_forced_split_link9_homotopy/`
- `runs/generalized_solver/n11_ghost_continuation_rail12.yaml`
- `runs/generalized_solver/n11_locked_damped_support_homotopy/`
- `scripts/embed_locked_split_route.py`
- `scripts/run_generalized_homotopy.py`
- `tests/test_generalized_homotopy.py`

### Standardized locked-route continuation audit (2026-09-15)

The corrected continuation driver was rerun from the exact locked 11-link
baseline with the same acceptance settings used for the current development
route: `lqr_scale=1`, full feedback tracking, finite-trace integrity checks,
the canonical `rail=3` constraint, and independent replay/FDDP passes. The
exact locked baseline passed with a `22.12 s` hold and `2.023 m` maximum cart
excursion, confirming that the audit itself can reproduce the incumbent.

The first accepted release was only `p=0.00003125`, with a real `22.12 s`
hold and `2.0704 m` maximum cart excursion. The neighboring release attempts
at `p=0.0000625`, `0.000046875`, `0.000039063`, `0.000035156`, and
`0.000033203` all failed by rail violation despite reconditioning. The
continuation therefore reached a narrow locked-route frontier below the
previous `p=0.000039` checkpoint; it did not produce an unlocked, uniform
11-link controller.

This audit is admissible development evidence because the accepted trace is
finite, starts near the selected state, remains physically nonzero, passes the
rail/hold gate, and is independently replayed. It is not a public release:
the canonical 11-link system still has no exact 20/100 noisy-episode result,
reset-free held-out video, or public evidence bundle. Keep the README and
GitHub About at 10 links until those gates pass.

Primary audit record:

- `runs/generalized_solver/n11_locked_split_homotopy_retry/continuation.json`
- `runs/generalized_solver/n11_locked_split_homotopy_retry/trials/trial_0006_p0.000031250_pass1.json`
- `runs/generalized_solver/n11_locked_split_homotopy_retry/trials/trial_0006_p0.000031250_pass2.json`

### Rail-margin unlock probe (2026-09-15)

The rail hypothesis was tested with the same locked-split controller and
release schedule, but with the rail staged from `6.0` to the canonical `3.0`
as the final split unlocked. The exact locked baseline passed on the six-unit
rail for `22.12 s` with `2.023 m` maximum cart excursion. The first four
release attempts (`p=0.001`, `0.0005`, `0.00025`, and `0.000125`) all failed
with finite, non-collapsed traces; the cart reached `6.02-6.14 m` and exited
the expanded rail before capture. The run was stopped after this bracket
because the failure was already at the first unlock, not at the rail
contraction endpoint.

Conclusion: more rail is useful as a diagnostic margin and confirms that the
locked route itself is not rail-limited, but it does not supply the missing
feedback transfer across the first unlocked split. Do not describe this as an
11-link result or continue spending compute on the same rail-only variation
without changing the controller handoff or curriculum coordinate.

Artifacts:

- `runs/generalized_solver/n11_locked_split_rail6_to3_continuation.yaml`
- `runs/generalized_solver/n11_locked_split_rail6_to3_homotopy/continuation.json`

### Free-chain two-expert capture probe (2026-09-15)

To test the proposed swing-up/capture split directly, 24 handoff states were
extracted from the best split-route trajectory after `7.0 s`, filtered to
`max_angle <= 0.20`, hinge and absolute-rate RMS below `0.75`, cart position
within `1.5 m`, and cart speed below `0.5`. These are real visited positions,
not synthetic upright samples. A reset-free two-mode affine CEM actor was then
optimized on the exact uniform 11-link target for `4.0 s` per state.

The search achieved zero full successes on every iteration and every state.
The best candidates produced only brief `0.20-0.56 s` upright windows before
the free distal mode diverged; the final evaluation also had zero successes.
This isolates the capture expert as an unresolved problem: the split-route
controller can create a quiet handoff on its assisted plant, but that handoff
does not transfer to the free uniform plant. The result is diagnostic only and
cannot be used for a link-count claim.

Artifacts:

- `runs/generalized_solver/n11_p39_low_momentum_handoff_states.json`
- `runs/generalized_solver/n11_uniform_capture_mixture_from_p39.json`
- `runs/generalized_solver/n11_uniform_from_p39_fddp100.json`

### Split-mass-floor and ghost-damping probes (2026-09-15)

The rigid-split XML path now exposes the bounded development-only lever
`env.rigid_split_mass_fraction`. While a split is locked, the numerical child
bodies retain this fraction of the aggregate group mass instead of using the
previous `1e-8` floor. The released target plant is unchanged: the floor is
irrelevant when the split lock and morphology continuation reach the canonical
uniform endpoint, and the total cart-plus-chain mass is preserved at every
stage. The default remains `1e-8`.

The lever was tested on the real `p=3e-5` ghost handoff route with floors from
`1e-6` through `1e-1`. None produced a complete capture. Floors from `1e-6` to
`1e-4` reached at most `0.70 s` of upright streak before a finite physical
rail violation at approximately `6.03-6.09 m`; larger floors reduced the
streak further, and `0.1` also produced an unstable rail exit. This is a
negative result for the numerical mass-floor hypothesis, not evidence that
the canonical plant is solved.

The same handoff was replayed with added distal damping from `0.001` through
`0.03`, with and without unit stiffness. These variants also reached at most
`0.70 s` before rail exit near `6.03 m`; damping `0.1` and the stiff variant
went numerically unstable and are rejected by the trace-integrity gate.

Decision: retain the mass floor as a controlled diagnostic because it makes the
lock-to-free representation explicit and testable, but do not promote it as a
capture method. The next experiment must improve the feedback handoff itself
and must use real saved swing states, exact canonical dynamics, finite evolving
traces, the physical rail, and independent replay. The playbook rule remains:
every new lever gets a named configuration, an artifact, a failure or pass
metric, and a disposition before the next lever is changed.

Artifacts:

- `src/gcartpole/mjxml.py` (`rigid_split_mass_fraction`)
- `runs/generalized_solver/n11_ghost_p3e-5_floor_f1e-6.json`
- `runs/generalized_solver/n11_ghost_p3e-5_floor_f1e-4.json`
- `runs/generalized_solver/n11_ghost_p3e-5_floor_f1e-3.json`
- `runs/generalized_solver/n11_ghost_p3e-5_floor_f1e-2.json`
- `runs/generalized_solver/n11_ghost_p3e-5_floor_f5e-2.json`
- `runs/generalized_solver/n11_ghost_p3e-5_floor_f1e-1.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d001_replay.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d003_replay.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d01_replay.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d03_replay.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d1_replay.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d03k1_replay.json`
- `runs/generalized_solver/n11_ghost_p3e-5_d1k1_replay.json`

### Real-handoff PPO capture probe (2026-09-15)

The next inherited-method test trained a bounded Torch PPO capture policy on
the exact uniform 11-link target, initialized only from 26 real states visited
by the `p=3e-5` ghost swing route. The states were filtered to `t=8.0-8.6 s`,
`max_angle <= 0.20`, hinge and absolute-rate RMS `<= 0.75`, cart position
within `1.0 m`, and cart speed within `1.0 m/s`; their hinge coordinates were
wrapped into `[-pi, pi)` without changing `qvel`. The canonical rail stayed at
`+/-3 m`, and the run used one uninterrupted 12-second episode per state.

The `600`-update run remained finite and produced real, nonzero traces. Its
best checkpoint improved the selected-state transient to a `0.28 s` maximum
upright streak, but the independent all-state replay scored `0/26` successes.
Every state reached an upright crossing, yet the mean maximum upright streak
was only `0.204 s` (range `0.16-0.28 s`), the mean maximum cart excursion was
`2.62 m`, and the run ended in rail violations or time limits. This is a
stronger learned local response than the affine CEM probe, but it is still not
a capture/stabilize expert and cannot be joined to the swing route.

The PPO run initially exposed a runtime reproducibility pitfall: the bundled
MuJoCo environment is Python 3.12 while the prior helper assumed a mixed
Python version. `scripts/torch_runtime.py` now imports system NumPy/PyTorch
first and appends the matching Python 3.12 MuJoCo site-packages afterward,
avoiding the duplicate OpenMP initialization that aborted the first attempt.

Decision: do not promote the PPO checkpoint. The result says the difficulty
is not merely discovering an upright crossing; the policy needs a stability-
preserving representation or a staged maintenance/capture teacher that
keeps the distal modes inside a viable basin. The next run must retain the
real-state bank, exact target dynamics, finite-trace gate, and independent
all-state replay while changing the capture architecture explicitly.

Artifacts:

- `configs/swingup11_capture_real_handoff.yaml`
- `runs/generalized_solver/n11_p3e-5_real_capture_states.json`
- `runs/generalized_solver/n11_capture_real_handoff_ppo/train_log.csv`
- `runs/generalized_solver/n11_capture_real_handoff_ppo/checkpoints/best.pt`
- `runs/generalized_solver/n11_capture_real_handoff_ppo/checkpoints/best.meta.json`
- `runs/generalized_solver/n11_capture_real_handoff_ppo/eval_all_26.json`
- `scripts/extract_trajectory_states.py`
- `scripts/torch_runtime.py`

### Frozen-start maintenance curriculum probe (2026-09-15)

Before attempting another swing/capture handoff, the maintenance prerequisite
was isolated with `configs/swingup11_ghost_maintenance_ppo_frozenstart.yaml`.
The actor was held at its zero-output initialization during the exact upright
stage so that the gate measured the known equilibrium, not accidental PPO
drift. It passed that diagnostic stage on `8/8` episodes with an `8.02 s`
upright streak and then advanced to the first nonzero morphology/noise stage,
`progress=0.025`.

The policy never passed that first nonzero stage in `600` updates. The held-out
evaluation was `0/8` at updates `150`, `200`, `250`, `300`, `350`, `400`,
`450`, `500`, `550`, and `600`; the final evaluation return was `88.2` but
success remained `0.00`. Rollouts became less negative late in training, but
that shaped-return improvement did not produce a sustained upright state. A
zero-action probe at the same stage also scored `0/8`, so the failure is a real
active-stabilization requirement rather than a bookkeeping artifact.

Decision: do not advance the morphology curriculum and do not treat the exact
upright gate as a maintenance solve. The current PPO observation/action
representation can preserve the trivial equilibrium but has no demonstrated
robust basin at the first disturbed stage. The next maintenance or capture
experiment must add an explicit stabilizing teacher, residual/teacher policy,
or another named feedback representation, and must retain the `0.025` held-out
gate, finite-trace checks, canonical rail, and independent replay. The
playbook rule is unchanged: a curriculum checkpoint is evidence only for the
exact stage and state distribution it actually passes.

Artifacts:

- `configs/swingup11_ghost_maintenance_ppo_frozenstart.yaml`
- `runs/generalized_solver/n11_ghost_maintenance_ppo_frozenstart/train_log.csv`
- `runs/generalized_solver/n11_ghost_maintenance_ppo_frozenstart/checkpoints/update_000600.pt`
- `runs/generalized_solver/n11_ghost_maintenance_ppo_frozenstart/checkpoints/latest.meta.json`

### Longer-rail residual search and Box-FDDP refinement (2026-09-15)

To test the rail-length hypothesis without changing the acceptance contract, a
`16 s` exact serial-MuJoCo CEM searched a bounded residual force waveform on a
`+/-12 m` discovery rail, centered on the released 10-link force route. The
candidate was scored against the canonical `+/-3 m` handoff limits rather than
being allowed to claim the discovery rail. The search stayed finite and its
best candidate stayed inside the canonical rail (`2.4108 m` peak cart
excursion), but its best late state was still `1.2353 rad` maximum angle,
`3.1929 rad/s` hinge RMS, and `3.9650 rad/s` absolute-rate RMS. The full
candidate therefore had no valid low-momentum handoff.

The exact saved waveform was then replayed into a target-plant Box-FDDP
refinement with angle, rate, cart, terminal, and canonical-rail penalties. The
optimizer stopped after `11` iterations without convergence; the refined
replay produced a finite-or-warning-invalid unstable trace, reached the
canonical rail at `3.018 m`, and held for `0 s`. The solver warning and the
large terminal Lyapunov value reject it as a route improvement.

Decision: a longer discovery rail can help explore the force waveform, but it
did not supply the missing 11-link capture basin in this controlled test. Keep
the longer rail available only as a search condition, and require a complete
canonical-rail replay plus independent integrity check before any candidate can
replace the current `p=3.90e-05` development incumbent.

Artifacts:

- `runs/generalized_solver/n11_residual_cem_n10center_16s.json`
- `runs/generalized_solver/n11_residual_cem_n10center_16s_fddp_warm.json`
- `runs/generalized_solver/n11_cem16_fddp_refined.json`

### Near-locked release boundary and wide-rail curriculum (2026-09-15)

The accepted split-release incumbent at `p=3.125e-5` was carried into the
next frozen plant at `p=3.3203125e-5` with stronger terminal cart, velocity,
angle, and rail penalties. The exact-MuJoCo Box-FDDP refinement became
numerically unstable after `14` iterations and ended at `3.088 m` cart
excursion with no upright hold. This does not improve the incumbent and does
not establish a free-chain result. The accepted `p=3.125e-5` artifact remains
an assisted, near-locked development result only.

The rail-length hypothesis was then isolated as a two-stage curriculum. A
`+/-12 m` rail was used for the split-release continuation, while the locked
baseline retained the known feedback replay rather than being re-optimized.
The replay baseline passed (`22.12 s` hold, `2.0233 m` peak cart excursion),
but the first releases at `p=0.001` and `p=0.0005` both drove the cart to the
wide rail (`12.02-12.09 m`) and failed capture. Waypoint repair at `24` and
`48` segment steps could not preserve the endpoint, and the subsequent FDDP
traces produced MuJoCo instability warnings. The run was stopped after the
same failure signature repeated; it is not evidence for any link count.

Decision: a longer rail is not sufficient to release the 11th joint. Preserve
the replay-protected baseline rule because re-optimizing a known-good route
can create a false negative, but do not spend more compute on rail widening
alone. The next valid lever must change the release/capture representation or
the continuation geometry, and must still finish on the canonical `+/-3 m`
uniform plant with independent replay. This is now a playbook hard negative:
wide-rail discovery cannot substitute for a canonical capture basin.

Artifacts:

- `runs/generalized_solver/n11_split_boundary_000033203_fddp_rail20k.json`
- `runs/generalized_solver/n11_locked_split_rail12_continuation.yaml`
- `runs/generalized_solver/n11_locked_split_rail12_homotopy/continuation.json`
- `runs/generalized_solver/n11_locked_split_rail12_homotopy/trials/locked_split_baseline_pass1.json`
- `runs/generalized_solver/n11_locked_split_rail12_replay_homotopy/continuation.json`
- `runs/generalized_solver/n11_locked_split_rail12_replay_homotopy/trials/trial_0001_p0.001000000_replay_only.json`
- `runs/generalized_solver/n11_locked_split_rail12_replay_homotopy/trials/trial_0002_p0.000500000_replay_only.json`

### Adjacent local repair at the `p=3.90e-5` boundary (2026-09-15)

The strongest assisted release controller at `p=3.90e-5` was replayed on the
adjacent `p=4.00e-5` plant before any optimizer was allowed to change it. The
feedback replay reached `3.042 m` and held for `0 s`. A `300`-iteration
Box-FDDP repair with rate-heavy terminal penalties improved the peak to
`3.021 m`, but stopped after `19` iterations with MuJoCo instability warnings
and no hold. A bounded `50`-iteration residual CEM around the same force route
also failed to produce a capture state: its best point at `6.64 s` had
`1.334 rad` maximum angle, `4.033 rad/s` hinge RMS, `3.754 rad/s` absolute-rate
RMS, `0.873 m/s` cart speed, and capture-constraint violation `98.0`.

Decision: the release cliff is not solved by simply reusing the preceding
route, increasing local FDDP iterations, or adding a low-dimensional force
residual. The `p=3.90e-5` route remains a curriculum incumbent only. The next
attempt must provide a new capture basin or a new continuation coordinate while
retaining the exact replay and trace-integrity gates.

Artifacts:

- `runs/generalized_solver/n11_algebraic_split_release_probe/p4.00e-05_replay_from_p3p9.json`
- `runs/generalized_solver/n11_algebraic_split_release_probe/p4.00e-05_fddp_from_p3p9.json`
- `runs/generalized_solver/n11_algebraic_split_release_probe/p4.00e-05_residual_cem_from_p3p9.json`

### Real-handoff prefix plus LQR capture probe (2026-09-15)

To isolate the capture expert from the release boundary, a saved near-settled
state from `n11_p39_low_momentum_handoff_states.json` was replayed on the free
uniform target. A `2.0 s` bounded prefix CEM was followed by the exact upright
LQR for a total of `8.0 s`; all candidates were evaluated in one uninterrupted
MuJoCo episode. The search did not improve after `50` iterations. Its best
candidate held upright for only `0.62 s`, reached `3.052 m`, ended in a rail
violation, and still had terminal maximum angle `2.827 rad` with absolute rate
RMS `47.354 rad/s`.

Decision: even a genuinely quiet handoff from the assisted route does not make
the free uniform 11-link LQR tail usable. The capture expert remains the
primary unresolved dependency; future work should learn or optimize a
nonlinear modal stabilizer before switching to LQR, and must evaluate it from
the real saved state bank rather than from synthetic upright states.

Artifact:

- `runs/generalized_solver/n11_uniform_prefix_lqr_state0.json`

### Modal partial-feedback-linearization probe (2026-09-15)

The repository's count-independent partial-feedback-linearization controller
was tested on the canonical uniform 11-link plant with its normal-mode energy
and internal-mode damping terms enabled. A `24`-iteration, `32`-member CEM ran
for `20 s` on the canonical `+/-3 m` rail and used the modal handoff objective
plus exact upright LQR capture. The best exact replay never entered the upright
set, reached `3.031 m`, and terminated by rail violation with `0 s` upright
streak.

Decision: modal energy shaping is not a sufficient 11-link swing-up/capture
expert in its current parameterization. Keep the implementation as a bounded
diagnostic, but do not combine it with the accepted route or use its shaped
score as evidence. A future mode-aware method needs a better release-aware
terminal objective and a demonstrated nonlinear capture basin.

Artifact:

- `runs/generalized_solver/n11_modal_pfl_cem.json`

### Fine release-boundary retries, modal phase seed, and sparse shooting (2026-09-15)

The split-count continuation was resumed below the previously recorded
`p=3.125e-5` frontier with steps down to `1e-7`. The inherited route was
tested at `p=3.3203125e-5`, `3.22265625e-5`, `3.173828125e-5`,
`3.1494140625e-5`, and `3.13720703125e-5`; each replay reached a rail
violation with `0 s` upright hold. The last trial required approximately
`3.219 m` of half-rail while the canonical plant provides `3.0 m`. The
manifest is therefore closed at a minimum-step frontier, not promoted as a
solution. This sharpens the result: reducing the continuation step does not
cross the free-capture basin.

An analytic first-mode phase seed was then materialized as a diagnostic warm
start. It had linear residual `1.44e-5`, but its exact replay reached
`3.073 rad` maximum angle, `3.020 m` cart excursion, and `0 s` upright hold.
The modal seed is useful for interpreting the phase geometry, not as a
controller or a solution artifact.

The free-target Box-FDDP route was also given an explicit running penalty on
the newly inserted relative angle and hinge rate. Both a fresh warm start and
the `p=3.90e-5` route were tried, including an initially feasible variant.
All three runs stopped early with MuJoCo instability warnings, no upright
hold, and cart excursions between roughly `3.01` and `3.07 m`. A sparse
multiple-shooting repair from the same `p=3.90e-5` route was bounded to `80`
evaluations; it ended with node norm `33.764`, maximum defect `9.46e-3`,
terminal value `4.18e18`, and `0 s` hold. These are numerical diagnostics,
not evidence of an 11-link capture route.

Decision: the playbook now rejects three tempting shortcuts at this boundary:
smaller homotopy steps alone, an analytic modal phase seed alone, and a local
running-cost or sparse-shooting repair around the assisted route. The
accepted near-locked route remains useful only as a curriculum source. The
next experiment must create a materially different free-chain capture basin
or transfer a longer, settled 10-link route with a new release-aware terminal
representation. Every candidate still requires canonical-rail replay,
trace-integrity checks, disjoint noisy gates, held-out video, and fresh-clone
reproduction before promotion.

Artifacts:

- `runs/generalized_solver/n11_locked_split_homotopy_retry/continuation.json`
- `runs/generalized_solver/n11_locked_split_homotopy_retry/trials/trial_0016_p0.000031372_pass2.json`
- `runs/generalized_solver/n11_modal_phase_seed_16s.json`
- `runs/generalized_solver/n11_free_split_cost_a10_r1.json`
- `runs/generalized_solver/n11_free_split_cost_a10_r1_feasible.json`
- `runs/generalized_solver/n11_free_split_cost_p39_a100_r10.json`
- `runs/generalized_solver/n11_free_multiple_shooting_p39_seg20.json`

### Settling-tail transfer and interior split-position probe (2026-09-15)

The distinct ten-link endpoint-tail refinement was transferred into the
locked distal-split 11-link plant and replayed on the canonical uniform
target. The first quick replay was discarded because it omitted saved
feedback (`tracking_gain=0`). The corrected replay with
`tracking_gain=1` reached `3.037 m` and held for `0 s`. A
four-second appended settling/capture tail with target-plant feedback rebuild,
relative inserted-angle/rate stage penalties, and a deferred horizon handoff
stopped after `18` Box-FDDP iterations with a terminal value of approximately
`6.82e15`, `3.015 m` cart excursion, and `0 s` hold. Extra time after the
10-link arrival is therefore not sufficient by itself.

The geometry lever was then changed: the additional joint was inserted at
source link `5` instead of the distal link. A config-aware transfer was added
in `scripts/embed_split_position_route.py`, using the same physical lift,
absolute-coordinate transform, feedback projection, and resampling math as
the validated generalized split path. This avoids treating a distal-only
adapter as proof for an interior split. The corrected locked-start replay
with saved feedback reached `3.023 m` and held only `0.84 s`; one exact
Box-FDDP repair stopped after `27` iterations with MuJoCo instability
warnings, `3.010 m` cart excursion, and `0.20 s` hold.

Decision: the longer-tail hypothesis and this interior split geometry are
rejected as 11-link capture routes. The interior result is now a valid
negative control because it survives the config-aware embedding and a bounded
target-plant repair. Keep the new helper for future split-position studies,
but return the active search to release-aware distal continuation or another
materially different free-chain capture representation. The optional split
angle/rate weights in `scripts/search_fddp_capture.py` are now explicitly
allowed to be zero in replay-only runs; the required solver thresholds remain
strictly positive.

Artifacts:

- `runs/generalized_solver/n11_n10_tail100_route_embedded.json`
- `runs/generalized_solver/n11_n10_tail100_direct_replay.json`
- `runs/generalized_solver/n11_n10_tail100_direct_replay_feedback1.json`
- `runs/generalized_solver/n11_n10_tail100_fddp_tail4.json`
- `configs/swingup11_split_link5_continuation.yaml`
- `runs/generalized_solver/n11_split_link5_n10_route_embedded_correct.json`
- `runs/generalized_solver/n11_split_link5_baseline_replay_correct.json`
- `runs/generalized_solver/n11_split_link5_baseline_replay_feedback1.json`
- `runs/generalized_solver/n11_split_link5_locked_fddp_repair.json`
- `scripts/embed_split_position_route.py`

### Capture dependency isolated with the real handoff bank (2026-09-15)

The saved `n11_p39_low_momentum_handoff_states.json` bank contains `24`
states selected from a real assisted swing-up rollout, not synthetic upright
states. Every one of those states was evaluated on the canonical uniform
11-link target with the exact upright LQR for `8 s`: `24/24` reached the full
tail, with `8.02 s` maximum upright/low-momentum hold and only `0.012 m`
maximum cart excursion. The repeatable artifact is
`runs/generalized_solver/n11_uniform_handoff_bank_lqr8s.json`.

This isolates the dependency cleanly. The canonical 11-link capture expert
can hold the real handoff-state bank; the unresolved problem is producing one
of those states from the hanging start on the uniform plant. This is a useful
positive component result, but it is not a swing-up claim.

The exact FDDP tool now accepts `--terminal-target-json` and
`--terminal-target-index`, so a swing-up route can optimize toward a measured
handoff state rather than only an abstract zero/Lyapunov endpoint. Three
targeted attempts were retained. A saved-feedback rebuild became unstable
before optimization; an open-loop rebuild stopped after `15` iterations at
`3.013 m` with `0 s` hold; and an eight-second appended horizon stopped after
`15` iterations at `3.004 m` with `0 s` hold and a MuJoCo instability warning.
The endpoint target is therefore useful for evaluation and future global
search, but it does not repair the transferred p39 swing route locally.

Decision: keep the two-expert decomposition. Treat the 24-state bank plus
LQR as the capture reference and focus the next search on a genuinely new
canonical swing-up expert that reaches this bank. Do not substitute this
component gate for the hanging-start 20/100 benchmark.

Artifacts:

- `scripts/evaluate_handoff_bank_lqr.py`
- `runs/generalized_solver/n11_uniform_handoff_bank_lqr8s.json`
- `runs/generalized_solver/n11_fddp_measured_handoff_target.json`
- `runs/generalized_solver/n11_fddp_measured_handoff_target_openloop_init.json`
- `runs/generalized_solver/n11_fddp_measured_handoff_target_h16_openloop.json`
- `runs/generalized_solver/n11_p39_uniform_replay_feedback1.json`

### Phase-scheduled inserted-mode feedback and ordered morphology gradient (2026-09-15)

The split-joint residual diagnostic was extended from an always-on/start-only
correction to an explicit `[start_step, end_step)` phase window. Five
canonical uniform replay variants and three adjacent `p=4.00e-5` release-cliff
variants were tested. The best canonical replay reached `3.003 m` but held for
`0 s`; the best local FDDP repair stopped after `17` iterations at `3.066 m`
with `0 s` hold. On the release cliff, the strongest variant reached only
`3.042 m`, and the bounded repair stopped after `20` iterations at `3.063 m`
with `0 s` hold. Phase-scheduled active support does not cross the boundary
under these gains.

The morphology-gradient order was then changed so the distal split remained
locked while lengths and masses moved to the uniform 11-link endpoint. The
locked baseline still reproduced `22.12 s` at `2.023 m`, but a `12.5%`
morphology jump collapsed into a trace-integrity failure and immediate FDDP
`QACC` instability. A direct `1%` replay reached `3.081 m` with `0 s` hold;
its `80`-iteration repair stopped after `20` iterations at `3.043 m` with a
MuJoCo warning. The adaptive ledger was operator-stopped and marked
`interrupted_invalid_trial`, not left as a running or accepted campaign.

Decision: phase-scheduled residual feedback and “morphology first, release
second” are recorded as negative controls for this implementation. The
positive handoff-bank screen remains the guide for the next global swing-up
search. The playbook rule is unchanged: every route must be trace-integrity
clean on the canonical rail before it can influence public evidence.

Artifacts:

- `scripts/add_split_joint_feedback.py`
- `runs/generalized_solver/early_a-1_r1_replay.json`
- `runs/generalized_solver/early_a-5_r5_replay.json`
- `runs/generalized_solver/early_a-20_r20_replay.json`
- `runs/generalized_solver/full_a-1_r1_replay.json`
- `runs/generalized_solver/late_a-5_r5_replay.json`
- `runs/generalized_solver/n11_phase_feedback_fddp_repair.json`
- `runs/generalized_solver/p4_release_early_a-5_r5_replay.json`
- `runs/generalized_solver/p4_release_early_a-20_r20_replay.json`
- `runs/generalized_solver/p4_release_full_a-1_r1_replay.json`
- `runs/generalized_solver/n11_p4_phase_feedback_fddp.json`
- `configs/swingup11_locked_morphology_only.yaml`
- `runs/generalized_solver/n11_locked_morphology_only_homotopy/continuation.json`
- `runs/generalized_solver/n11_locked_morphology_only_p001_replay.json`
- `runs/generalized_solver/n11_locked_morphology_only_p001_fddp.json`

### Measured-handoff-bank CEM and full-trace warm-start audit (2026-09-15)

The positive 24-state LQR bank was used as an explicit terminal manifold for
a new serial exact-MuJoCo CEM swing search. A fresh hanging-start search stayed
at the hanging equilibrium, confirming that a terminal target alone does not
create swing energy. Residual searches around the assisted route improved the
nearest-bank distance to roughly `5.60` on a temporary `+/-6 m` rail, but the
best state still had large angle/rate error and the canonical replay exited at
`3.055 m` during the LQR tail with `0 s` hold.

The search then exposed and fixed a warm-start bookkeeping issue: the p39
artifact's `controller.controls` contains only the eight-second swing phase,
while its complete `result.trajectory` contains the later LQR actions. Earlier
residual runs appended zeros after the swing phase. The corrected loader now
prefers the full recorded trace and has a regression test. A corrected
full-trace CEM on the `+/-6 m` discovery rail still failed its canonical
replay at `3.008 m` with `0 s` hold. Restoring the canonical rail inside the
optimizer did not retain a viable population.

Decision: target-bank scoring is retained as the right terminal diagnostic,
but open-loop CEM plus a temporary rail is not the missing 11-link swing or
capture expert. The next direct-policy probe must be judged on the canonical
plant and may not promote any discovery-rail score.

Artifacts:

- `scripts/search_swingup_handoff_bank_cem.py`
- `scripts/evaluate_handoff_bank_route.py`
- `tests/test_search_swingup_handoff_bank_cem.py`
- `runs/generalized_solver/n11_handoff_bank_cem_fresh16.json`
- `runs/generalized_solver/n11_handoff_bank_cem_residual_p39_rail6.json`
- `runs/generalized_solver/n11_handoff_bank_cem_residual_p39_rail6_canonical_replay.json`
- `runs/generalized_solver/n11_handoff_bank_cem_fulltrace_rail6.json`
- `runs/generalized_solver/n11_handoff_bank_cem_fulltrace_rail6_canonical_replay.json`
- `runs/generalized_solver/n11_handoff_bank_cem_rail6_to_canonical.json`

### Direct uniform PPO policy-capacity probe (2026-09-15)

Two bounded PPO probes were run against the canonical uniform 11-link plant
from the hanging start. The first used the configured curriculum for `200`
updates and reached the full target morphology only at the end; its final
eight-episode target evaluation was `0/8` with mean return `-827.8`. The second
removed that confounder and trained directly at target morphology for `600`
updates (`1,228,800` environment steps, evaluation every `25` updates). Every
recorded target evaluation remained at `0.00` success. By the final update the
mean episode length had collapsed to `18.8` steps, with mean evaluation return
`-293.6`, indicating an early-failure policy rather than a swing-up policy.

Decision: archive both PPO runs as negative controls for the current generic
policy/objective setup. They do not weaken the positive capture result: exact
upright LQR still holds all `24/24` real handoff states on the canonical
uniform plant. They do show that a fresh direct PPO learner is not discovering
the missing hanging-start reachability problem at this scale. The next search
must therefore preserve the solved 7-to-10-link two-expert structure and add an
explicit reachability/teacher signal for entering the measured handoff bank;
another uninitialized full-target PPO run is not justified without a material
change to the observation, reward, or initialization.

Artifacts:

- `runs/generalized_solver/n11_uniform_ppo_probe/train_log.csv`
- `runs/generalized_solver/n11_uniform_ppo_direct_p1/train_log.csv`
- `runs/generalized_solver/n11_uniform_ppo_direct_p1/config.resolved.yaml`

### Recorded-state route distillation and PPO warm start (2026-09-15)

The near-locked p39 artifact was used as a teacher in a state-feedback
experiment. The first implementation replayed the saved action list through
the current source XML to reconstruct pre-action observations. That replay
matched the beginning of the trace but accumulated `0.212 m` of cart error by
the end, so `n11_route_imitation_teacher` is superseded bookkeeping and is not
clean route evidence.

The loader was corrected to use the artifact's recorded states directly:
selected initial state plus each recorded post-action state paired with the
next action. The corrected teacher has `1,500` labels and a best validation MSE
of about `0.0211`, but its canonical hanging-start rollout still reached
`0/1` success, no upright event, and a `3.028 m` rail exit at `5.46 s`.

A low-exploration direct-target PPO continuation was also run from the
superseded replay-based teacher (`600` updates, `1,228,800` environment steps,
time-aware observation). Its target evaluation remained `0.00` success; the
final evaluation return was `-381.8` with a `61`-step mean episode length.
Because its initializer was later invalidated by the trace-reconstruction
audit, this run is retained only as a diagnostic, not as a clean teacher
result.

Decision: recorded route imitation does not by itself bridge the free-chain
release. Do not spend another long PPO run on the same setup. The next route
search must optimize closed-loop reachability into the measured handoff bank
on the canonical plant, with the bank distance and exact capture replay kept
as separate gates.

Artifacts:

- `scripts/distill_swing_route_policy.py`
- `tests/test_distill_swing_route_policy.py`
- `runs/generalized_solver/n11_route_imitation_teacher/`
- `runs/generalized_solver/n11_route_imitation_teacher_recorded/`
- `runs/generalized_solver/n11_route_imitation_ppo_p1/`

### Wider-rail route-teacher PPO diagnostic (2026-09-15)

The corrected recorded-state teacher was retrained and continued with a
`+/-12 m` discovery rail, time-aware observations, zero curriculum advance,
`300` updates, `16` environments, and `614,400` environment steps. The final
held-out evaluation remained `0.00` success with a return of `-1343.0`; no
upright event or canonical evidence was produced. The wider rail therefore
did not repair the teacher's closed-loop mismatch. This is a bounded
discovery diagnostic, not a canonical failure attributable to the rail size.

Decision: preserve the wide rail only as a training-wheel option. The next
experiment must put the route in the action loop as a closed-loop residual and
score whether it reaches the measured handoff bank, rather than applying more
unstructured PPO updates to the same actor.

Artifacts:

- `runs/generalized_solver/n11_route_imitation_ppo_rail12/`

### Closed-loop route-residual CEM and energy/modal probes (2026-09-15)

The route was then kept in the action loop instead of being distilled into a
free actor. A serial exact-MuJoCo CEM searched a 48-knot additive residual
around the saved p39 time-varying feedback route, scoring the real handoff
bank. On the `+/-12 m` rail, `60` iterations with `32` candidates each still
rail-terminated every elite; the best late bank distance was `8.35` and the
maximum cart excursion was `12.01 m`. A second run added a per-step
cart-centering cost, used a `+/-20 m` rail and half-strength route feedback,
and found full-horizon candidates, but the best bank distance was still `8.34`
(`8.22` at the terminal state), with `18.52 m` maximum cart excursion and no
upright event. Its exact canonical replay rail-terminated after `52` steps at
`3.034 m`.

As an independent controller-family probe, the low-dimensional exact
mass-matrix energy-shaping CEM ran `31` generations (`48` candidates per
generation) for `20 s` on the `+/-20 m` rail. Its best score was `3166.5`; the
best recorded state had `1.172 rad` maximum absolute angle and a centered
endpoint, but the run used `12.49 m` of rail and never entered upright hold.
Eight existing modal-coherence gain pairs were also screened; none produced
an upright event. These probes support the current diagnosis: cart centering
and energy injection help individually, but neither has yet synchronized the
11 internal modes into the measured capture basin.

Decision: keep the route-residual objective and exact replay utility as
reusable diagnostics, but do not promote either family. The next search must
couple route reachability to a terminal capture value or a staged, measured
handoff curriculum; another generic PPO or unshaped open-loop search is not
justified.

Artifacts:

- `scripts/search_swingup_handoff_bank_cem.py`
- `scripts/evaluate_route_residual_candidate.py`
- `tests/test_search_swingup_handoff_bank_cem.py`
- `runs/generalized_solver/n11_route_feedback_residual_cem_rail12.json`
- `runs/generalized_solver/n11_route_feedback_residual_cem_rail20_centered.json`
- `runs/generalized_solver/n11_route_feedback_residual_cem_rail20_centered_replay.json`
- `runs/generalized_solver/n11_route_feedback_residual_cem_rail20_centered_canonical_replay.json`
- `runs/generalized_solver/n11_energy_shaping_cem_rail20.json`

### Staged handoff-bank route-residual PPO (2026-09-15)

The measured handoff bank was then exposed as an explicit curriculum signal.
The actor received a residual around the saved p39 feedback route, plus route
phase and the normalized distance to the nearest real handoff state. Training
used a `+/-20 m` discovery rail, `16` environments, `600` updates, and
`1,228,800` environment steps. This was a direct test of the two-expert idea:
learn the missing swing-up/reachability behavior while retaining the already
verified capture basin.

The run completed without numerical failure, but every target evaluation was
`0.00` success. The selected best checkpoint was replayed deterministically.
On the development rail it reached `0/1` success, never became upright, and
terminated at `122` steps on a `20.028 m` rail excursion. Replaying the same
weights on the canonical `+/-3 m` rail reached `0/1` success and terminated at
`52` steps with `x=3.013 m`; it never entered upright or capture. The final
training evaluation was `0.00` success with return `-1395.4`, and the best
checkpoint replay was `return=-811.5` on the wide rail. This closes the
staged residual-PPO variant as a negative result: a bank-distance reward and
route residual did not provide enough reachability signal for the free
11-link plant, and widening the rail only allowed the policy to spend more
distance before failure.

Decision: keep the handoff-bank evaluator, route wrapper, and exact replay
artifacts as reusable infrastructure, but do not treat this as an 11-link
result or as evidence for a public-frontier update. The next method must make
the swing-up phase itself more controllable or identify a genuinely reachable
terminal funnel; repeatedly increasing PPO duration, rail width, or bank
reward would repeat a falsified setup.

Artifacts:

- `configs/swingup11_route_residual_handoff.yaml`
- `src/gcartpole/route_feedback.py`
- `tests/test_route_feedback.py`
- `runs/generalized_solver/n11_route_residual_handoff_ppo_rail20/train_log.csv`
- `runs/generalized_solver/n11_route_residual_handoff_ppo_rail20/checkpoints/best.safetensors`
- `runs/generalized_solver/n11_route_residual_handoff_ppo_rail20/eval_rail20.json`
- `runs/generalized_solver/n11_route_residual_handoff_ppo_rail20/eval_canonical_rail3.json`

### LQR saturation and morphology-gradient curriculum probes (2026-09-15)

The next tests returned to the mechanism used by the 7-to-10 campaign: an
upright finite-difference LQR teacher, with a `policy_on_saturation` escape so
the learned actor receives full authority when the local stabilizer clips.
The first uniform-plant run used a quadratic hanging-angle curriculum and a
`+/-12 m` rail. It passed the `progress=0.05` stage (`1.00` evaluation
success and a `20.02 s` upright streak), but the `progress=0.10` evaluation
fell to `0/8` success with only a `0.20 s` maximum upright streak. A separate
route-residual schedule with the same LQR warm start showed the same narrow
basin: the LQR-only handoff failed at `progress=0.10`, and switching to the
route at `progress=0.05` produced `0/8` after `100` updates.

The morphology-gradient variant copied the 7-link training wheels: heavier
base links, higher damping, nonzero friction, and a longer rail annealed back
to the uniform target. It did not repair the mismatch. A deterministic zero
policy replay of the fixed uniform LQR gain already failed on the easier
gradient plant at `progress=0.025`, and the PPO run remained at
`progress=0.05` with `0/8` success at update `150`. The target-plant direct
probe remains useful evidence: the same gain holds the uniform plant through
`progress=0.05`, but not `0.10`.

Decision: the inherited LQR-plus-saturation architecture is valid near the
upright equilibrium but does not, by itself, bridge the n=11 nonlinear basin.
Do not present any of these curriculum checkpoints as swing-up evidence. The
route scheduler, LQR warm-start selector, and gradient configs remain
available for future work, but the next serious search needs a better
full-state swing-up teacher or a model-based reachable funnel rather than
more PPO updates around this same local anchor.

Artifacts:

- `configs/swingup11_route_residual_angle_curriculum.yaml`
- `configs/swingup11_lqr_saturation_curriculum.yaml`
- `configs/swingup11_lqr_saturation_gradient.yaml`
- `runs/generalized_solver/n11_route_residual_angle_curriculum_ppo_rail12/`
- `runs/generalized_solver/n11_lqr_saturation_curriculum_ppo_rail12/`
- `runs/generalized_solver/n11_lqr_saturation_gradient_ppo_rail12/`

### Capture-chain route CEM, direct-force CEM, and parked-route transfer (2026-09-15)

The next route search coupled the cart-position swing expert to the actual
capture/stabilize chain. This was intended to prevent a transient upright
angle from being mistaken for a usable handoff. A two-iteration smoke test on
a `+/-12 m` discovery rail confirmed that the unmodified lower-count route
never reached the 11-link capture gate. A subsequent exact-MuJoCo trajectory
CEM (`16` generations, `48` candidates, `24 s`) improved the late handoff
candidate to `0.2575 rad` maximum angle, `1.065 rad/s` hinge RMS,
`2.148 rad/s` cumulative absolute-rate RMS, `0.758 m/s` cart speed, and
`2.880 m` maximum cart excursion. It did not hold.

Initializing the coupled capture-chain CEM from that candidate produced an
angle-only crossing at `0.1179 rad` and entered the capture phase, but the
actual handoff had `4.454 rad/s` hinge RMS, `1.806 m/s` cart speed, and the
rollout used `12.139 m` of rail. The LQR capture expert held for only `0.02 s`.
This is a hard negative for angle-only chain gating: the route reached the
visual top while carrying too much internal motion.

A direct force-knot CEM was then initialized from the same route (`14`
generations, `40` candidates, `101` action knots, `20 s`) with an absolute
velocity-aware cold-handoff score. Its best late point was only `1.492 rad`
from upright, with `1.374 rad/s` hinge RMS, `1.428 rad/s` absolute-rate RMS,
`3.057 m/s` cart speed, and `12.089 m` exploratory-rail use. A higher-weight
cart-route refinement also regressed to `0.779 rad` and never entered a
usable handoff. The extra cart-position layer is not the only problem, but
direct open-loop force knots do not solve it either.

The 10-link inheritance test was run at the highest-fidelity architecture:
the 10-link feedback route was transferred to 11 links, followed by target-
plant Box-FDDP and the same parked-target LQR contract. The transferred-
nominal pipeline became unstable after `13` FDDP iterations and replayed to a
`3.108 m` rail violation. Rebuilding the target-plant feedback warm start at
`0.2` scale did not help; the replay violated at `3.084 m` after `14`
iterations. Neither route was packaged or promoted.

Decision: retain the coupled-chain evaluator and the late handoff metrics,
but reject angle-only capture, direct force knots, and unrefined 10-to-11
transfer as solutions. The next free-chain method must reduce internal rate
before the capture switch, not merely reach a smaller maximum angle or spend
more exploratory rail.

Artifacts:

- `scripts/search_swingup_chain.py`
- `scripts/search_swingup_trajectory.py`
- `scripts/search_swingup_action_sequence.py`
- `runs/generalized_solver/n11_chain_capture_cem_smoke/`
- `runs/generalized_solver/n11_trajectory_capture_ready_cem_rail12/`
- `runs/generalized_solver/n11_chain_capture_cem_from_trajectory_rail12/`
- `runs/generalized_solver/n11_direct_force_cold_handoff_cem_rail12/`
- `runs/generalized_solver/n11_trajectory_cold_handoff_refine_rail12/`
- `runs/generalized_solver/n11_nominal_transfer_pipeline_v2/`
- `runs/generalized_solver/n11_rebuilt_transfer_pipeline_v3/`

### Release-aware split curriculum resume (2026-09-15)

The strongest existing teacher was the exact locked split-link route, so its
adaptive release ledger was resumed rather than replaced. The ninth source
link split configuration starts with the inserted joint locked and releases
it through the exact MuJoCo target plant. Each proposed progress value uses
waypoint repair, Box-FDDP, full-state replay, rail checks, and the existing
`22.12 s` hold gate.

The resumed ledger retained `19` trials. It accepted
`p=2.05078125e-5`, `p=2.0751953125e-5`, and `p=2.08740234375e-5`, each with a
`22.12 s` exact hold and approximately `2.077 m` maximum cart excursion. The
next value, `p=2.099609375e-5`, failed repeatedly: waypoint repair could make
an endpoint, but exact Box-FDDP replay lost the hold and crossed the `+/-3 m`
rail (`3.039 m` in the final retry). The ledger stopped at its configured
minimum step with accepted progress `2.08740234375e-5` and failed upper bound
`2.099609375e-5`.

This is useful evidence that the release boundary is real and reproducible,
but it is not a canonical 11-link result. The inserted joint is still almost
fully supported at this progress, and the run has not reached the free
uniform endpoint `p=1`. Do not use its hold, route, or video as public
11-link evidence. The next attempt should change the release parameterization
or add a full-state reachable funnel before resuming another expensive
bisection; repeating the same step below the current boundary has already
been tested.

Artifacts:

- `runs/generalized_solver/n11_forced_split_link9.yaml`
- `runs/generalized_solver/n11_forced_split_link9_warm.json`
- `runs/generalized_solver/n11_forced_split_link9_homotopy/continuation.json`
- `runs/generalized_solver/n11_forced_split_link9_homotopy/trials/trial_0016_p0.000020752_waypoint_x4_fddp.json`
- `runs/generalized_solver/n11_forced_split_link9_homotopy/trials/trial_0018_p0.000020874_waypoint_x4_fddp.json`
- `runs/generalized_solver/n11_forced_split_link9_homotopy/trials/trial_0019_p0.000020996_pass2.json`

The playbook rule remains: a curriculum state may become a teacher seed, but
the frontier advances only after a free uniform hanging-start route passes the
canonical 20/100 gates, exact replay, hashes, and reset-free video.

### Settled-launch audit of inserted-joint feedback (2026-09-15)

The first 81-point inserted-joint angle/rate feedback screen was replayed from
the raw noisy hanging reset with `conditioning_seconds=0`. That is not the
7-to-10 controller contract: the released method first parks the cart and lets
the chain settle under the hanging-equilibrium LQR. The raw-start screen is
therefore retained as a diagnostic, not used as a settled-launch comparison.

The comparison was corrected by adding the full `16 s` hanging-LQR conditioning
phase, while keeping the target plant uniform n=11, the canonical `+/-3 m`
rail, the same inherited 10-to-11 route, `tracking_gain_scale=1`, and the
same exact replay evaluator. The exact locked-split route and representative
constant inserted-joint residuals (`angle/rate` pairs including
`(0.5,1)`, `(0,1)`, `(1,1)`, `(5,5)`, and `(0.5,5)`) all remained at `0/1`:
they never reached the upright/capture gate and exited at roughly
`3.06-3.29 m` required rail. The full grid cannot be promoted; the corrected
subset is sufficient to close the constant-feedback family because the result
does not change from the raw-start screen after the actual launch contract is
restored.

Decision: keep the constant residual utility as a regression diagnostic, but do
not spend more search budget on one fixed angle/rate pair. The missing n=11
mode needs phase-dependent authority or a new target-plant swing trajectory.
The next experiment is an exact serial MuJoCo action-route search initialized
from the inherited route, with zero reset noise inside the optimizer and a
separate settled-launch replay afterward. Any discovery-rail or raw-search
candidate remains development-only until the canonical replay passes.

Artifacts:

- `scripts/add_split_joint_feedback.py`
- `tests/test_add_split_joint_feedback.py`
- `runs/generalized_solver/n11_inserted_joint_feedback_grid/`
- `runs/generalized_solver/n11_locked_split_exact_embed_warm.json`
- `tmp/n11_exact_split9_settled_eval.json`
- `tmp/n11_grid_settled_a0p5m1.json`
- `tmp/n11_grid_settled_a0m1.json`
- `tmp/n11_grid_settled_a1m1.json`
- `tmp/n11_grid_settled_a5m5.json`
- `tmp/n11_grid_settled_a0p5m5.json`

### Rate-heavy exact CEM and target-plant refinement (2026-09-15)

To test phase-varying action authority without another policy-learning run, a
serial exact-MuJoCo CEM searched a 16-second route around the best existing n11
force proposal. It used `64` action knots, `32` candidates, `48` iterations,
canonical uniform morphology, and a `+/-3 m` rail preference with strong
terminal angle, hinge-rate, absolute-rate, cart, and capture-barrier terms.
The best measured transient stayed inside the rail and reached `1.0587 rad`
maximum angle at `15.48 s`, but still carried `3.0004 rad/s` hinge RMS,
`4.3268 rad/s` absolute-rate RMS, `0.8906 m/s` cart speed, and capture
constraint violation `36.26`. It was not a handoff.

The exact route was converted into a dynamically consistent Box-FDDP warm
start. Target-plant refinement stopped at iteration `16` with terminal cost
`2.46e14`; exact replay exited at `3.055 m` with no upright or capture event.
A separate six-second Box-FDDP solve from the CEM's best measured transient
also stopped at iteration `16` and exited at `3.036 m` with no latch. Thus the
best CEM transient is an angle/rate diagnostic, not a swing-up-plus-capture
controller.

Decision: close this particular residual-CEM-to-FDDP branch. The next
permitted search must change the route representation or provide a measured
reachable funnel; increasing CEM generations around the same force route or
refining the same high-rate transient repeats a falsified basin. No n=11
frontier advancement, README/About change, video, or public push is justified.

Artifacts:

- `runs/generalized_solver/n11_global_cem_from_residual16_rateheavy.json`
- `runs/generalized_solver/n11_global_cem_from_residual16_rateheavy_warm.json`
- `runs/generalized_solver/n11_global_cem_from_residual16_rateheavy_fddp.json`
- `runs/generalized_solver/n11_global_cem_best_state_capture_fddp.json`

### Free split with temporary distal damping (2026-09-15)

The next morphology-gradient test used a physically free n=11 split rather
than a locked or supported ghost. At progress `p=0`, the last two links were
`0.15 m`/`0.05 kg` each, all eleven joints had zero lock and zero stiffness,
and only the final joint received temporary damping `0.01`; the target
endpoint is the canonical uniform n=11 plant. The inherited split route was
first evaluated after the normal `16 s` hanging-LQR conditioning phase. It
hit the `+/-3 m` rail at `3.0593 m`, with no upright or handoff event.

A bounded target-plant Box-FDDP repair was then attempted with explicit
inserted-joint angle/rate stage costs, `8 s` horizon, and the same physical
rail. The first invocation omitted `--progress 0` and therefore optimized the
uniform endpoint because `search_fddp_capture.py` defaults to progress `1.0`;
that artifact is rejected as evidence and retained only as a provenance
regression example. The corrected run pinned both plant and LQR progress to
`0.0`: Box-FDDP stopped after `7` iterations, exact replay exited at
`3.0261 m`, reached `0 s` upright hold, and recorded `30.77 rad/s` hinge-rate
RMS and `48.47 rad/s` absolute-rate RMS. No continuation step was accepted.

Decision: close this specific free-damping branch. Temporary damping without
a reachable target-plant swing route does not preserve the 10-link incumbent,
and the inserted mode becomes violently unstable under local repair. Keep the
configuration as a negative-control curriculum artifact, but do not use its
route, replay, or any wider-rail variant as n=11 evidence. Future morphology
experiments must pin `--progress` and `--lqr-progress` explicitly and must
pass the free p=0 replay before spending budget on a homotopy.

Artifacts:

- `runs/generalized_solver/n11_damped_unlock_relax.yaml`
- `tmp/n11_damped_unlock_relax_p0_eval.json`
- `runs/generalized_solver/n11_damped_unlock_relax_p0_fddp.json` (rejected: defaulted to target progress)
- `runs/generalized_solver/n11_damped_unlock_relax_p0_fddp_correct.json`

### Late-tail iLQR on the rate-heavy n=11 route (2026-09-15)

The next route representation kept the inherited exact force route through
`12 s`, then optimized only the final `4 s` with exact-MuJoCo iLQR. The tail
used the canonical uniform n=11 plant, zero reset noise, a `+/-3 m` rail
soft limit of `2.85 m`, and a `5,000,000` rail penalty. This tested whether
the late high-rate approach could be made capturable without reopening the
whole swing search.

The `50`-iteration solve did not converge. Its cost was `17,883.96`, the
replay hit the rail at `3.0271 m` around `16.2 s`, and it produced no upright
streak or handoff. The last recorded state was still `2.0849 rad` from the
top with `23.42 rad/s` hinge-rate RMS and saturated action. The tail cannot
repair the inherited route's high-rate basin.

Decision: close late-tail iLQR for this source route. The next route search
must change the approach trajectory or optimize against a measured reachable
capture funnel from earlier in the swing; adding more terminal iterations to
this tail would only repeat rail saturation. No n=11 advancement or public
artifact update follows from this branch.

Artifact:

- `runs/generalized_solver/n11_rateheavy_tail_ilqr_12to16.json`

### Measured n=11 handoff target in full-route iLQR (2026-09-15)

The maintenance component already provides 24 real n=11 low-momentum upright
states that hold under exact target-plant LQR. To make the swing objective
explicitly downstream-aware, the first state in
`n11_p39_low_momentum_handoff_states.json` was supplied as the terminal
`qpos/qvel` target for a new exact hanging-start iLQR route. The run used the
uniform n=11 plant, zero reset noise, the inherited 16-second force route as
the warm start, a controllability terminal metric, a finite feedback-horizon
terminal metric, and the canonical rail penalty.

After `12` iLQR iterations the terminal state was still `2.9599 rad` from
upright, with `16.7868 rad/s` hinge-rate RMS and `14.1253 rad/s` absolute-rate
RMS. The best intermediate composite state occurred at `15.2 s` with
`1.1605 rad` angle, `4.3523 rad/s` hinge-rate RMS, and `4.1074 rad/s`
absolute-rate RMS. The run had no usable capture state and did not approach
the measured handoff target.

Decision: keep the saved maintenance states as the correct capture target and
close this particular full-route iLQR warm start. The result separates the
problem cleanly: the capture/stabilize expert has a measured basin, while the
current n=11 swing route cannot reach it. The next swing experiment must use a
different approach policy or a staged reachable-funnel objective; more weight
on this already-diverging terminal solve is not justified.

Artifacts:

- `runs/generalized_solver/n11_p39_low_momentum_handoff_states.json`
- `runs/generalized_solver/n11_handoff_target_ilqr.json`
- `runs/generalized_solver/n11_handoff_target_ilqr_handoff.json`

### Long-windup, wide-rail, and closed-loop modal tests (2026-09-15)

The next batch tested the physical hypothesis that an 11-link swing needs a
longer rail and a longer, slower wind-up. The corrected exact-MuJoCo CEM used
the canonical uniform n=11 plant, a +/-12 m discovery rail, a 32 s route, 96
action knots, 24 candidates, 30 iterations, and no handoff before 20 s. The
first invocation was correctly rejected before evidence collection because its
32 s route exceeded the 30 s episode cap. With the cap raised to 35 s, the
best candidate reached only `2.2409 rad` from upright at `21.36 s`, with
`1.1907 rad/s` hinge RMS, cart position `-6.3553 m`, and capture-constraint
violation `109.2861`. A longer wind-up and wider discovery rail alone did not
produce a reachable handoff.

A closed-loop energy/modal policy was also tested on the +/-12 m rail for 24 s
with 30 CEM iterations and 32 candidates. Its best angle was `1.1077 rad`,
but the run eventually reached `12.1736 m`, had no upright or capture event,
and ended in a rail violation. The low-dimensional policy was not a valid
replacement for the route-plus-capture experts.

Decision: keep the long-rail relationship as a real design variable, but do
not treat discovery-rail success or an angle crossing as evidence. Any future
long-rail route must replay on the canonical rail and satisfy the full
low-momentum handoff gate.

Artifacts:

- `runs/generalized_solver/n11_global_cem_long32_rail12.json`
- `runs/generalized_solver/n11_energy_modal_policy_rail12.json`

### Terminal-feedback transfer and capture-ready route refinement (2026-09-15)

To test whether the 10-link transfer failed only because the added distal mode
was left unactuated, the 10-link terminal absolute-angle feedback and terminal
relative-rate feedback columns were copied into the new n=11 distal mode. The
settled-launch evaluation was `0/5`: every episode ended in a rail violation,
with maximum cart excursion `3.0431 m`, required-rail ratio `1.0744`, and no
upright interval. Copying predecessor terminal feedback into the added mode
is closed as a transfer rule.

A separate exact-MuJoCo capture-ready CEM optimized the route for angle,
hinge rate, absolute rate, cart speed, and cart position on a +/-12 m
discovery rail. Its best composite transient was `0.2987 rad` with
`0.8913 rad/s` hinge RMS, `2.3218 rad/s` absolute-rate RMS, cart speed
`0.2909 m/s`, and position `1.9994 m`; it never held. The lowest-angle
transient reached `0.1666 rad` but still carried `1.5972 rad/s` hinge RMS and
`3.0048 rad/s` absolute-rate RMS. These are useful diagnostics, not capture
states.

Artifacts:

- `runs/generalized_solver/n11_n10_terminal_feedback_copy_warm.json`
- `runs/generalized_solver/n11_n10_terminal_feedback_copy_eval5.json`
- `runs/generalized_solver/n11_trajectory_capture_ready_refine_rail12.json`

### Expert-chain gate audit and local capture basin test (2026-09-15)

The best capture-ready transient was fed into the actual expert-chain search.
The chain reached an angle-only capture event at about `0.128 rad`, but the
handoff still had `1.9503 rad/s` hinge RMS, `1.5402 m/s` cart speed, and only
`0.02 s` of upright streak before the cart exited the +/-12 m discovery rail
at `12.0518 m`. This is a direct failure of an angle-only capture gate: visual
upright is not a stable handoff.

A local Box-FDDP capture solve was then initialized from the measured
trajectory state. The requested LQR scale `1.3` was rejected by the spectral
stability check (`rho=3.5276`). At the stable bank scale `1.0`, the solve was
feasible on paper but diverged numerically (`min_v=7.79e12`, terminal
`v=1.06e15`); exact replay exited at `3.022 m` with zero hold. The measured
state is outside the usable capture basin, so it must not be promoted as a
handoff example.

Decision: the next chain search must require low angle, low internal rate, low
cart velocity, and bounded cart position before it switches to capture. The
playbook's active expert contract is now explicit: the swing expert hands off
only a state that the stabilizer can actually retain; a brief top crossing is
not sufficient.

Artifacts:

- `runs/generalized_solver/n11_chain_capture_refine_from_low_streak.json`
- `runs/generalized_solver/n11_trajectory_capture_ready_refine_state.json`
- `runs/generalized_solver/n11_capture_from_trajectory_refine_state_fddp.json`

### Explicit low-momentum capture gate (2026-09-15)

The chain evaluator now supports an optional state gate that must be satisfied
before the swing expert can hand control to capture: the searched angle limit,
hinge-velocity RMS, absolute cart velocity, and absolute cart position. The
legacy angle-only behavior remains the default for old artifacts; the new
thresholds are recorded in each gated search result.

The first n=11 gated CEM used hinge RMS `<=0.90 rad/s`, cart speed `<=0.50
m/s`, and `|x|<=2.0 m` on a +/-12 m discovery rail. It found a visual crossing
at `0.0677 rad` with cart speed `0.2377 m/s`, but hinge RMS was `2.6078 rad/s`.
The gate therefore never switched to capture (`capture_reached=false`), even
though the cart stayed within `2.5994 m` of center and the run did not hit a
rail. This validates the gate as a diagnostic: it prevents an internally
violent crossing from being mislabeled as a handoff.

Decision: retain the gate in the expert-chain tool and use it for subsequent
searches. A threshold sweep may be useful for mapping the reachable basin, but
every candidate must still be replayed with the measured LQR retention test
and the canonical rail before it can advance the n=11 frontier.

Artifacts:

- `scripts/search_swingup_chain.py`
- `runs/generalized_solver/n11_chain_low_momentum_gate_rail12.json`

### Capture-gate threshold sweep (2026-09-15)

Relaxing only the hinge threshold to `2.5 rad/s` while retaining cart speed
`<=0.50 m/s` and `|x|<=2.0 m` still produced no handoff. The closest state
was `0.0786 rad`, `2.5916 rad/s`, and `0.5504 m/s` at `x=1.8698 m`, so it
missed both the hinge and cart-speed conditions by a small amount.

Replaying that exact controller with hinge `<=3.0 rad/s` and cart speed
`<=0.75 m/s` did activate capture, but it immediately exposed why the tighter
gate matters. The switch occurred at the state above; the best later capture
state had already degraded to `0.2431 rad`, `3.8084 rad/s`, and `0.8643 m/s`.
The LQR run exited at `12.0452 m`, with final hinge RMS `64.54 rad/s` and zero
upright hold. This is not a near miss and is not a valid capture state.

Decision: do not widen the capture gate to manufacture a transition. The
approach expert must reduce internal and cart momentum before the switch. The
threshold sweep is closed as a basin map and the strict gate remains the
working contract.

Artifacts:

- `runs/generalized_solver/n11_chain_gate_hinge25_rail12.json`
- `runs/generalized_solver/n11_chain_gate_hinge30_replay.json`

### Closed-loop receding tail after the near-top crossing (2026-09-15)

The measured near-top state was also handed to the exact-MuJoCo receding
horizon CEM tail controller. It replanned `43` times from the live state with
an 80-step horizon and 10-step application window on the +/-12 m discovery
rail. The run first reached upright at `11.10 s`, but held only `0.02 s` and
then exited at `x=12.0091 m`; final hinge RMS was `4.7711 rad/s` and the
maximum continuous hold was zero beyond the brief crossing.

Decision: feedback replanning alone does not arrest the n=11 internal modes
from this approach state. Preserve the receding controller as a diagnostic,
but move the next search upstream and force a slower late wind-up.

Artifact:

- `runs/generalized_solver/n11_gate25_receding_mpc_tail_rail12.json`

### Capture-arrest residual policy and maintenance-preservation test (2026-09-15)

The capture expert was trained as an MLX PPO residual around the analytic LQR
teacher, using a frozen bank of 24 measured quiet n=11 handoffs plus four
scaled copies of the measured high-rate crossing (`0.25x`, `0.50x`, `0.75x`,
and `1.00x` velocity). The first Torch invocation was invalid in this runtime
because PyTorch is unavailable; the equivalent MLX run completed 300 updates
and 614,400 environment steps.

The aggregate training evaluation showed `0/28` full successes. A deterministic
indexed replay of the best checkpoint made the failure more specific: all 28
states reached the upright neighborhood, but the learned residual destroyed
the known maintenance basin. On the 24 quiet states, the longest upright hold
was only `0.20--0.36 s` and the mean was `0.2843 s`, versus multi-second holds
from the analytic LQR bank. The four high-rate states still failed as well;
one exited the rail and the other three never held. The run therefore learned
an upright crossing signal, not a capture-arrest controller.

Decision: do not promote this checkpoint or use it as evidence of capture.
The next capture-policy experiment must stage the curriculum so the residual
first preserves the proven quiet-state maintenance behavior, then introduces
incoming rate in small increments. Every stage remains subject to the real
five-second hold and rail gates; a high `ever_upright_rate` is insufficient.

Artifacts:

- `runs/generalized_solver/n11_capture_arrest_high_rate_ppo/config.resolved.yaml`
- `runs/generalized_solver/n11_capture_arrest_high_rate_ppo/checkpoints/best.safetensors`
- `runs/generalized_solver/n11_capture_arrest_high_rate_ppo/checkpoints/best.meta.json`
- `runs/generalized_solver/n11_capture_arrest_high_rate_ppo/train_log.csv`

### Capture interface curriculum and failed local basins (2026-09-15)

The next experiments kept the same measured n=11 handoff states and changed
only the capture interface. The purpose was to separate three questions that
had been getting conflated: whether a policy can preserve the quiet LQR bank,
whether it can arrest incoming velocity, and whether the measured state is
locally reachable by a model-based force sequence.

The strict action-LQR switch PPO run used the four scaled high-rate states
(`0.25x`, `0.50x`, `0.75x`, `1.00x`) with tight switch thresholds. It completed
500 updates and 1,024,000 steps. Direct deterministic replay of every saved
checkpoint produced `0/4` successes; the maximum hold was only `0.02--0.08 s`,
with no low-momentum interval and no useful LQR takeover. The trainer's
inherited evaluation path also pointed at an older p=3e-5 bank, so the direct
high-rate replay is the authoritative result.

An exact-MuJoCo CEM capture actor initialized from the quarter-rate state held
for only `0.06 s` and never reached centered low-momentum capture. A cold
Box-FDDP force trajectory from the same state stopped after eight iterations
with a rail exit and terminal Lyapunov value `6.03e15`; its minimum value
reported during the solve was `5.77e12`. These are local-basin failures, not
evidence that a permissive capture threshold is acceptable.

Two curriculum variants made the required anchor explicit. Restoring the real
position with zero velocity was already outside the usable LQR basin, so the
velocity-only curriculum stayed at `p=0` and was stopped. Restoring position
and velocity together passed the centered p=0 anchor (`4/4`, `8.02 s` hold,
zero cart excursion), then failed at `p=0.1`; the inherited relative-coordinate
LQR gain became numerically ill-conditioned, switched for one step, saturated,
and exited the rail. A finite-difference absolute-angle Riccati diagnostic
still required impractically large gains and was not promoted as a fix.

Raw PPO with reset-state normalization and no analytic teacher also failed at
the exact upright anchor: it never exceeded roughly `0.3 s` deterministic hold
and could not advance the curriculum. This confirms that random-action PPO is
not a sufficient maintenance teacher for this high-dimensional upright.

The follow-up exact-teacher run kept the analytic switch active only at the
exact p=0 equilibrium (`1e-5` angle/rate thresholds) and handed every nonzero
state scale to raw PPO. It completed 600 updates and 1,228,800 steps. The p=0
teacher passed `4/4` with `8.02 s` hold and zero excursion; the first nonzero
band, p=0.025, never advanced and ended at `0/4` with `1.0` ever-upright rate,
`0.28 s` maximum upright hold, and `0.22 s` maximum low-momentum hold. The
high return during the p=0 anchor is therefore teacher validation, not a
capture result.

Decision: the capture expert still lacks a demonstrated arrest basin. Do not
claim n=11, do not promote any of these policies, and do not widen the gate to
turn a violent crossing into a handoff. The next useful lever is a smaller,
explicitly supervised maintenance-to-capture curriculum or a teacher that
controls the incoming modal energy while preserving the proven quiet bank.
The playbook contract remains: real measured state, reset-free handoff,
canonical rail, five-second hold, disjoint noisy replay, and saved artifacts.

Artifacts:

- `configs/swingup11_capture_switch_high_rate.yaml`
- `runs/generalized_solver/n11_capture_switch_high_rate_ppo/train_log.csv`
- `runs/generalized_solver/n11_capture_switch_high_rate_ppo/checkpoints/best.meta.json`
- `configs/swingup11_capture_real_handoff.yaml`
- `runs/generalized_solver/n11_capture_high_rate_cem_state0.json`
- `runs/generalized_solver/n11_capture_high_rate_fddp_state0.json`
- `configs/swingup11_capture_switch_velocity_curriculum.yaml`
- `runs/generalized_solver/n11_capture_switch_velocity_curriculum_ppo/train_log.csv`
- `configs/swingup11_capture_switch_full_state_curriculum.yaml`
- `runs/generalized_solver/n11_capture_switch_full_state_curriculum_ppo/train_log.csv`
- `configs/swingup11_raw_capture_full_state_curriculum.yaml`
- `runs/generalized_solver/n11_raw_capture_full_state_curriculum_ppo/train_log.csv`
- `configs/swingup11_capture_exact_teacher_curriculum.yaml`
- `runs/generalized_solver/n11_capture_exact_teacher_curriculum_ppo/checkpoints/best.meta.json`
- `runs/generalized_solver/n11_capture_exact_teacher_curriculum_ppo/train_log.csv`

### Rich local capture laws and explicit two-expert arrest (2026-09-15)

The measured quarter-rate high-momentum state was used to test two additional
state-feedback interfaces. A CEM over a static affine controller with the
complete absolute-plus-relative angle/rate feature vector completed 21
generations on a +/-12 m discovery rail. Its best deterministic hold was
`0.06 s`, with zero centered and zero low-momentum hold. A second CEM gave the
controller two independent affine experts, an explicit `0.50 s` arrest phase,
and a fixed stabilizer phase. After 25 generations it still held only
`0.06 s` and never produced a centered or low-momentum interval.

Decision: a richer static feature map and a hard two-expert time split do not
solve the incoming modal energy. Keep the two-expert idea in the architecture
roadmap, but the switch must be reached by a trajectory that has already
cooled the internal modes; local capture policy search alone is not enough.

Artifacts:

- `runs/generalized_solver/n11_capture_affine_full_state_cem_state0_rail12.json`
- `runs/generalized_solver/n11_capture_two_phase_affine_state0_rail12.json`

### Long-rail slow-windup and late handoff-bank residual tests (2026-09-15)

The long-rail hypothesis was tested directly on the canonical uniform n=11
plant with a +/-20 m discovery rail and a 20--32 second cart-position
wind-up. The first exact-MuJoCo CEM reached late candidates around
`0.66--0.96 rad` from upright with hinge-rate RMS around `1.33--1.50
rad/s`; it never produced an upright event. A lower-variance refinement kept
the best angle near `0.633 rad` and reduced hinge-rate RMS only to about
`1.255 rad/s`. Both searches completed their intended 36 second horizons and
were rejected because no capture or hold occurred.

A second search kept the long wind-up as the center waveform and optimized a
late residual against the measured quiet handoff bank. It ran 24 exact CEM
iterations with residual action beginning at 14 seconds and the handoff
window beginning at 20 seconds. The best terminal distance remained `3.846`
in the handoff metric, the best state distance was `4.261`, and the cart
reached `13.611 m` on the +/-20 m discovery rail. The run therefore did not
find a capture-ready state; matching a bank objective without a feasible
terminal state is not evidence of a handoff.

Decision: retain the longer rail and slow wind-up as legitimate discovery
settings, but do not treat them as a solution or silently transfer their
metrics to the canonical rail. The next route must improve the full terminal
state together: absolute angle, internal hinge rates, absolute rates, cart
velocity, and cart position. A visually closer top crossing remains a failed
approach until the actual stabilizer can retain it.

Artifacts:

- `runs/generalized_solver/n11_long_rail_slow_windup_cem_rail20.json`
- `runs/generalized_solver/n11_long_rail_slow_windup_refine_cem_rail20.json`
- `runs/generalized_solver/n11_long_windup_handoff_bank_residual_rail20.json`

### Playbook operating rule for the 11-link frontier (2026-09-15)

`docs/levers_and_pitfalls.md` is the living experiment ledger and the
project's working playbook. Every new route or controller branch must add a
dated entry with: the physical/controller lever changed, the exact command or
artifact, the rail and episode conditions, the strongest quantitative result,
the failure reason when applicable, and the decision that follows. Discovery
results on widened rails, altered morphology, or curriculum plants stay
explicitly labeled as development evidence. Only a reset-free replay on the
canonical uniform plant, canonical rail, required hold, and disjoint noisy
gates can advance the link-count claim in `README.md`, `ROADMAP.md`, or the
GitHub About text.

This rule is intentionally part of the playbook rather than a side note: it
prevents a promising trajectory, a near-top crossing, or a curriculum
checkpoint from being promoted as a solved link count before the capture and
maintenance experts have both been demonstrated.

### Widened-rail replay of the near-locked split-link boundary (2026-09-15)

The failed upper-bound continuation trial at split-link progress
`p=0.000020996` was replayed with its exact saved feedback controller on a
diagnostic +/-20 m rail. This separates rail clearance from the morphology
continuation itself. The controller did not reach its capture phase: the cart
ran to `20.0798 m`, the minimum whole-chain angle was still about `0.767 rad`,
and the replay had zero upright time and zero hold. The canonical replay of
the same trial had already failed at `3.0386 m`, so the wider rail changes the
termination point but not the underlying capture failure.

One preliminary replay command attempted to rewrite the rail with a BSD
`sed` range expression that did not match; its reported 3 m limit was retained
as a setup check and is not evidence for this experiment. The corrected
artifact records the actual 20 m plant. This is a useful tooling pitfall:
every widened-rail diagnostic must report and verify the instantiated
`rail_limit` before its metrics enter the ledger.

Decision: a near-locked split-link route cannot be promoted by widening the
rail. The continuation must produce a better terminal trajectory and a valid
capture basin as the split is unlocked; otherwise it is only a morphology
escape hatch, not an 11-link solution.

Artifacts:

- `runs/generalized_solver/n11_forced_split_link9_p0.000020996_widerail20_replay_v2.json`
- `runs/generalized_solver/n11_forced_split_link9_p0.000020996_widerail20_replay.json` (setup-check artifact; reported rail remained 3 m)

### Distal split-position release probe (2026-09-15)

The alternate split at source link `10` was tested with the same 10-to-11
locked-start teacher, exact MuJoCo replay, waypoint repair, and target-plant
Box-FDDP contract used for the link-9 branch. The locked distal split is a
valid development baseline: `22.12 s` hold and `2.023 m` maximum cart
excursion. The first two unlock brackets were then completed:

| Release progress | Strongest completed result | Decision |
| --- | --- | --- |
| `p=0.001` | Waypoint endpoint at `2.023 m`, but target-plant refinement exited at `3.066 m` with `0 s` hold | Rejected |
| `p=0.0005` | Waypoint endpoint at `2.024 m`, but target-plant refinement exited at `3.034 m` with `0 s` hold | Rejected |

The repaired endpoints were physically finite and close to the incumbent,
which is useful evidence that waypoint construction itself is not the only
obstacle. Every exact target-plant refinement became unstable or left the
canonical rail before a sustained capture. A third `p=0.00025` waypoint
search was interrupted during its longest lookahead; because it did not
complete its refinement/replay gate, it is explicitly not counted as a
result.

Decision: the distal split does not materially widen the release basin. Keep
the link-10 variant as a negative split-position control, and return compute
to a controller/interface change that addresses full-chain modal energy and
capture, rather than testing more split locations with the same fragile local
repair.

Artifacts:

- `runs/generalized_solver/n11_forced_split_link10.yaml`
- `runs/generalized_solver/n11_forced_split_link10_warm.json`
- `runs/generalized_solver/n11_forced_split_link10_homotopy/continuation.json`
- `runs/generalized_solver/n11_forced_split_link10_homotopy/trials/trial_0001_p0.001000000_pass2.json`
- `runs/generalized_solver/n11_forced_split_link10_homotopy/trials/trial_0002_p0.000500000_pass2.json`

### Free-added-mode support sweep (2026-09-15)

The next support hypothesis was tested on the canonical uniform n=11 plant:
start with the added mode physically unlocked, but temporarily give that mode
small damping and stiffness while replaying the best locked-route teacher. This
keeps the added joint free at progress zero; it is not the invalid shortcut of
locking the mode and calling the result an 11-link solve. The teacher was the
exact saved n=11 locked-route controller, replayed for 8 seconds with a
diagnostic +/-6 m rail.

The baseline free-mode replay reached `6.028 m` and produced `0 s` of hold.
The support sweep also produced `0 s` of hold for all four tested settings:

| Temporary added-mode support | Maximum cart excursion | Hold | Decision |
| --- | ---: | ---: | --- |
| stiffness `3`, damping `0.03` | `6.043 m` | `0 s` | Rejected |
| stiffness `10`, damping `0.03` | `6.016 m` | `0 s` | Rejected |
| stiffness `30`, damping `0.03` | `6.039 m` | `0 s` | Rejected |
| stiffness `10`, damping `0.10` | `6.076 m` | `0 s` | Rejected |

These are diagnostic replays, not canonical evidence. The result closes the
simple “copy the locked route and let the new mode become free gradually” path:
temporary local support does not preserve the teacher's capture basin. A
successful curriculum must re-optimize the route while the new mode is free,
and must replay the final route with every temporary support removed.

Artifacts:

- `tmp/n11_free_ghost_support/config.yaml`
- `tmp/n11_free_ghost_support/p0_replay.json`
- `tmp/n11_free_ghost_support/k3d03/replay.json`
- `tmp/n11_free_ghost_support/k10d03/replay.json`
- `tmp/n11_free_ghost_support/k30d03/replay.json`
- `tmp/n11_free_ghost_support/k10d10/replay.json`

### Online literature triage for the next route (2026-09-15)

The next controller interface is informed by three primary references, but
none of them is evidence for this project. The MIT Underactuated Robotics
cart-pole notes describe energy shaping followed by local LQR capture. Xin,
She, and Yamasaki describe virtual composite-link coordinates for an n-link
planar robot with one passive joint. Shkolnik et al. describe searching in a
low-dimensional task space while simulating the complete joint dynamics, with
an objective that combines energy error and weighted joint-space distance.
Xin et al. also show why rail length belongs in the controller objective by
using a coupling-energy/barrier formulation for a cart-pole with limited-track
length.

Use these as design prompts for the 11-link experiments: score the full
absolute-angle shape, internal rates, cart state, energy, and rail barrier
together; keep a local upright stabilizer; and validate every candidate on the
full plant. Do not copy their results into the project claim, since their
models, link counts, and evaluation contracts differ.

References:

- [MIT Underactuated Robotics: Acrobot and cart-pole control](https://underactuated.csail.mit.edu/acrobot.html)
- [Xin, She, and Yamasaki, Swing-up Control for n-Link Planar Robot with Single Passive Joint](https://www.jstage.jst.go.jp/article/sicetr/45/5/45_5_251/_article/-char/en)
- [Shkolnik et al., High-Dimensional Underactuated Motion Planning](https://groups.csail.mit.edu/robotics-center/public_papers/Shkolnik08.pdf)
- [Xin et al., Coupling-energy-based swing-up control for a pendulum-cart system with limited-track length](https://journals.sagepub.com/doi/10.1177/10775463251405631)

### Hanging-start free-mode FDDP continuation (2026-09-15)

The p=0 ghost continuation was re-optimized from the actual hanging state,
not from a previously captured upright state. The starting state was the
canonical eleven-link hanging configuration with zero velocity. The teacher
was the saved locked-route controller that holds the supported p=0 plant for
`22.12 s`; the continuation then optimized on a diagnostic `+/-6 m` rail with
the added mode physically free. This is the relevant test for whether the
locked route can be made into a genuine hanging-start route by a small release
and local FDDP repair.

| Release progress | Initial-feedback treatment | FDDP result | Live replay | Decision |
| --- | --- | --- | --- | --- |
| `p=0.001` | Saved locked-route feedback | `23` iterations; terminal value about `9722` | `6.002 m` cart excursion, `0.180 s` hold | Rejected |
| `p=0.01` | Saved locked-route feedback | `14` iterations; terminal value about `0.01` | `6.046 m` cart excursion, `0 s` hold | Rejected |
| `p=0.1` | Saved locked-route feedback | `14` iterations; terminal value about `1.52` | `5.727 m` cart excursion, `0 s` hold | Rejected |
| `p=0.001` | Rebuilt initial feedback, scale `.25` | `21` iterations; terminal value about `3.05e6` | `6.076 m` cart excursion, `0 s` hold | Rejected |
| `p=0.001` | Rebuilt initial feedback, scale `.50` | `17` iterations; terminal value about `3.63e6` | `6.015 m` cart excursion, `0 s` hold | Rejected |
| `p=0.001` | Rebuilt initial feedback, scale `.75` | `40` iterations; terminal value about `1293` | `6.032 m` cart excursion, `0.180 s` hold | Rejected |

Several runs emitted MuJoCo instability warnings. The exact saved route and
the feedback-rebuild variants all left the diagnostic rail before producing a
repeatable capture basin. This closes the simple warm-start explanation: the
failure is not just that the old feedback matrix was stale. A successful
continuation needs a new trajectory/interface or a more global search, not
another minor p-step or feedback-scale sweep around this seed.

Decision: keep the ghost route as a development boundary and do not promote
any of these runs to an 11-link result. The canonical uniform hanging-start
plant, canonical rail, reset-free replay, and held-out gates remain unmet.

Artifacts:

- `runs/generalized_solver/n11_ghost_hanging_route_p0001_fddp.json`
- `runs/generalized_solver/n11_ghost_hanging_route_p001_fddp.json`
- `runs/generalized_solver/n11_ghost_hanging_route_p01_fddp.json`
- `runs/generalized_solver/n11_ghost_hanging_route_p0001_rebuildfb025_fddp.json`
- `runs/generalized_solver/n11_ghost_hanging_route_p0001_rebuildfb050_fddp.json`
- `runs/generalized_solver/n11_ghost_hanging_route_p0001_rebuildfb075_fddp.json`

### Modal-coherence and VCL tail-composite probe (2026-09-15)

Two state-shape corrections were screened on the canonical uniform eleven-link
plant using the generalized energy controller. The first used absolute-angle
and rate coherence. The second used virtual composite-link (VCL) angles and
rates: each VCL is the center-of-mass direction of a distal tail. The VCL
implementation is a representation and feedback probe inspired by the
literature cited above; the cited robot and actuation assumptions are not the
cart-pole evidence contract.

The coherence branch did not produce an upright switch in any tested setting.
Ungated gains either consumed the diagnostic rail (about `3.00 m`) or reduced
the excursion without reaching capture. Energy-proximity gates softened the
rail behavior, but still produced no upright hold. The strongest gated
coherence replay reached about `2.375 m` without a capture event.

The VCL branch was similarly non-solving. The mildest useful setting reduced
the rail ratio below one and reached a minimum VCL angle RMS of about `0.294`
rad, but produced no upright switch or hold. Larger gains consumed the rail;
the strongest rail-safe setting still produced no capture.

| Branch / setting | Maximum cart or rail ratio | Upright/hold result | Decision |
| --- | ---: | --- | --- |
| Coherence, gated `g=.08`, `p=.02`, `v=.02`, `w=1` | `2.375 m` | No switch, no hold | Rejected |
| VCL, gated `g=.08`, `p=.25`, `v=.5`, `w=.25`, limit `1` | rail ratio `.890` | No switch, no hold; min VCL RMS `.294 rad` | Rejected |
| VCL, gated `g=.08`, `p=.2`, `v=.2`, `w=1`, limit `.5` | rail ratio `.892` | No switch, no hold; min VCL RMS `1.15 rad` | Rejected |
| VCL, higher-gain settings | rail ratio `1.06-1.07` | Rail violation, no hold | Rejected |

Decision: VCL coordinates are useful diagnostics for tail shape, but a late
least-squares coherence correction is not the missing swing-up mechanism. Keep
the code available as a measurement/control primitive, but do not add it to
the active claim route until it is paired with a genuine energy-pumping and
capture strategy that survives the canonical rail and held-out gates.

Artifacts:

- `runs/generalized_solver/n11_coherence_base.json`
- `runs/generalized_solver/n11_coherence_p025v05w025l1.json`
- `runs/generalized_solver/n11_coherence_p05v1w05l1.json`
- `runs/generalized_solver/n11_coherence_p1v2w05l2.json`
- `runs/generalized_solver/n11_coherence_p02v02w1l05.json`
- `runs/generalized_solver/n11_coherence_g08_p025v05w025.json`
- `runs/generalized_solver/n11_coherence_g06_p05v1w05.json`
- `runs/generalized_solver/n11_coherence_g08_p05v1w05.json`
- `runs/generalized_solver/n11_coherence_g10_p1v2w05.json`
- `runs/generalized_solver/n11_coherence_g08_p02v02w1.json`
- `runs/generalized_solver/n11_vcl_g08_p025v05w025.json`
- `runs/generalized_solver/n11_vcl_g08_p05v1w05.json`
- `runs/generalized_solver/n11_vcl_g06_p05v1w05.json`
- `runs/generalized_solver/n11_vcl_g10_p1v2w05.json`
- `runs/generalized_solver/n11_vcl_g08_p02v02w1.json`

### VCL phase/rate energy-pump screen (2026-09-15)

The VCL probe was extended from a late coherence correction to an opt-in
swing-phase signal. The controller added signed weighted VCL phase, natural-
time-scaled VCL phase rate, and energy-error times VCL-rate terms to the same
exact mass-matrix energy pump. The upright LQR handoff and canonical
`+/-3 m` rail were unchanged. This directly tests whether the VCL coordinates
can supply the missing swing phase rather than merely damp the tail near the
top.

Eight one-episode canonical screens all failed the swing-up/hold gate. The
most rail-efficient candidate (`vcl_pump_gain=-2`) stayed at `0.647` required
rail ratio, but ended with about `2.83 rad` maximum final angle and no upright
event. The positive pump candidates either left the rail (`1.06-1.08` ratio)
or remained far from the top. Phase/rate combinations changed the rail use
but did not produce capture; the best rail-safe combination still had about
`2.68 rad` final angle and zero hold.

Decision: VCL tail phase/rate is not, by itself, the missing energy-pumping
interface for the uniform 11-link plant. Keep the feature and diagnostics
available for a future trajectory-conditioned controller, but stop tuning this
low-dimensional additive screen. The active route returns to a measured
handoff-reaching swing expert, with the capture bank and playbook gates kept
separate.

Artifacts:

- `runs/generalized_solver/n11_vclpump_p2.json`
- `runs/generalized_solver/n11_vclpump_m2.json`
- `runs/generalized_solver/n11_vclpump_p5.json`
- `runs/generalized_solver/n11_vclpump_m5.json`
- `runs/generalized_solver/n11_vclphase_p2_rate1.json`
- `runs/generalized_solver/n11_vclphase_m2_ratem1.json`
- `runs/generalized_solver/n11_vclmix_p2_phm1_ratem05.json`
- `runs/generalized_solver/n11_vclmix_m2_php1_rate05.json`

### Receding-horizon swing planner and measured LQR-tail audit (2026-09-15)

The next two-expert experiment used a live exact-MuJoCo receding-horizon CEM
as the swing expert. It replanned from the state emitted by the previous
action, rather than replaying an open-loop force trace. A canonical-rail probe
reached only `0.522` potential fraction before a `3.035 m` excursion. On a
diagnostic `+/-12 m` rail, the same planner reached `0.973` potential fraction
and a `0.699 rad` best angle, but produced no upright hold and emitted
instability warnings.

A quieter-action/capture-weight screen on the wide rail reached `0.996`
potential fraction and a `0.254 rad` angle crossing at `6.561 m` maximum cart
excursion. The state was not capture-ready: hinge RMS was `8.27 rad/s`,
absolute-rate RMS was `15.57 rad/s`, and cart speed was `10.77 m/s`. The exact
upright LQR was then replayed from the twelve best recorded near-top states;
all `12/12` failed, with `0 s` hold. The best-angle state eventually reached
the wide rail boundary and ended at about `2.97 rad`.

| Planner interface | Rail / settings | Strongest result | Decision |
| --- | --- | --- | --- |
| Height-scored receding CEM | Canonical `+/-3 m` | Height `.522`, angle `1.986 rad`, `3.035 m` excursion | Rejected |
| Height-scored receding CEM | Diagnostic `+/-12 m` | Height `.973`, angle `.699 rad`, `12.050 m` excursion | Development only |
| Quiet-action capture-weight CEM | Diagnostic `+/-12 m` | Height `.996`, angle `.254 rad`, `6.561 m` excursion, zero hold | Rejected as handoff |
| Exact LQR from twelve near-top states | Diagnostic `+/-12 m`, `8 s` | `0/12` success, `0 s` hold | Rejected |

The planner was then upgraded so every candidate can be scored by simulating
the actual upright LQR for a short terminal tail. The first implementation
incorrectly applied that tail cost while candidates were still near the
hanging state; the gate was corrected to activate it only above the predicted
capture-height threshold. The corrected tail-scored runs reached heights
`.787` and `.895` with best angles `1.598` and `1.298 rad`, maximum excursions
`9.048 m` and `8.150 m`, and zero hold. A lighter tail penalty recovered more
height but still did not enter a capture basin.

Decision: this closes the current receding-CEM interface as an 11-link
solution path. It establishes a useful quantitative target for the next
swing expert: reach the measured capture manifold with low cart velocity and
low absolute rates, not merely high potential or a visually vertical chain.
The planner and LQR-tail scorer remain reusable diagnostics, but all wide-rail
and near-top results stay explicitly outside the public link-count claim.

Artifacts:

- `scripts/search_swingup_online_mpc.py`
- `runs/generalized_solver/n11_online_mpc_swing_canonical_probe.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_probe.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_quietcapture.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_strictcapture.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_captureweight12.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_captureweight30.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_capture_candidates.json`
- `runs/generalized_solver/n11_online_mpc_swing_rail12_capture_lqr_screen.json`
- `runs/generalized_solver/n11_online_mpc_swing_lqr_tail_rail12.json`
- `runs/generalized_solver/n11_online_mpc_swing_lqr_tail_rail12_gated.json`
- `runs/generalized_solver/n11_online_mpc_swing_lqr_tail_rail12_light.json`

### Nonlinear capture tail from the best wide-rail crossing (2026-09-15)

The best recorded wide-rail crossing (`0.254 rad` maximum angle) was handed
to the exact-MuJoCo nonlinear receding-horizon capture search. This removes
the possibility that the failure was caused only by using a linear LQR tail.
The capture planner ran for `4 s` with `50` replans on the `+/-12 m` diagnostic
rail. It produced no upright or low-momentum streak; the final state was
about `1.433 rad` maximum angle, `4.77 rad/s` hinge RMS, `5.996 rad/s`
absolute-rate RMS, and `7.264 m` cart position.

Decision: the near-top crossing is not a usable capture handoff for either
the linear or nonlinear capture expert tested here. The next swing search
must target the measured quiet handoff manifold directly, including cart
position/velocity and internal rates, instead of treating angle or potential
height as sufficient.

Artifact:

- `runs/generalized_solver/n11_online_capture_from_mpc_top_state_rail12.json`

### MPC residual and whole-route FDDP warm-start probes (2026-09-15)

The recorded quiet-action MPC route was tested as a route-level initialization
for the handoff-bank CEM and for Box-FDDP. The residual CEM was constrained to
the route after the nominal handoff window and used the diagnostic `+/-12 m`
rail. Its best bank-distance score was about `7.22`, terminal distance about
`13.59`, and maximum cart position about `7.28 m`; it produced no capture or
hold. This is a negative result for local correction around the MPC route, not
evidence against the measured capture bank itself.

The same route was serialized as a 500-action, 10-second warm start with
nominal states. The first whole-route FDDP continuation used the measured
handoff target, rebuilt initial states from the route, and kept the diagnostic
rail. It diverged almost immediately: `7` iterations, `converged=false`,
minimum value about `7.67e6`, terminal value about `1.30e18`, maximum cart
position `12.03 m`, and zero hold. The initial route feedback gains were zero,
so this is specifically a failed open-loop route warm start rather than a
proper feedback-route test.

Decision: reject the current residual and zero-gain route initialization. The
next route-level test must construct feedback gains along the recorded path
before asking FDDP to refine it. Keep the rail and canonical replay gates
separate from these diagnostic probes.

Artifacts:

- `runs/generalized_solver/n11_online_mpc_quietcapture_center.json`
- `runs/generalized_solver/n11_online_mpc_quietcapture_handoff_residual_rail12.json`
- `runs/generalized_solver/n11_hanging_state_exact.json`
- `runs/generalized_solver/n11_online_mpc_quietcapture_feedback_warm.json`
- `runs/generalized_solver/n11_online_mpc_quietcapture_fddp_terminal_target_rail12.json`

### LTV route-feedback warm start and rail-rescue probe (2026-09-15)

The recorded MPC trajectory was converted into a reusable finite-horizon
linear-time-varying feedback route. Exact MuJoCo finite differences and a
backward Riccati sweep were used to compute a gain at every route step. The
late MPC path is too far outside a local tracking neighborhood for this to be
well-conditioned: the default route reached a maximum feedback-gain norm of
about `4.75e4` and a local closed-loop spectral radius of about `214`. A
`0.05`-scaled exact replay on the diagnostic `+/-12 m` rail still became
unstable at the start and exited at `12.01 m`; no hold occurred. This closes
the route-feedback construction as a rescue for the existing MPC path.

The near-release rigid-split route was then replayed at the next release
trial with diagnostic rail half-lengths `3.30 m` and `3.50 m`. The first
replay reached `3.376 m` and the second `3.578 m`, with zero upright hold in
both cases. The excursion increased with the allowance instead of converging
to a bounded swing, so this is boundary chasing rather than evidence that a
longer rail solved the release. The canonical `3.0 m` rail and uniform free
plant remain unchanged.

Decision: reject both the current LTV warm start and simple rail rescue. A
future longer-rail experiment must include a new controller objective that
reduces the required rail and must return through rail contraction; replaying
the same route on a larger rail is not sufficient.

Artifacts:

- `scripts/build_route_ltv_feedback.py`
- `runs/generalized_solver/n11_online_mpc_quietcapture_ltv_feedback_default.json`
- `runs/generalized_solver/n11_online_mpc_quietcapture_ltv_r1000_t01.json`
- `runs/generalized_solver/n11_online_mpc_quietcapture_ltv_replay_g005_rail12.json`
- `runs/generalized_solver/n11_forced_split_link9_p000020996_rail330_replay.json`
- `runs/generalized_solver/n11_forced_split_link9_p000020996_rail350_replay.json`

### Measured quiet-target and transferred 10-link-route replay (2026-09-15)

The terminal target was replaced with an actual quiet upright state measured
from the assisted 11-link handoff bank, then the current dimension-lifted
10-link route was used as the whole-route warm start. The measured bank state
was from source row `1495` at `t=29.92 s`: angle about `1.9e-6 rad`, hinge-rate
RMS about `4.4e-7 rad/s`, cart position about `-0.00054 m`, and cart velocity
about `7.3e-5 m/s`. Whole-route FDDP still diverged after `7` iterations:
minimum value about `7.67e6`, terminal value about `2.78e15`, replay maximum
cart position `3.094 m`, and zero hold.

The transferred route was also replayed directly on a diagnostic `+/-12 m`
rail. The open-loop replay reached `|x|=12.031 m`; the feedback replay reached
`|x|=12.077 m`. Both had zero hold. These wide-rail replays show that the
failure is not just the canonical `+/-3 m` boundary.

Decision: the measured quiet target and settled launch contract are not the
missing ingredient until a free-chain swing trajectory can reach them. Reject
the transferred 10-link route as an 11-link warm start, keep it as a negative
control, and do not advance the 11-link status, README, GitHub About, or public
evidence. The next route search must use a materially different
parameterization with an explicit low-rate capture objective.

Artifacts:

- `runs/generalized_solver/n11_measured_quiet_terminal_target.json`
- `runs/generalized_solver/n11_n10_transfer_targeted_quiet_fddp.json`
- `runs/generalized_solver/n11_n10_transfer_open_replay_rail12.json`
- `runs/generalized_solver/n11_n10_transfer_feedback_replay_rail12.json`

### Mode chirp, retiming, aggregate-tail transfer, and longer MPC screens (2026-09-15)

The measured hanging normal modes were used to screen a compact frequency
chirp before adding another learned policy. The eleven mode frequencies were
approximately `0.346`, `0.794`, `1.211`, `1.512`, `1.896`, `2.329`, `2.757`,
`3.186`, `3.616`, `4.054`, and `4.498 Hz`. Eight exact serial replays varied
the chirp amplitude (`0.02` or `0.05`) and end frequency (`2.5` or `4.5 Hz`)
from the exact hanging state on a diagnostic `+/-12 m` rail. The best case
reached only `2.876 rad` from upright, with zero upright streak; all cases
remained in the hanging basin. This closes a blind measured-mode chirp as a
standalone swing expert.

The transferred ten-link route was then retimed in the opposite direction
from the earlier slow-windup tests. A `0.50x` route used `1.80 m` of cart
excursion during its 4-second materialization, while a `0.75x` route used
`3.94 m` during its 6-second materialization. On the matched `+/-20 m`
diagnostic replay both ran to the rail (`20.17 m` and `20.02 m`) with zero
upright streak; the faster timing did not reveal a capture basin.

A structural transfer was also tested. The last two target links were
aggregated into a physically matched ten-link surrogate with the final body
length `0.5454545 m` and mass `0.1818182 kg`, preserving the target chain's
total length and link mass. The ten-link route transferred into this surrogate
with a bounded `2.06 m` open replay, but its aligned 8-second replay never
got closer than `2.03 rad` to upright. Splitting that aggregate body back into
two locked target links reached `3.052 m` on the canonical rail with zero
hold. Targeted FDDP toward the measured quiet handoff stopped after `7`
iterations and reached `12.064 m` on the diagnostic rail; a force-limit
training-wheel run at `120 N` stopped after `9` iterations, emitted MuJoCo
instability warnings, and reached `12.020 m`. The aggregate-tail route is a
better-conditioned transfer seed than the ordinary lift, but it is not a
swing-up route.

Finally, a longer-horizon receding CEM was scored by simulating the exact
upright LQR tail from every predicted terminal state. With a `3.2 s` planning
horizon, `80`-step tail, and `10 s` replay on the `+/-12 m` rail, the best
potential fraction was `0.626`, the best angle was `1.692 rad`, maximum cart
excursion was `7.775 m`, and upright streak was `0 s`. This does not improve
the existing wide-rail MPC boundary.

Decision: close measured-mode chirps, faster global retiming, aggregate-tail
transfer, force-authority FDDP, and the longer tail-scored MPC as current
11-link solution paths. The active requirement is unchanged: a free uniform
11-link hanging-start route must reach the measured quiet handoff manifold on
the canonical rail before the capture expert can be credited. Keep the
aggregate-tail config as a reusable control experiment, but do not promote
any of its diagnostic routes, wide-rail runs, or surrogate holds.

Artifacts:

- `runs/generalized_solver/n11_mode_chirp_screen.json`
- `configs/swingup10_aggregate_tail.yaml`
- `configs/swingup11_aggregate_split_continuation.yaml`
- `runs/generalized_solver/n10_aggregate_tail_transfer_warm.json`
- `runs/generalized_solver/n10_aggregate_tail_fddp.json`
- `runs/generalized_solver/n10_aggregate_tail_fddp_aligned.json`
- `runs/generalized_solver/n10_aggregate_tail_replay_aligned8.json`
- `runs/generalized_solver/n11_aggregate_tail_transfer_warm.json`
- `runs/generalized_solver/n11_aggregate_tail_exact_replay_rail20.json`
- `runs/generalized_solver/n11_aggregate_tail_replay_rail20.json`
- `runs/generalized_solver/n11_aggregate_tail_target_fddp.json`
- `runs/generalized_solver/n11_aggregate_split_warm.json`
- `runs/generalized_solver/n11_aggregate_split_replay_p0.json`
- `runs/generalized_solver/n11_aggregate_tail_force120_fddp.json`
- `runs/generalized_solver/n11_n10_transfer_retime050_open_wide_warm.json`
- `runs/generalized_solver/n11_n10_transfer_retime050_open_replay_rail20.json`
- `runs/generalized_solver/n11_n10_transfer_retime075_open_wide_warm.json`
- `runs/generalized_solver/n11_n10_transfer_retime075_open_replay_rail20.json`
- `runs/generalized_solver/n11_n10_transfer_multiple_shooting8s.json`
- `runs/generalized_solver/n11_time_reversed_downfall_rail12.json`
- `runs/generalized_solver/n11_online_mpc_longtail_screen.json`

### Route-tracking repair and free-split stiffness boundary (2026-09-15)

The route-tracking MPC evaluator was corrected to extract the physical
MuJoCo state with `mj_getState`; the full MuJoCo state vector includes time,
so treating it as `[qpos, qvel]` had been corrupting the receding-horizon
rollout. The corrected exact-physics MPC still did not produce a legal
handoff. On a diagnostic `+/-12 m` rail, the best ordinary plan reached a
potential fraction of about `0.998`, but its best angle was only `0.496 rad`
from upright, its cart excursion was `12.160 m`, and its handoff speed was
about `6.62 m/s`. Tail-aware variants also escaped to the rail without an
upright hold. The state-extraction bug is fixed, but the transferred route is
still not a capturable 11-link route.

A whole-route correction CEM and a phase-dependent distal-feedback residual
CEM were then tested around the transferred route. The whole-route search
found a best route peak of `5.562 m` and no hold. The feedback-residual
search reached a best potential of `0.762`, angle `2.519 rad`, and cart
excursion `1.960 m`, again with zero hold. Both searches remain negative
controls: local corrections around the ten-link transfer do not recover a
legal free-chain capture.

The next curriculum screen inserted a temporary stiffness and damping at the
split joint between links 9 and 10 while leaving the split physically free
(`rigid_split_inertia=false`, joint locks zero). This is explicitly a
training wheel, not the canonical plant. The first embedded solver summaries
looked encouraging at progress zero on a diagnostic `+/-6 m` rail:
`K=100, c=2` reported `22.02 s` and `K=50, c=1` reported `29.94 s`. Those
were false positives. The trajectories contained MuJoCo instability warnings
followed by zeroed state rows, which the old evaluator counted as upright.
After the integrity guard was added, independent replays reported
`trajectory_integrity=false` and `termination_reason=numerical_instability`
for both; the `K=50` trace reached a nonphysical `48,901 m` cart excursion,
while the `K=100` trace retained a spurious `22.02 s` streak after warnings.

The lower-support replays were also negative: `K=40, c=.8` produced only
`0.12 s` before a `6.784 m` rail violation, `K=35, c=.7` produced `0.12 s`
before `6.441 m`, and `K=30, c=.6` and `K=25, c=.5` produced no upright
streak before violating at `6.017 m` and `6.065 m`. The boundary is even
sharper above `K=50`: `K=49.5`, `48.5`, and `48` produced instability warnings
and enormous nonphysical cart values; `K=45`, `42.5`, and `41` failed within
`0.08 s` or less. A continuation optimizer run at `K=50` simply inherited
the unstable route with no iterations, and a `K=20` continuation became
unstable and failed. This branch is therefore a numerical artifact, not a
support-assisted hold or a useful path to the uniform free 11-link plant.

The evaluator repair is part of the playbook: `search_ilqr_capture.py` now
rejects nonfinite or absurd state values and early zero-collapse traces, and
the regression tests cover both rejection and a genuine upright tail.

Decision: keep the corrected state extraction and the support curriculum as
diagnostics, but reject the transferred MPC, whole-route correction,
feedback-residual, and temporary-stiffness branches as release paths. Do not
advance the 11-link status, README, GitHub About, video, or paper. The next
experiment must remove the artificial support and change the route/capture
parameterization materially, while using this section as a boundary check.

Artifacts:

- `scripts/search_route_tracking_mpc.py`
- `scripts/search_route_feedback_residual_cem.py`
- `scripts/build_free_split_stiffness_continuation.py`
- `scripts/search_ilqr_capture.py`
- `tests/test_ilqr.py`
- `runs/generalized_solver/n11_route_tracking_mpc_rail12.json`
- `runs/generalized_solver/n11_route_tracking_mpc_tail_rail12.json`
- `runs/generalized_solver/n11_route_correction_lqr_capture_search.json`
- `runs/generalized_solver/n11_route_feedback_residual_cem_rail12.json`
- `runs/generalized_solver/n11_free_split_stiffness_k100_d2_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_d1_fddp.json`
- `runs/generalized_solver/n11_free_split_stiffness_k100_d2_replay_integrity_checked.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_d1_replay_integrity_checked.json`
- `runs/generalized_solver/n11_free_split_stiffness_k20_d04_fddp.json`
- `runs/generalized_solver/n11_free_split_stiffness_k40_d08_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k35_d07_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k30_d06_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k25_d05_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k49p5_d099_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k48p5_d097_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k48_d096_replay.json`

### Full-waveform CEM, physics shaper, and support-gradient follow-up (2026-09-15)

The corrected exact serial CEM was widened from phase-local feedback to a
48-knot additive force waveform over the entire transferred route. On a
diagnostic `+/-12 m` rail, `50` iterations improved the best objective to
`406177`, but the best state was still `2.794 rad` from upright, with
`4.388 rad/s` hinge RMS, `4.602 rad/s` absolute-rate RMS, and `2.091 m`
maximum cart excursion. No upright event or capture occurred. This closes
whole-waveform additive CEM around the current transfer as well as the
earlier phase-feature residual search.

An exact mass-matrix cart-acceleration energy-shaping CEM was also run for
`30` iterations and `48` candidates per iteration on the same wide rail. Its
best late state reached `1.834 rad` from upright but the cart then violated
the diagnostic rail at `12.030 m`; upright streak remained zero. The
interpretable energy expert did not provide a usable swing-up prefix.

Finally, the new builder option that keeps `K=50, c=1` fixed while changing
only split geometry and masses was screened at morphology progress
`0.05, 0.10, 0.20, 0.40, 0.70, 1.0`. Direct replays failed at every nonzero
progress, with instability or rail exits as early as `0.06 s`. A nominal
`p=0.01` FDDP run again produced an embedded success summary, but its
integrity-checked replay rejected it as numerical instability. The fixed-
support morphology gradient therefore does not rescue the support artifact.

Decision: the CEM, energy-shaping, and fixed-support morphology-gradient
branches are closed. Continue using the playbook requirement that every
candidate be independently replayed with finite-state integrity before its
metrics can influence the next search. The active 11-link problem remains
the free uniform hanging-start route on the canonical rail.

Artifacts:

- `runs/generalized_solver/n11_global_exact_cem_transfer_residual_rail12.json`
- `runs/generalized_solver/n11_energy_shaping_cem_rail12.json`
- `tmp/n11_free_split_stiffness_k50_constant.yaml`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p0p05_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p0p10_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p0p20_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p0p40_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p0p70_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p1p00_replay.json`
- `runs/generalized_solver/n11_free_split_stiffness_k50_constant_p0p01_fddp.json`

### Real crossing export, relaxed capture replay, and extended low-momentum CEM (2026-09-15)

The expert-chain evaluator was first replayed with the best known cart-target
trajectory and the capture gate relaxed to hinge RMS `<=3.0 rad/s`, cart speed
`<=1.0 m/s`, and `|x|<=2.0 m` on the diagnostic `+/-12 m` rail. The swing
expert reached a genuine upright crossing at `t=6.04 s` with maximum angle
`0.0786 rad`, hinge RMS `2.5916 rad/s`, cart speed `0.5504 m/s`, and
`x=1.8698 m`. The LQR capture action saturated on the next step; the chain
then reached `0.2431 rad`, `3.8084 rad/s`, and `0.8643 m/s` before exiting at
`12.0452 m`. The relaxed gate therefore exposes the failure but does not
define an admissible handoff.

The state exporter had been saving only rows after the stage switched to
capture, which could omit the exact pre-action state responsible for the
handoff. It now saves qualifying rows from the swing stage as well. The
replayable state file contains the real `t=6.04 s` crossing, including its
full `qpos/qvel`, so future capture training can use the state actually handed
over by the first expert.

A larger `40`-iteration, `64`-candidate CEM then searched the existing
cart-position trajectory family against the strict gate: angle `<=0.15`, hinge
RMS `<=0.90`, cart speed `<=0.50`, and `|x|<=2.0 m`. It improved the best
geometric crossing to `0.0548 rad`, but the handoff still had hinge RMS
`2.2851 rad/s` and cart speed `1.2485 m/s`; capture was never entered and
upright hold remained `0 s`. This is a stronger swing transient, not a solve.

Decision: preserve the real crossing-state artifact and keep the strict gate.
Do not train or claim capture from a post-switch state, and do not widen the
gate to turn a violent crossing into evidence. The cart-target trajectory
family is now saturated at an internal-rate boundary; the next route search
must change the late approach parameterization or use a nonlinear capture
teacher while retaining the exact saved-state and independent-replay checks.

Artifacts:

- `scripts/search_swingup_chain.py`
- `runs/generalized_solver/n11_chain_gate_exact_handoff_states.json`
- `runs/generalized_solver/n11_chain_gate_exact_handoff_state_replay.json`
- `runs/generalized_solver/n11_chain_gate_relaxed_capture_v2.json`
- `runs/generalized_solver/n11_chain_low_momentum_cem_extended_rail12.json`

### Timing-only refinement, capture-ready state searches, and settled launch (2026-09-15)

The next route searches kept the cart-position feedback structure but changed
the late timing and the handoff objective. A `20`-iteration timing-only CEM
reached a sharp `0.0211 rad` angle crossing, but hinge RMS was `2.1240
rad/s` and cart speed was `1.1173 m/s`; it did not enter capture. A
capture-ready CEM that ranked angle, hinge rate, absolute angular rate, cart
speed, and cart position found a genuine angle-ranked crossing at `0.1104
rad`, hinge RMS `1.3706 rad/s`, absolute-rate RMS `2.3754 rad/s`, cart speed
`1.3679 m/s`, and `x=1.6656 m`. A centered rate refinement degraded the main
score and still found no capture. These runs confirm that a low angle alone,
or even a low angle plus moderate hinge rate, is not the required interface.

The saved angle-ranked state was replayed from its exact `qpos/qvel` with two
nonlinear capture-actor CEMs: a full absolute/relative-state actor and a
lower-dimensional absolute-state actor. Their best centered upright streaks
were only `0.06 s` and `0.02 s`, respectively, with no low-momentum hold. The
capture expert cannot be credited from a hand-picked near-upright state when
the measured rates are outside its basin.

An open-loop direct-force waveform was then replayed with the same seed and
the same initial route candidate. It failed at about `0.34 s`, reaching
`3.141 rad` angle and the rail. This is a useful control-architecture result:
the successful predecessor route's cart-target feedback is doing essential
state-dependent work, so exporting its action waveform is not an equivalent
controller.

The evaluator now supports an explicit settled-launch phase. Before the
swing route starts, a hanging LQR centers the cart and damps the resting
chain; exported row timestamps include the conditioning time, and the result
stores the exact post-conditioning `qpos/qvel` as `conditioning_state`. The
old route was first replayed after an `8 s` conditioning phase. Its tiny
numerical launch drift changed the phase-sensitive trajectory and removed the
old `0.0786 rad` crossing, which is expected for an open-loop timing route.
The route was then re-optimized with the conditioning phase included in every
evaluation. The best settled-launch candidate reached a real upright crossing
at total time `21.00 s` with angle `0.1130 rad`, cart speed `0.0792 m/s`,
`x=0.4228 m`, and hinge RMS `3.0660 rad/s`; maximum cart excursion was
`2.730 m`. The strict capture gate remained closed and the hold was `0 s`.

The evaluator was then repaired again for capture-aware experiments: when the
gate opens, it snapshots the pre-action state as `stage=swing_handoff` before
the first capture action, while retaining post-action rows as `capture`. In a
diagnostic relaxed-gate replay, the exact handoff was `0.1199 rad`, `3.1415
rad/s` hinge RMS, `0.0498 m/s` cart speed, and `x=0.4216 m`. The LQR capture
action immediately saturated and the run later violated the rail, so this is
interface evidence and not a successful capture.

Decision: the settled launch is now part of the route interface and must be
used for future candidates, but it is not a solution. Keep the exact
pre-capture state requirement, the strict hinge/cart gate, and independent
replay. The next route budget should optimize the downstream capture margin
and the full-chain rates jointly, rather than reward another isolated angle
crossing. Do not update the public frontier, README, About metadata, paper,
or video until the canonical 11-link evidence contract passes.

Artifacts:

- `scripts/search_swingup_chain.py`
- `runs/generalized_solver/n11_chain_slow_late_time_cem_rail12.json`
- `runs/generalized_solver/n11_chain_capture_ready_absolute_rate_cem_rail12.json`
- `runs/generalized_solver/n11_chain_capture_ready_centered_rate_refine_rail12.json`
- `runs/generalized_solver/n11_full_state_capture_actor_from_cem_handoff.json`
- `runs/generalized_solver/n11_absolute_state_capture_actor_from_cem_handoff.json`
- `runs/generalized_solver/n11_direct_force_absolute_capture_ready_refine_rail12_seedmatched.json`
- `tmp/n11_capture_ready_state_minangle.json`
- `tmp/n11_conditioning_smoke.json`
- `tmp/n11_settled_launch_baseline.json`
- `runs/generalized_solver/n11_chain_settled_launch_capture_ready_rail12.json`
- `tmp/n11_settled_relaxed_capture_replay_v2.json`
- `tmp/n11_settled_relaxed_capture_states_v2.json`

### Sequential eleven-through-twenty campaign: corrected baselines (2026-10-01)

The user set a finite, sequential goal through twenty. The roadmap now records
that objective, the shared synthesis pattern, and the unchanged release gates.
`runs/frontier_campaign_20261001/campaign.json` prepares configurations for
11–20 and reserves disjoint 20/100/video seeds. These seeds have not been used
for tuning. Only eleven is active; none of these new configurations constitutes
a promotion.

Correctness changes precede further broad search. Feedback warm-start rebuilding
now returns both the controls actually applied and their resulting states.
FDDP receives measured initial feasibility instead of an unconditional flag;
an explicitly requested feasible start with nonzero dynamics gaps is rejected.
The exact environment and optimizer reject integration warnings and time
discontinuities. FDDP rejects invalid line-search trials using infinite cost
so they trigger smaller steps rather than silent simulator reset dynamics.
Riccati designs are checked for finite values, positive-definite returned
matrices, residuals, and closed-loop stability. These numerical checks are not
nonlinear capture certificates.

A separate opt-in continuous-angle route representation keeps the MuJoCo
transition, its derivatives, quadratic target costs, and Euclidean FDDP gaps
on the same joint-angle lift. The selected terminal winding is saved, and
runtime evaluators align nominal joint branches at launch. Existing artifacts
keep their original angle representation by default.

The bounded initial screens preserve the target force, rail, and action cadence:

| Screen | Result | Interpretation |
| --- | --- | --- |
| Corrected exact feedback rebuild, inherited 8-second route, 60-iteration cap | Stopped after 1 iteration; zero hold; rail exit at 3.0368 m | A dynamically consistent replay can still be a poor discovery seed |
| Transferred nominal with actual infeasible initialization, 100-iteration cap | First aborted on an invalid integration trial; after adding the rejection barrier, 14 iterations and rail exit at 3.0690 m | Correct feasibility activates gap repair but does not make the inherited route reachable |
| Continuous-angle transferred nominal, inherited terminal metric | 15 iterations; nominal still infeasible; zero hold; rail exit at 3.0275 m | Repairing the angle contract alone does not solve eleven |
| Continuous-angle route with reduced Lyapunov terminal weight, otherwise same settings | 14 iterations; nominal terminal value 1391.33 below the 1800 gate, but nominal still infeasible; live rail exit at 3.0322 m | A small nominal endpoint value cannot replace a dynamically feasible trajectory and uninterrupted replay |
| Canonical nonlinear FDDP capture from actual settled-route crossing at t=21 s; 3-second capture horizon and 9-second component episode | 100 iterations; 0.02 s upright streak; rail exit at 3.0349 m | This local capture seed does not bridge the measured swing crossing to maintenance within the remaining episode budget |

The crossing screen is a saved-state component experiment, not a hanging-start
solve. Its source came from a wider-rail search; capture itself used the canonical
plant. The initial baseline runs span successive correctness edits, and their
logs and commands record that limitation. Subsequent runs use
`scripts/run_frontier_experiment.py` to freeze source/configuration and declared
input artifacts before execution.

The subsequent frozen-source eleven-link replay measures a maximum final
state/control defect of `1.776927` in the declared scaled Euclidean coordinates,
against a feasibility tolerance of `1e-8`. Its median defect is `0.193963`.
Thus the low nominal endpoint value is attached to a strongly discontinuous
trajectory, and the independent runtime replay reproduces the rail failure.
The next optimizer must reduce the dynamics gaps, not merely refine that
nominal endpoint further.

The default upright design passes the new local numerical checks at eleven,
but fails at every audited count from twelve through twenty. This is a failure
of the current design procedure, not a physical impossibility result. A stable
local controller must be synthesized and tested before relying on it as the
next count's terminal expert.

Regression evidence: all 253 root tests pass after these changes, the ten-link
release verifier passes, and two fresh noisy parked-launch ten-link episodes
pass with 6.12-second maximum holds. A deterministic continuous-angle replay
also reproduces the ten-link 22.12-second hold and 2.0233-meter maximum cart
excursion. Frozen released controller artifacts were not replaced.

Decision: eleven remains active. The corrected local transfer baselines are
negative; subsequent discovery needs explicit dynamics-gap control and
actuator-aware arrival shaping. Preserve the phased architecture and record
any new branch generator as a separate lever. Continue higher-count feedback
diagnostics independently, and advance each count only after its full canonical
bundle passes.

Artifacts and exact initial commands:

- `runs/frontier_campaign_20261001/campaign.json`
- `runs/frontier_campaign_20261001/experiments.json`
- `runs/frontier_campaign_20261001/n10_integrity_regression_two_seeds.json`
- `runs/frontier_campaign_20261001/n10_continuous_replay_calibration/`
- `runs/frontier_campaign_20261001/n11_low_lyapunov_exact_replay/`

### Explicit sparse dynamics restoration (2026-10-01)

The next correction concerns control derivatives in the SciPy waypoint and
multiple-shooting paths. The exact transition casts normalized actions to
float32, while SciPy's default control perturbation can be smaller than one
representable action increment. At controls -1, +1, and the float32-exact
values near +/-0.9, the measured default derivative was zero; an explicit
1e-4 perturbation produced derivative norms of approximately 6.4–6.9 on the
same eleven-link state. This does not invalidate every old derivative:
the zero and 0.123456789 probes agreed. Fifteen inherited controls were near
saturation and nine were exactly representable in float32. The diagnostic
is `runs/frontier_campaign_20261001/action_derivative_quantization.json`.
Both SciPy repair paths now request a larger control perturbation. A regression
test requires a saturated float32 control to move to a reachable target.

`src/gcartpole/dynamics_restoration.py` adds a shared bounded sparse repair
problem with explicit transition Jacobians, continuous-angle dynamics,
force limits, and cart bounds at every decision node. It jointly adjusts
all route states and controls. Dynamics feasibility remains an independently
measured condition, separate from optimizer termination and terminal cost.
The runner preserves the original initial state and terminal winding and
saves an exact serial open-loop replay alongside the optimizer's nodes.

| Frozen experiment directory | Result | Interpretation |
| --- | --- | --- |
| `n11_sparse_restoration_d1e3_t1e3_15` | 15 residual evaluations, 158.32 s; maximum coordinate gap 1.776927 -> 0.016527; median 0.003510; 287,200 transition calls | Substantial dynamics repair, but still far above the 1e-8 tolerance. Exact open-loop terminal error is 2488.73 in the declared weighted continuous coordinates. |
| `n11_sparse_restoration_d1e5_t1e3_20` | Restart from the preceding output, stronger defect penalty; 20 residual evaluations, 141.24 s; maximum gap 0.006347, median 0.001359; 389,200 transition calls | The accumulated maximum-gap reduction is about 280 times. Actual serial replay still misses the target, with weighted terminal error 3591.49. This is a continuation, not an independent matched-start ablation. |
| `n11_restored_d1e3_fddp100` | 14 FDDP iterations, 31.76 s; maximum gap 0.016398; zero hold; live rail exit at 3.00635 m | The repaired seed does not by itself fix FDDP's loss of progress. |
| `n11_restored_d1e5_isotropic_fddp100` | Different repaired input and Lyapunov terminal weight reduced to 1e-14; 23 iterations, 87.86 s; measured final gaps exactly zero; zero hold; live rail exit at 3.02741 m | Closing dynamics gaps alone does not establish capture. The soft rail and endpoint objective still permit a physically poor route. This changes both input and terminal metric and cannot isolate the metric's effect. |

The sparse repairs' exact open-loop cart excursions are 2.45184 and 2.55219 m
over the eight-second route. Neither reaches capture. These fixed-start
component screens cannot establish a complete noisy hanging-start solve.
Their large unwrapped endpoint errors are not upright-angle errors in radians;
they include scaled velocities and the saved angle winding.

An experimental alternative uses a convexified dynamics subproblem, L1
penalties on virtual dynamics gaps, bounded state/control changes, and exact
nonlinear merit checks. It is inspired by the virtual-control and trust-region
construction in [Mao, Szmuk, and Acikmese (2016)](https://arxiv.org/abs/1608.05133),
without inheriting its convergence guarantees. Virtual gaps are optimizer
variables, never external forces or simulator resets, and must disappear in
exact feasibility checks. An initial implementation failed a controllable
double-integrator test because the QP penalty scaling caused repeated native
iteration limits. Its eleven-link run was stopped, retained as
`n11_sparse_l1_osqp_d1e3_t1e3_15`, and is invalid as physical evidence.
Rescaling the entire QP objective preserves its minimizer and repairs the
linear test: maximum gap approximately 1.4e-17, terminal norm below 1e-11.
The corrected eleven-link screen,
`n11_sparse_l1_osqp_scaled_d1e3_t1e3_15`, took 372.11 seconds and 302,400
transition evaluations. Only two of fifteen QPs produced accepted steps;
the others reached native iteration limits. L1 merit decreased from 4756.00
to 2500.52, but maximum gap increased from 0.006347 to 0.043068. The serial
rollout still missed the target (weighted terminal error 959.71). This
implementation is therefore not yet an effective restoration replacement:
the native QP accuracy policy and concentration of residual gaps need work.
All 257 root tests passed after the scaling fix.

A separate inspection found that the optional LTV route-feedback builder
omitted the `K.T R K` term when propagating its Riccati value matrix. The
term is now included, and a regression compares its first feedback gain
against an independently condensed finite-horizon quadratic optimum.
This correction affects newly built feedback, not frozen released gains.

Every completed screen has its command, declared input hashes, source/config
snapshot, output, runtime, and negative disposition. Later snapshots also
include tests and pyproject metadata; runtime metadata now records SciPy,
OSQP, and Crocoddyl versions when installed. Existing snapshots are retained
as executed. The active frontier remains eleven and final held-out cohorts
remain unused.

Final regression for this restoration round: 258 root tests pass. The frozen
`n10_restoration_changes_regression` screen uses two fresh noisy canonical
ten-link episodes (seeds beginning at 20261003), the released parked-route
controller, sixteen-second park, and -0.05 m target. Both succeed, with a
6.12-second maximum hold and 1.9949 m maximum cart excursion. This is a
regression screen, not a new robustness estimate. The next lever is exact
closed-loop tracking of the repaired route using the corrected LTV builder,
alongside better-conditioned constrained restoration; all candidates must
still connect to uninterrupted capture before eleven can advance.

### Feedback replay, finite capture intervals, and high-precision upright design (2026-10-01)

The corrected LTV builder now accepts saved optimization routes with their
fixed initial state, coordinate transform, and angle lift. It separately
measures the source's dynamics gaps; it never labels repaired optimizer
nodes as actual trajectories. The ten-link calibration uses the same
control cost 0.5 and terminal scale 1000 as the eleven-link test and
reproduces the 22.12-second hold and 2.02328 m cart maximum. Eleven's repaired
route instead exits the rail at 3.03187 m after 182 transitions, with zero
hold. Forty-nine of those actions are near saturation, compared with only
three near-saturated feedforward controls in the whole 400-step nominal
route. This is an actual tracking failure, not a successful endpoint with
an inconvenient handoff threshold.

The next L1 restoration screen permits approximate native-QP candidates
only when their recorded residuals satisfy an explicit tolerance, then
checks the exact nonlinear merit and gap. Native nonconvergence no longer
shrinks the trust region as though it measured nonlinear model accuracy;
derivatives are reused while the state/control decision is unchanged.
`n11_sparse_l1_inexact_monotone_d1e3_t1e3_15` nevertheless fails: 424.61 s,
148,400 transition calls, maximum coordinate gap 0.007638 and exact serial
terminal error 1719.32. Its monotonic guard used the maximum node L2 norm,
whereas the established feasibility screen uses the largest absolute
coordinate (L-infinity). The former decreased while the latter increased.
New code makes the guard use L-infinity too, with an adversarial regression
that rejects exactly this mismatch. Existing frozen runs retain their
executed norms. The label "Euclidean StateVector" identifies the unwrapped
coordinate space, not an L2 feasibility metric.

`high_precision_discrete_lqr` implements the structure-preserving doubling
recurrence from [Poloni (2020), equation 33](https://arxiv.org/html/2005.08903).
An 80–100 digit calculation produces stabilizing designs for the supplied
linear matrices at every count from 11 through 20. It promotes existing
binary64 finite-difference matrices exactly; it does not recover missing
identification digits, change MuJoCo arithmetic, or certify nonlinear
capture. Gains are rounded back to binary64 for actual execution. Neither
the default runtime design nor the released 7–10 gains is replaced.

| Links | High-precision upright gain norm | Exact local replay successes at amplitude 1e-9 / 1e-11 / 1e-13 |
| --- | --- | --- |
| 11 | 1.612e6 | 4/4 / 4/4 / 4/4 |
| 12 | 1.043e7 | 0/4 / 4/4 / 4/4 |
| 13 | 7.063e7 | 0/4 / 0/4 / 4/4 |
| 14 | 4.938e8 | 0/4 / 0/4 / 0/4 |
| 15 | 3.518e9 | 0/4 / 0/4 / 0/4 |
| 16 | 2.530e10 | 0/4 / 0/4 / 0/4 |
| 17 | 1.820e11 | 0/4 / 0/4 / 0/4 |
| 18 | 1.299e12 | 0/4 / 0/4 / 0/4 |
| 19 | 9.135e12 | 0/4 / 0/4 / 0/4 |
| 20 | 6.326e13 | 0/4 / 0/4 / 0/4 |

Each column contains four saved uniform random joint-angle/rate directions
with cart position/velocity zero, an eight-second canonical component
episode, and exact runtime dynamics. Amplitude denotes the multiplier on
those directions, not a basin-radius estimate or initial-noise benchmark.
The table uses the later replay that preserves tiny in-range angles.
The earlier high-precision screens test fewer amplitudes and are retained
separately. A sampled failure does not establish physical impossibility.

The same rounded twenty-link gain gives a computed spectral radius of
approximately 0.996223 when assembly and eigenvalue calculation use 100
digits, versus 17585.6 with binary64 assembly and NumPy eigenvalues. This
exposes severe sensitivity in the numerical linear analysis; it cannot
establish that the implemented nonlinear controller is stable. The actual
twenty-link component replays still fail. The default twelve-link DARE
failure and the nonlinear capture failure must therefore be reported as
different problems.

The common wrapping expression `(angle + pi) % (2*pi) - pi` also erases or
quantizes very small upright errors. `gcartpole.angles.wrap_angle` now
preserves angles already in [-pi, pi) without arithmetic, retaining the
old convention elsewhere. Environment feedback, transformed coordinates,
capture features, and predictive sampling share it. Tiny-angle and pi-boundary
regressions pass. The fresh noisy ten-link replay after this change is 2/2
with 6.12-second maximum hold and 1.9783 m maximum cart excursion. The local
probe table still does not constitute an end-to-end solve at any new count.

This evidence motivates a finite upright interval in trajectory synthesis,
rather than relying only on arrival at an increasingly narrow static-LQR
funnel. The sparse runner can append a five-second capture interval, retain
the exact target dynamics and action cadence, and bound every absolute-angle
decision coordinate inside 0.14 radians of the saved upright winding. Those
are optimizer-node constraints; independent dynamics feasibility and actual
uninterrupted replay remain mandatory. Its first canonical eleven-link
screen is `n11_capture_interval5s_sparse_d1e5_t1e3_30`; its measured outcome
is still negative. Thirty residual evaluations and 996,450 transition calls
take 741.12 s. Maximum coordinate gap drops from 0.090319 to 0.004995, median
gap reaches 0.000423, and the final nominal weighted endpoint error is
5.38e-5. All nominal capture-window absolute angles lie within approximately
0.00368 rad of upright, well inside the imposed 0.14 rad bound. Exact serial
replay nevertheless misses the terminal target (weighted error 2016.49).
Nominal upright occupancy therefore still cannot replace feasible replay.

Building feedback over this longer upright window exposes another numerical
failure: dense Riccati propagation returns a negative control Hessian
(-79357.54 at step 528) despite strictly positive costs. The builder now
propagates a QR square-root Joseph factor, including the control-cost term,
so it evaluates control curvature through squared factors rather than an
ill-conditioned dense matrix. Tests compare both a well-conditioned case
and a weak-input-mode case against independent optimal-control calculations.
The new thirteen-second builder completes, but its actual eleven-link
feedback replay still exits the rail at 3.02575 m with zero hold after 211
transitions. This fixes the numerical failure, not the reachability problem.

The subsequent exact FDDP screen weights the five-second upright interval
separately (1000, with angle/rate factors 20/4), rather than using only the
last endpoint. `n11_capture_interval5s_fddp100` stops after 18 iterations,
82.43 s, and closes its measured dynamics gaps exactly. Its feasible route
still has poor capture cost and the live controller exits at 3.05719 m with
zero hold. These finite-interval inherited-route screens remain negative;
the next branch generator must produce a different physical route rather
than another attractive discontinuous endpoint. All 267 root tests pass
after the square-root feedback and phased-cost changes. Eleven remains the
active frontier and no final held-out seeds have been used.

Artifacts:

- `runs/frontier_campaign_20261001/n11_restored_ltv_r05_t1000_v2/`
- `runs/frontier_campaign_20261001/n11_restored_ltv_r05_t1000_replay/`
- `runs/frontier_campaign_20261001/n10_ltv_r05_t1000_calibration_v2/`
- `runs/frontier_campaign_20261001/n10_ltv_r05_t1000_replay/`
- `runs/frontier_campaign_20261001/n11_n12_n13_high_precision_lqr80/`
- `runs/frontier_campaign_20261001/n14_to_n20_high_precision_lqr100/`
- `runs/frontier_campaign_20261001/n11_to_n20_preserved_small_angles_local_replay/`
- `runs/frontier_campaign_20261001/high_precision_lqr_limits.png` and `.pdf`
- `runs/frontier_campaign_20261001/n10_angle_precision_regression/`
- `runs/frontier_campaign_20261001/n11_capture_interval5s_sparse_d1e5_t1e3_30/`
- `runs/frontier_campaign_20261001/n11_capture_interval5s_sqrt_ltv_r05_t1000/`
- `runs/frontier_campaign_20261001/n11_capture_interval5s_sqrt_ltv_replay/`
- `runs/frontier_campaign_20261001/n11_capture_interval5s_fddp100/`

## Physical descent seeds and inverse-dynamics spline screen (2026-10-01)

The next branch generator uses actual controlled descent on the canonical
eleven-link plant. A deterministic 54-case sweep varies a brief upright
cart-force kick, cart velocity damping, and position feedback, selecting
quiet hanging endpoints after at least six seconds. The best bounded
descent ends at 11.7 seconds with hanging-quality score 0.2657 and maximum
cart excursion approximately 0.0903 m. It still has a 1.4086 physical-state
norm mismatch from the exact hanging launch after reversal. Reversing its
states and velocity signs does not reverse joint dissipation: its maximum
discrete coordinate defect is 6.828, median 0.5515. The saved first-node
projection is an optimizer reference, never a runtime reset. Quiet descent
is therefore useful seed information but not a feasible upward route.

A subsequent 48-case screen balances the reversed force waveform's impulse
and first moment before clipping, varies force scale and timing, and adds
explicitly recorded cart-rail feedback. Every branch is replayed forward
from the target's exact hanging state. None captures. The best selected
prefix reaches a maximum absolute angle of 2.072 rad at 4.46 seconds, far
outside the 0.15-rad threshold. These are new physically feasible route
prefixes, not transferred or reset trajectories.

Exact Box-FDDP refinement of two retained branches remains negative. The
first, with a four-second appended interval, stops at 16 iterations in
81.50 seconds. Its measured gaps close exactly, but replay exits the rail
at 3.00436 m after 322 steps with zero hold. The second consumes all 200
iterations, 1396.83 seconds, and 3,544,617 optimization transition calls.
Its measured gaps are also zero, with final optimization cost 542395.44;
feedback replay exits at 3.02000 m after 368 steps with zero hold. Dynamics
feasibility alone does not establish a useful arrival or a successful
feedback connection to maintenance.

An independent formulation now represents cart and joint positions with
quintic B-splines, fixes hanging and upright endpoint positions, and enforces
zero endpoint velocity and acceleration. The optimizer penalizes required
unactuated joint forces and cart force above 80 N. Unactuated forces are
never applied during benchmark replay. A separate cart-only RK4 replay and
discrete-defect measurement remain mandatory. MuJoCo's inverse-derivative
helper rejects RK4 models, so this implementation differentiates continuous
inverse dynamics directly on the unchanged model and uses its mass matrix
for acceleration derivatives. Independent directional differences and
cart-only forward acceleration checks verify those calculations.

The initial eight-second, twenty-control-point torque formulation converges
on two links in 28 residual evaluations, but still requires up to 1.724 Nm
of joint torque, has maximum local coordinate gap 0.2803, and achieves zero
hold. Increasing to forty points and 100 evaluations reduces the gap to
0.07238 and required joint torque to 0.5687 Nm, still with zero hold. Thus
this initial spline parameterization is not yet calibrated as a reliable
solver even at the smaller count.

The matched eleven-link twenty-point torque screen consumes 50 residual
evaluations and 34.16 seconds. Its maximum normalized joint-torque residual
is 0.2152, yet the same missing forces produce up to 309.03 in the mixed
cart/relative-joint acceleration vector (m/s² and rad/s², respectively).
Its maximum discrete coordinate gap is 7.9859. Gravity-normalized torque
residuals therefore hide important inertial amplification in this example.

The alternative residual explicitly measures `M(q)^-1 * [0, tau_joint]`,
the acceleration discrepancy caused by omitting those joint forces while
applying the required cart force. In a matched 50-evaluation eleven-link
screen, maximum dense acceleration discrepancy drops to 4.835 and maximum
discrete coordinate gap to 0.1862. It takes 26.72 seconds and 992,353 inverse
dynamics calls. The actual replay still holds upright for zero seconds,
with maximum cart excursion 2.7289 m. The matched two-link acceleration
screen also remains negative (gap 0.1348, zero hold). These differing
objectives expose a scaling issue; they do not establish overall superiority
or a solution. All 275 root tests pass. Eleven remains the active frontier,
all these bounded jobs have completed, and final evaluation seeds remain
unused.

The full commands, frozen inputs, physical replays, optimizer budgets, and
negative outcomes are indexed in
`runs/frontier_campaign_20261001/reverse_descent_and_inverse_spline_experiments.json`.
The next spline work must establish a feasible small-count calibration and
introduce adaptive trajectory resolution or better route initialization,
rather than treating least-squares convergence as completion. A feasible
eleven-link candidate still needs a sustained capture interval and the
unchanged held-out release gates.

## Control cadence, runtime precision, and refined spline calibration (2026-10-01)

The user proposed that more links require more frequent and more precise
corrections. The cadence audit separates action frequency from physics
accuracy: it retains the canonical 0.005-second RK4 integrator and changes
only `frame_skip` from four to two to one, yielding 50, 100, and 200 Hz.
An independent check confirms that held force over the same four physics
steps produces exactly identical physical states in all three variants.
Every rate receives a newly designed upright LQR with the same Q/R weights;
high-precision Riccati arithmetic is used where necessary. The initial
relative-joint angle/rate directions are identical across rates at each
count. Changed cadences are diagnostic benchmark variants, not releases.

The fastest estimated upright growth rate rises only from approximately
27.005/s at ten links to 28.497/s at twenty in this particular compiled
model. At eleven, its amplification over one control interval changes
from 1.7329 at 50 Hz to 1.3164 at 100 Hz and 1.1473 at 200 Hz. Faster
updates therefore materially reduce growth between corrections. However,
the gain norm also increases in this controller family: approximately
1.61e6, 6.36e6, and 1.32e7 at eleven, respectively. At twenty it reaches
approximately 6.33e13, 1.11e15, and 5.06e15. Those gains increase sensitivity
to saturation and arithmetic; the observed growth rates do not by
themselves explain the sharp link-count performance cliff.

The first 180 local component replays use ten, eleven, twelve, fourteen,
and twenty links, three rates, three perturbation amplitudes, and four
matched directions per amplitude. All rates yield the same success totals
at the coarse tested amplitudes: ten and eleven pass 4/4 at 1e-9 and 1e-13,
twelve passes only the 1e-13 cohort, and fourteen and twenty fail all three
cohorts. Every count fails the 1e-5 cohort. An additional 216 replays
resolve the smaller-count boundary more finely:

| Count and initial amplitude | 50 Hz | 100 Hz | 200 Hz |
| --- | --- | --- | --- |
| 10, 1e-7 | 2/4 | 3/4 | 3/4 |
| 11, 1e-8 | 1/4 | 2/4 | 2/4 |
| 12, 1e-10 | 3/4 | 4/4 | 4/4 |

There is a small positive effect in these matched cohorts. Four directions
cannot establish a robustness distribution or a basin radius, and this
does not demonstrate a hanging-start solve. In particular, route arrivals
can have strongly correlated errors, so generic near-equilibrium
perturbation amplitudes are not tolerances on the released hanging-start
noise distribution. The twenty-link local tests still fail at 200 Hz.

A separate 360-replay precision screen holds the saved gains, initial
directions, and physics fixed while comparing float32 actions with binary64
dot products, float64 actions with binary64 dot products, and float64
actions with 80-digit dot products. The latter promotes the saved rounded
gain and current binary64 state; it cannot recover precision lost in model
identification or gain storage. All three variants give identical success
totals in the sampled cohorts. Eleven passes all tested amplitudes from
1e-9 down to 1e-19; fourteen fails 1e-9 and 1e-13 but passes 4/4 from 1e-15
downward; twenty fails every tested amplitude at both 50 and 200 Hz. The
diagnostic float32 stepping agrees exactly with canonical stepping in a
unit comparison and all 48 overlapping full audit episodes. Action casting
and dot-product accumulation alone do not explain these particular
failures. This is not a proof that faster control, different controllers,
or more accurate models cannot help.

The rate/precision comparisons support retaining frequency as an explicit
experimental variable while investigating weak actuator coupling, model
and gain conditioning, and arrival structure. The implementation defaults
and fifty-hertz release gates remain unchanged. Relevant methodological
background is the [MuJoCo integration documentation](https://mujoco.readthedocs.io/en/3.3.5/computation/)
and the [MIT discrete/continuous LQR notes](https://underactuated.mit.edu/lqr.html).
The saved scientific figure is
`runs/frontier_campaign_20261001/control_frequency_figure/control_frequency.png`
with a matching PDF.

Meanwhile, exact midpoint knot insertion now increases spline resolution
without changing the saved curve, velocity, or acceleration. Refining the
two-link forty-point spline to 145 points and optimizing for 200 residual
evaluations reduces maximum required joint torque to 0.007456 Nm. Its
50-Hz discrete gap remains 0.05230 and actual open-loop hold is only 0.18 s:
continuous force feasibility and zero-order-held cart force remain distinct.
Exact FDDP then closes the measured gap to 3.94e-10. Its first execution
uses the script's historical zero tracking-feedback default and holds for
1.7 s before rail exit. Explicit selection of its unscaled saved solver
feedback, with tracking and upright LQR scales both one, succeeds from
exact hanging over the complete thirty-second calibration episode: 24.6 s
maximum uninterrupted hold, 2.6063 m maximum cart excursion, and LQR
handoff at eight seconds. This is a small-count calibration of the shared
pipeline, not an eleven-link release or a held-out gate.

At eleven, refining the generic twenty-point acceleration spline to 65
points and spending 100 residual evaluations yields gap 0.1421, dense
acceleration discrepancy 1.1449, and zero hold. Two subsequent FDDP jobs
stop at LQR/Lyapunov preflight because the commands omit the established
`lqr-scale=1` and inherit 1.3, whose linear spectral radius is 3.5276.
These are setup errors, not completed optimization failures. With the
correct explicit feedback settings, the thirteen-second FDDP screen
completes 14 iterations in 63.20 s and closes its gaps exactly, but replay
exits at 3.02636 m after 29 steps with zero hold.

Transferring the resolved two-link spline to eleven interpolates absolute
angles at normalized link centers, preserving the cart curve and endpoint
conditions. It is an optimizer initialization, not inherited dynamics
feasibility. A 100-evaluation target-plant acceleration refinement yields
gap 0.3876, dense acceleration discrepancy 2.3348, and zero hold, despite
remaining inside the rail during the eight-second open-loop replay.
Smaller count increments are the next calibration of this route family;
the active frontier remains eleven. All 280 tests pass. The fresh ten-link
reference regression passes 2/2 with 6.12-second maximum hold and 1.9823 m
maximum cart excursion. Complete commands, source hashes, budgets, and
outcomes are indexed in
`runs/frontier_campaign_20261001/frequency_and_spline_resolution_experiments.json`.
Reserved final evaluation seeds remain unused.

The first smaller increment, two to three links, now completes the same
fixed hanging-start calibration. Material-coordinate spline transfer,
100 target inverse-dynamics residual evaluations, and 100 exact FDDP
iterations produce an exactly feasible nominal route. The thirty-second
feedback episode holds upright for 24.24 s, reaches maximum cart excursion
2.85143 m, and hands off to upright LQR at eight seconds. Optimization
takes 150.32 s. These are discovery/calibration results, not new releases;
the unchanged seven-to-ten artifacts and eleven-link promotion gate remain
authoritative. The four-link increment initially fails: 26 FDDP iterations
leave maximum gap 0.12549 and the actual episode exits the rail with zero
hold. Thirty sparse restoration evaluations reduce the gap to 0.00019222;
the serial open-loop rollout still misses the target by 888.98 in the
weighted terminal coordinates. A further 100 FDDP iterations close the gap
exactly and the full feedback episode succeeds: 24.44-second hold,
2.85115 m maximum cart excursion, and LQR handoff at eight seconds.

### Adjacent inverse-route synthesis and feasibility preconditioning

`scripts/synthesize_inverse_increment.py` now packages material-coordinate
transfer, acceleration-residual inverse refinement, exact target FDDP with
explicit feedback settings, conditional sparse repair, and full exact-start
replay. It refuses a source without a dynamically feasible nominal route
and a successful complete thirty-second physical replay, and refuses
non-adjacent source/target counts. This is a discovery screen; its successful
two-, three-, and four-link calibrations are not noisy evaluation gates or
frontier promotions. Each nested stage has a frozen execution journal.

The first five-link run exposes a failure of post-hoc conditional repair.
Its inverse route has gap 0.19516. Box-FDDP initially reduces nominal cost
from about 2189 to 2133 while reducing gaps only slightly. It eventually
closes gaps to zero by accepting a route costing 91,965,959.75. Actual
feedback replay then exits at 3.04139 m after 250 steps with zero hold.
Native feasibility and measured zero defects correctly describe the new
route, but say nothing about whether that route captures upright. Since
the nominal route is now feasible, the old conditional repair does not
run; repairing this already poor route is not the intended next experiment.

A matched prerequisite-repair screen instead starts from the raw inverse
route. Thirty evaluations reduce its maximum gap from 0.19516 to
0.00010793 in 74.04 s while retaining a small nominal weighted endpoint
error. Serial rollout still has weighted terminal error 1044.78, so this
remains a warm start rather than a capture result. The subsequent 100
FDDP iterations close gaps exactly and keep cost at 2133.31; actual feedback
replay succeeds for all 1500 steps with 24.22-second hold, 2.79883 m maximum
cart excursion, and LQR handoff at eight seconds. This matched five-link
comparison changes only prerequisite restoration before the same FDDP
and execution recipe. Its optimization costs include 74.04 s restoration
plus 289.37 s FDDP, versus 101.37 s FDDP in the failed baseline. It supports
the usefulness of preconditioning this candidate, not a universal
convergence claim. The runner exposes this
experimental order as `--pre-restore`, preserving the earlier recipe for
comparison. The same prerequisite-repair test is applied to the active
eleven-link material-transfer route. The six-link calibration increment
uses the successful five-link source and the same prerequisite-repair
recipe; it cannot promote eleven or change the released seven-to-ten
controllers.

Eleven-link prerequisite restoration reduces maximum gap from 0.38761 to
0.00109497 in 242.51 s, but serial terminal error remains 1766.39. Native
FDDP accepts an initial uphill feasibility step from nominal cost about
4807 to 1.549e9. It stops after 31 iterations in 108.48 s with measured
zero gaps, yet actual replay has zero hold and exits the rail. This
reproduces the separation between feasible route and useful capture.

The [Crocoddyl 3.2.1 acceptance rule](https://github.com/loco-3d/crocoddyl/blob/v3.2.1/src/core/solvers/fddp.cpp)
permits uphill steps for infeasible trajectories using default threshold
two. An explicit `--fddp-ascent-acceptance 0` ablation retains nominal cost
about 4807 but stops after fourteen iterations in 22.58 s without closing
the 0.001095 gaps; actual replay still fails with zero hold. Its minimum
nominal Lyapunov value 3.98 does not certify an executable arrival. This
option changes only native acceptance settings when supplied; defaults
are preserved. Neither cost preservation alone nor unconditional gap
closure solves this route. A matched thirty-evaluation restoration with
LSMR inner budget 3000 instead of 300 reduces final maximum gap further to
7.34354e-5 and restoration cost from 11.56394 to 0.03510. Both trials use
the same raw inverse route, thirty outer evaluations, weights, and physical
plant; wall times are 242.51 and 241.86 s. The terminal nominal error is
0.0070853, but serial terminal error is still 1998.63. Better linear solves
improve the nominal conditioning result without establishing actual
capture. Matching native FDDP stops after 25 iterations in 75.3 s with
measured zero gaps, but its high-cost route again has zero actual hold and
exits at 3.025 m. The extra inner iterations improve the warm-start defect
metric without improving capture in this direct two-to-eleven route family.

The shared prerequisite-repair recipe subsequently passes six-link
calibration with 23.54-second hold, 2.85036 m maximum cart excursion,
eight-second LQR handoff, full 1500-step clean replay, and exactly zero
measured nominal gaps. The seven-link increment then passes with
23.56-second hold, 2.38183 m maximum cart excursion, full clean 1500-step
episode, and zero measured gaps. Its prerequisite repair reduces gap from
0.28102 to 0.00018134 in 126.0 s; FDDP takes 173.79 s.
Complete stage commands, input/source hashes,
and budget/result records are indexed in
`runs/frontier_campaign_20261001/inverse_increment_experiments.json`.
The five-link preconditioning comparison spends extra compute and is not
an equal-compute ablation. No reserved held-out seeds are used.

A separate smaller finite-difference probe uses upright identification
epsilon 1e-11 instead of 1e-7 with 100-digit Riccati design. It reproduces
the sampled fourteen- and twenty-link local outcomes: fourteen passes
4/4 at amplitudes 1e-17 and 1e-19 and fails at 1e-9 and 1e-13; twenty fails
all four directions at all four amplitudes. Changing that identification
step alone does not resolve the tested failures. This is local diagnostic
evidence, not hanging-start evidence or an impossibility claim. All 282
root tests pass after the runner and coordinate-provenance changes.

### Eight-link calibration: capture before the nominal endpoint

The next identical inverse/pre-restoration/FDDP recipe reaches a feasible
eight-link route but fails the full episode. Inverse gap 0.31799 falls to
0.00030900 after thirty restoration evaluations (135.50 s), then to zero
after 100 FDDP iterations (378.23 s). At the forced eight-second handoff,
largest absolute angle is only 0.00013965 rad and relative hinge-velocity
RMS is 0.0016600, but Lyapunov value is 592.11. The first upright-LQR action
is -0.08489 normalized, followed by growing oscillations and saturation.
Maximum hold is 2.22 s and the rail exits at 3.01068 m after 493 steps.
Tiny componentwise endpoint errors do not establish a usable capture.

The same physical route reaches smaller Lyapunov values earlier. Removing
only `--defer-handoff-until-horizon`, with the original threshold 1800 and
all other state gates, switches at 6.54 s and succeeds for all 1500 steps:
23.48-second hold and 2.303 m maximum cart excursion. The optimization,
saved trajectory, gains, plant, rate, and development seed are unchanged;
no new optimization, reset, or state overwrite occurs. A separate
threshold-five replay switches at 7.12 s and also passes. Timing alone is
therefore sufficient to recover this particular baseline; tightening the
threshold is not needed for its recovery. The shared runner now exposes
`--state-gated-handoff` and a recorded optional threshold. A fresh complete
eight-link synthesis reproduces the failure and automatically recovers it
through a replay-only early-capture stage: handoff 6.54 s, hold 23.48 s,
maximum cart 2.30256 m, full clean episode. The driver tries this cheap
recovery after a nominally feasible route fails, before requiring another
optimization. The nine-link experiments use threshold five.

Two further 100-iteration refinements start from the same feasible failed
eight-link route while preserving its forced eight-second handoff. Weight
0.01 reduces nominal terminal Lyapunov value to 41.76 and full replay passes
with 23.46-second hold. Weight 18 reduces it to 1.46 and also passes with
23.46-second hold. Both have about 2.379 m maximum cart excursion. The
code divides terminal Lyapunov weight by the handoff threshold, so weight
18 corresponds to coefficient 0.01 on P at threshold 1800, matching the
isotropic running state cost. P solves the preset linear LQR's state-cost
Lyapunov equation; this interpretation applies locally without saturation,
does not include the future control-cost term, and is not a nonlinear
certificate. The first 0.01 trial was initially described as matched scale
before the normalizer was noticed; it is retained as a weaker-pressure
trial. The matched additional 100 iterations with original weight 1e-14
fail: nominal endpoint value 1239.58, hold 1.88 s, rail 3.02963 m after
481 steps. All three continuations start from the same feasible candidate
and use the same iteration budget, seed, gates, and forced handoff. Extra
iterations alone do not explain these two value-weight recoveries. This is
a controlled comparison for one candidate, not a general robustness claim.

Transferring the old inverse spline also omits source FDDP corrections:
at seven, the passed route differs by up to 0.24934 m cart position and
0.11351 rad absolute link angle. `fit_inverse_spline_from_route.py` fits
positions and velocities to the same clamped quintic basis, aligns the
initial angle branch, refuses incompatible terminal winding, and validates
the source replay and full configuration hash. Two first invocations fail
at setup (direct-script import, then an extra reconstructed config field);
both are preserved and are not algorithmic negatives. The corrected fit
has errors 0.004187 m and 0.002178 rad. Matched transfer/refinement to eight
still fails its forced-horizon replay: 2.12-second hold, rail 3.030 m,
terminal Lyapunov value 956.27. Better curve transfer alone does not fix
this tested handoff. The fitter now declares physical versus nominal
curve selection; physical mode includes the actually executed early
capture interval. An eight-link physical fit is available as a subsequent
initialization, with no inherited dynamics-feasibility claim.

The scientific comparison plot and PDF are under
`runs/frontier_campaign_20261001/n8_capture_handoff_comparison_figure_v3/`.
The experiment ledger records every completed and pending stage. All 287
tests pass. Runtime evidence now reads Crocoddyl's module version when
Conda supplies no distribution metadata, reporting 3.2.1 for new runs;
older null version fields are not rewritten. The active frontier remains
eleven, frozen seven-to-ten releases are unchanged, and final seeds remain
unused. These exact-start discovery episodes are not noisy release gates.

### Nine-link recovery: sparse inner accuracy and faithful gate evaluation

The original nine-link increment uses the passed threshold-five eight-link
controller and its raw inverse spline. Thirty pre-restoration evaluations
with LSMR cap 300 leave maximum gap 0.00073880. FDDP accepts a feasible
high-cost route after 24 iterations (56.97 s, cost 337,933,996.77), with zero
hold and rail exit 3.04639 m after 294 steps. Changing only the LSMR inner
cap to 3000, with the same outer budgets, source inputs, state gate, seed,
and plant, leaves gap 0.00007592. Subsequent 100-iteration FDDP closes the
gaps and passes the full episode: hold 23.50 s, rail maximum 2.31032 m,
handoff 7.14 s. Stronger sparse solves recover capture here, unlike the
previous direct two-to-eleven route. The initializer's conditioning matters
to which feasible solution FDDP reaches.

A matched physical-curve refit from the passed eight-link controller also
passes nine with the stronger inner budget: hold 23.52 s, cart 2.388 m,
zero measured gaps. The raw transfer is simpler and has slightly more rail
margin; it is selected for the ten-link increment, which is running with
the same 100/30/100 outer budgets, LSMR cap 3000, and threshold-five state
capture. This remains development calibration, not sequential promotion.

The parked evaluator previously discarded the saved early-capture flag and
forced the entire route, potentially restoring the bad handoff recovered
above. It now opts into state capture only for an explicit false
`defer_handoff_until_horizon` flag. It reconstructs the certified Lyapunov
metric and requires its hash to match the artifact, applies all saved gates
around the parked cart target, and latches capture without a state overwrite
or reset. Missing flags retain the legacy full-route behavior. Tests check
translation, angle/velocity limits, irreversible capture latching, one
reset, and legacy timing. The two new zero-noise eight/nine replays reproduce
their 6.54/7.14-second handoffs and 23.48/23.50-second holds.

Four fresh noisy development seeds per count with 16-second hanging-LQR
parking at -0.05 m then pass 4/4 at both eight and nine: holds 7.48/7.50 s,
maximum cart excursions 2.37382/2.37281 m. Eight captures at 22.54 s; nine
at 23.12–23.14 s. No reserved gate seeds are used. A fresh legacy ten-link
regression remains 2/2 with 6.12-second hold and 1.97114 m maximum cart
excursion. The released artifacts are unchanged, eleven is unsolved, and
these small development screens do not replace its required final gates.

### Shared ten calibration and the adjacent eleven failure

The ten-link 100/30/100 recipe with LSMR cap 3000 reaches independently
feasible gaps 6.9368e-9 but fails its threshold-five execution. The minimum
actual value is 5.20011, so the gate never opens; hold is 1.54 s and rail
exit is 3.04936 m. Replaying unchanged controls, nominal states, and gains
with switch thresholds 10 and 25 passes all 1500 steps: handoffs 6.96 and
6.78 s, holds 23.48 s, maximum cart 2.07770 m. Thresholds 100 and 1800 both
switch at 6.54 s and fail after 365 steps with 0.32-second hold. The
optimization handoff normalizer remains five during these replay-only
comparisons; only `switch_lyapunov` changes. This separates objective
normalization from policy selection.

The driver now tries a bounded development grid [5, 10, 25, 100, 1800]
after a feasible route fails, retaining the optimizer objective and stopping
at its first complete passing replay. A fresh entire ten synthesis reproduces
the baseline failure and automatically selects threshold ten. This is
development selection, not an independent robustness gate. Exact zero-noise
parked replay reproduces hold 23.48 s and handoff 6.96 s.

On four noisy starts with 16-second parking, the new ten policy passes 3/4;
seed 20261032 fails before capture. Changing only parking to 17.5 s recovers
all four, with 5.98-second hold. A fresh matched twenty-seed cohort passes
15/20 at 16 s and 20/20 at 17.5 s (maximum cart 2.15515 m). A separate
hundred-seed development cohort at 17.5 s passes 93/100. These results are
not the reserved frontier gates and do not claim the old ten release's
100/100 robustness. They show that a nominal discovery success still
requires launch-distribution contraction and independent noisy validation.

The first adjacent eleven transfer begins from the new successful ten
candidate, not from an old released frontier waveform. Pre-restoration
reduces maximum gap 0.34641 to 7.16145e-5 in 436.6 s under parallel CPU
load. Native 100-budget FDDP stops after 29 iterations (122.2 s), closes
all gaps, and destroys the useful route: cost 911,354,293.45, zero hold,
rail 3.06049 m after 208 steps. Minimum actual capture value is 5,798,899,
far above every predeclared grid threshold, so alternative gates cannot
open anywhere in this identical failed trajectory. No runtime reset occurs.

Fitting the passed ten's physical curve is accurate (cart 0.003723 m,
absolute angle 0.002768 rad, hinge-velocity RMS error 0.01837), but matched
adjacent-eleven synthesis still produces a feasible failed route: cost
562,912,750.73, zero hold, rail 3.05969 m after 268 steps. Extra seventy
restoration evaluations from the raw pre-restored route reduce gap only to
6.36352e-5; subsequent FDDP again closes gaps and fails capture. Raising
initial FDDP regularization to one instead of 1e-6 also fails. Curve fitting,
extra outer iterations, and this regularization change are insufficient
for the tested branch.

Two targeted next comparisons retain the original low-defect initializer:
dynamics penalty 1e7 instead of 1e5 reduces maximum gap to 9.78959e-7 after
thirty evaluations (210.9 s), with larger terminal norm 0.00813. Separately,
FDDP terminal Lyapunov weight 0.1 at optimizer normalizer ten gives
coefficient 0.01 on the preset LQR state-value matrix, matching the running
state coefficient. The initializer's terminal value is 116,262.16 despite
small componentwise errors. This value-weighted refinement is still running;
it retains a low-cost route while closing gaps, unlike the baseline cost
explosion. Full replay remains required before claiming eleven capture.

### Evidence video must depict the evaluated controller

The legacy parked video script uses a different hanging regulator from the
parked evaluator. New gate evidence should not silently substitute that
policy. `render_parked_evaluation_trace.py` instead renders every recorded
physical post-step state from the evaluator, checks complete cadence,
finite states, action/rail limits, pose-derived upright status and hold,
and verifies the source config and generated plant geometry hashes. It
does not resimulate or interpolate states. The camera uses one isotropic
fixed scale and fits the entire uniform chain through twenty, including
extreme rail/upright/horizontal/hanging poses. Legacy render defaults and
released videos remain unchanged.

A ten-link development video contains the 1500 recorded frames from seed
20261030's successful 17.5-second parked episode, with 24.46-second capture
and 5.98-second hold. Hanging, swing, and capture frames were visually
checked; source/controller/config/geometry/video hashes and the executed
gate are saved under `n10_shared_evaluator_trace_video`. This is a
development video, not eleven's reserved video or a new release promotion.
All 291 root tests pass.

### Eleven passes the canonical bundle; twelve becomes the active frontier

The later directional-value continuations complete the earlier eleven
comparison. From the same useful 200-iteration source, another 100 iterations
with regularization 10 and fixed eight-second handoff fails at coefficient
0.01 on the saved capture value (3.70-second hold, rail loss). Changing only
that coefficient to one succeeds for all 30 seconds: 23.50-second hold,
maximum cart 2.071622 m, independently measured local gap 1.70e−10. An offline
feedback rebuild saves the actually applied controls and physically rolled
states; the same stronger-value refinement then has exactly zero checked
initial/final gaps, passes full physical replay, and reaches 2.071341 m.
The native solver has not converged; acceptance uses the checked full replay.

This also sharpens the feasibility correction. Crocoddyl's initial-feasibility
flag now requires exactly zero gaps, not merely reporting tolerance 1e−8.
A tested unstable example shows that individually tiny gaps can accumulate
into a materially different serial rollout. Reporting tolerance remains a
separate diagnostic, and the first nonzero-gap successful candidate remains
preserved rather than relabeled as exactly feasible.

Initial eleven noisy screens pass only 2/4 after sixteen-second parking and
3/4 after seventeen-and-a-half. Rebuilding the source does not itself recover
the failing seed. Hanging regulator action penalties 1000, 100 and 10 all
give the same 3/4 at 17.5 s. Route feedback scales .75 and 1.25 fail all four;
changing the parked target to zero also does not recover the failing seed.
Longer default parking passes all four at 18 and 18.4 s but leaves only
5.50 and 5.10 seconds for the hold.

The causal launch improvement comes from the cart-state cost. At 17.5 s,
weights 10 for cart position and 5 for cart velocity reduce the failing
seed's launch tilt 7.67e−4→6.80e−7 rad and speed .009325→8.34e−6 m/s. It
then passes using the identical swing controls/gains. Weights 1/1 also recover
the four-seed case. The stronger 10/5 design passes all four at sixteen
seconds with 7.50-second hold. On a fresh matched twenty, default weights
give 16/20 at 17.5 s and 19/20 at 18 s; 10/5 gives 20/20 at both 16 and
17.5 s. A separate development hundred passes 100/100 at sixteen seconds.
The remaining launch distribution, not nominal capture alone, was causal.

The policy is frozen before reserved seeds are used. An isolated local Git
source bundle at commit 1920582683b2fd500386a986b276951469966c45 includes
the policy, route, configs, source and tests. A fresh clean clone passes
reserved noisy 20/20, disjoint noisy 100/100, exact 20/20 and video seed
211500. Every episode completes 1500 steps with 7.50-second hold; the
noisy hundred's maximum cart excursion is 2.121369 m. The trace video uses
the actual 1500 physical states and no resimulation/interpolation/reset.
Hanging, swing and capture frames were visually inspected. A new
count-agnostic verifier checks the unchanged plant/noise contract, policy,
gains, source cleanliness, disjoint reserved cohorts, full-episode success,
actual pose-derived video hold and artifact hashes. It passes without errors.
The manifest and appendix are under `runs/swingup11_uniform/` and
`docs/eleven_link_swingup_paper.md`. Nothing has been publicly published.

All 302 root tests pass at release, with 99 focused tests in the clean
source clone. The existing ten verifier and a new 2/2 noisy regression pass.
The evaluator now distinguishes an earlier five-second success from full
episode completion; a hold followed by rail failure cannot generate accepted
release evidence. Failed requested evidence is saved before rejection.

Twelve is now active; all higher counts remain queued. The shared synthesis
runner records a bounded capture-value recovery schedule, restarting from
the useful pre-restored initializer instead of a collapsed feasible route.
Coefficients .01/.01/1 each receive the declared refinement budget, with
regularization 1e−6/10/10 and exact offline feedback rebuilding before the
last stage. This encodes the intervention developed at eleven; independent
automatic re-synthesis has not yet validated it.

An opt-in precision path supplies promoted-input Riccati designs and a
factored Lyapunov terminal value. It checks stability of the rounded gain
in extended precision, solves the Lyapunov equation for that actual rounded
gain, and computes values/gradients through the saved factor. A regression
demonstrates a small direction lost by Gram-matrix rounding but retained
by the factor. This does not recover lost identification digits or certify
nonlinear capture. Default production behavior and the frozen eleven source
remain unchanged. In the ongoing local development checkout, all 304 root tests pass, and a fresh four-seed
eleven replay after these opt-in changes passes 4/4. The first twelve
adjacent inverse/repair/refinement pipeline is running with 80-digit design,
the same canonical plant, source eleven release and separate development
seed 20261048. No twelve release seeds have been used.

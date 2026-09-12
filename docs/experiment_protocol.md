# Experiment Protocol

## Core hypothesis

For an n-link cart-pole, direct training on uniform links is unnecessarily hard. A gradiented morphology can improve practical controllability. A curriculum can then anneal gradients away until the plant is uniform.

## Morphology generator

For link index `i = 1..n`, define normalized position:

```math
s_i = \frac{i-1}{n-1}
```

For a parameter budget `P` and gradient strength `alpha_p`:

```math
p_i = P \frac{e^{-\alpha_p s_i}}{\sum_j e^{-\alpha_p s_j}}
```

This is used for:

```math
\ell_i, m_i, d_i
```

Positive `alpha` means base-heavy/base-long/base-damped. `alpha=0` is uniform.

## Curriculum

Default mode: `mass_last`.

1. Remove damping gradient first.
2. Remove damping budget toward target.
3. Remove length gradient.
4. Remove mass gradient last.

Reasoning: damping is the easiest/passive crutch, length gradients can create fast top modes, and mass gradient seems like the strongest practical conditioning knob.

## Fixed-Profile Morphology Screen

The implementation exposes a fixed `--progress` diagnostic in the exact
global search tools. This matters because curriculum progress is not itself a
morphology value: `swingup_slow` intentionally keeps the full p0 gradient
unchanged through its early damping stages. Every profile must record its
realized length, mass, damping, and friction vectors.

The first exact serial screen used the same 32-parameter energy/modal
controller at p0, p0.25, p0.50, p0.75, and p1.00, with a true hanging start,
fixed `+/-12 m` development rail, 12-second horizon, 12 CEM iterations, and
24 candidates per iteration. The best angles were `0.572`, `0.902`, `0.775`,
`0.778`, and `0.931 rad`, respectively. All five profiles had `0/2` upright
evaluation episodes and `0 s` sustained hold. The p0 gradient is the strongest
discovery condition in this small screen, but the result is not monotonic and
does not transfer a capture policy or satisfy any canonical gate.

Batched global-CEM profile proposals are not counted until serial replay
matches their saved states. The first nonuniform profiles failed that audit by
large qpos/qvel errors and remained near hanging in serial replay. This is a
required negative control for any future morphology sweep.

The isolated axis screen then used the same exact energy/modal CEM at p0 on a
fixed `+/-12 m` rail, with true hanging initialization and canonical damping
or friction budgets unless that axis was the one being tested. Length-only,
mass-only, damping-only, and friction-only profiles reached best angles of
approximately `0.849`, `1.106`, `1.177`, and `0.959 rad`, respectively, with
`0/2` upright evaluation episodes in every case. The coupled base-heavy p0
profile remained stronger than each isolated axis in the comparable small
screen. A live receding-horizon CEM found a length-only geometric crossing at
`0.165 rad` and `0.997` height proxy, but it carried `25.95 rad/s` relative
hinge RMS, `21.98 rad/s` cumulative absolute-rate RMS, and `25.92 m/s` cart
velocity. This is a phase-synchronization diagnostic, not a handoff.

All live planner searches must disable `action_lqr_residual` and
`action_lqr_switch` internally. A prior run inherited those teachers from the
curriculum config and was rejected because its batched predictions did not
match the applied rollout. Tail proposals must likewise be replayed through
the exact serial environment; `scripts/replay_swingup_tail.py` now performs
that check from a recorded `trajectory`/`trace` state and supports fixed
morphology progress.

## Required controls

Run these eventually:

| Run | Purpose |
|---|---|
| uniform direct PPO | Baseline difficulty |
| length-gradient only | Tests musical/length intuition |
| mass-gradient only | Tests strongest current hypothesis |
| full gradient | Tests combined curriculum |
| shuffled gradient | Tests whether ordered gradient matters |
| gradient-to-uniform | Actual claim |

## Acceptance metrics

For each run record:

- wall-clock time
- total environment steps
- mean return
- success rate
- mean episode length
- max absolute link angle
- checkpoint path
- video path
- config hash / config copy

## Global Discovery Checkpoint

The external packet review added a feasibility checkpoint before broad policy scaling. It is not a replacement for the P1, P2, or P3 gates.

Use the exact uniform six-link MuJoCo plant and the published hanging-start distribution. A valid checkpoint must contain one reset-free feedback rollout that reaches the all-link upright condition and holds for at least `5 s`, plus replay from three additional held-out hanging-start seeds. Do not initialize from an exported handoff state.

Save the complete `qpos`, `qvel`, action sequence, seed, resolved config hash, controller/checkpoint hash, action cadence, capture time, maximum continuous hold, maximum cart excursion, rail termination, and numerical/domain status. If a coarse simulator, collocation path, or optimizer proposes the behavior, replay the candidate in exact MuJoCo before labeling it a teacher trajectory.

The training curriculum may use longer rails, altered morphology, friction, easier starts, reverse resets, or saved promising states, but each is a named training condition. Final discovery checks return to the canonical six-link rail, force, damping, action frequency, and hanging-start distribution. Report global discovery, capture, sustained hold, numerical validity, and verification as separate outcomes.

If this checkpoint fails within the declared compute budget, record the failure and change the discovery method. Do not treat a faster verifier, a lower collocation defect, a higher shaped return, or a state-specific capture controller as evidence that the global swing-up problem has been solved.

## Master handoff integration rules

The Cart-Pole Research Master Handoff is a source of protocol components, not a replacement benchmark. When borrowing its mechanics or learning code:

- keep the packet's legacy/nonuniform `+/-15 N`, `+/-2.4 m` rail, actuator lag, and 19 s horizon under a separately named matched-benchmark config;
- log the normalized policy action, the held command, the actually applied force, and any actuator state at every policy interval;
- forward-replay saved teacher or handoff states in the exact target simulator before using them as labels, and never use a saved handoff as final evaluation initialization;
- retain one known positive replay and one known negative replay in the integration test suite;
- compare morphology proposals with energy-gap-matched controls, then restore uniform geometry before a canonical gate;
- treat a numerically closed orbit, stable linear poles, or an optimizer endpoint as a proposal until the canonical hanging-start MuJoCo rollout reaches and holds upright.

## Measured Handoff Requirement

Morphology gradients are allowed to generate training conditions, but the
capture expert must be initialized from states actually visited by the swing
expert. Use `scripts/extract_trajectory_states.py` to filter a serial
`trajectory` or `trace` artifact into a state list. The output records the
source file hash, source row, physical `qpos`/`qvel`, absolute-angle metrics,
relative and cumulative rates, cart state, and selection bounds.

Do not fill a sparse handoff set with synthetic interpolations and call it a
successful two-expert transfer. A state list with only a few high-rate states
is evidence that the swing expert needs a broader, quieter arrival curriculum.
Evaluate the capture expert on both the measured states and held-out nearby
states, and report success, first-upright, capture-quality, rate, cart, rail,
and uninterrupted hold metrics separately.

The first implementation of this rule is recorded in
`runs/swingup7_p0000_capture_states_from_real_swing.json`. It contained three
real p0 states from a `+/-12 m` planner trace. Both a fixed-p=0 PPO capture
probe and a serial linear CEM actor failed on them, so those artifacts remain
negative component evidence. The next curriculum should deliberately expand
the measured capture basin from lower-rate states before asking the swing
expert to produce high-energy seven-link arrivals.

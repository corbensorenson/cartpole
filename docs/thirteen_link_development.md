# Thirteen-link development chronology

Thirteen is now validated by all canonical reserved gates and a full video;
fourteen is active. The chronology below records the development state before
freezing. Statements about unused reserved seeds refer to that historical
stage. Final evidence is in [the canonical appendix](thirteen_link_swingup_paper.md).
The benchmark remains the uniform 3 m, 1 kg chain, 1 kg cart, 80 N force limit,
3 m rail, 50 Hz control and uninterrupted thirty-second physical episode.
Canonical promotion still requires noisy20/20, disjoint noisy100/100,
exact20/20, a held-out full video, a clean frozen source and hash verification.

The adjacent initialization uses the released nine-second twelve-link physical
route. Material interpolation maps absolute angles and absolute angular rates
at normalized link centers, then converts them back to relative joint states.
Cart position and velocity are retained. This preserves geometry, not target
plant dynamics. An explicitly virtual five-second capture tail supplies offline
optimizer nodes; it must never replace runtime plant states.

Fitting the released physical curve initially failed because the spline
expected +pi while the executed trajectory began at -pi and ended at zero.
A constant 2pi translation also moves the endpoint, so it does not preserve the
intended winding. A fit with both physical endpoint branches preserved has
maximum cart error0.002781 m and maximum absolute-angle error0.003926 rad.
The inverse optimizer's separate hardcoded +pi start check then rejected the
equivalent -pi hanging branch. It now validates canonical hanging modulo whole
turns, finite dimensions, exact zero cart/velocity and agreement with the
spline's first point, while retaining the supplied winding. All failed attempts
and source snapshots are retained.

The fitted inverse initializer runs100 inverse evaluations,30 sparse
restoration evaluations,100 FDDP iterations, another30 restoration evaluations,
and bounded capture-value refinements with100-digit Riccati design. FDDP
accepts zero iterations on these thirteen-link cases. The restored FDDP screen
has maximum coordinate gap0.000632922 and fails physical replay on the rail at
7.32 s with no hold. Native FDDP can expose nonzero backward-pass gains even
when it accepts no optimization iterations; iteration count alone does not mean
that the returned gain array is zero. The later feedback rebuild has exactly
zero dynamics gaps but retains a physically unsuccessful route. Exact
feasibility alone is therefore insufficient.

The direct physical-curve initializer starts with maximum physical gap
0.0635511. Twenty full-horizon residual-SCvx iterations reduce this to
0.00101238; nominal cart motion remains within2.031 m. Actual feedback replay
instead saturates force and violates the rail at4.04 s with no hold. This is a
negative development result, not capture evidence. The inspected comparison
figure and exact source hashes are in
`runs/frontier_campaign_20261001/n13_adjacent_initialization_failure_report/`.
It separates virtual nominal nodes from actual uninterrupted plant trajectories.

Two continuation branches start from that exact same saved twenty-iteration
source: forty additional iterations with OSQP(max10000 inner iterations) or
Clarabel0.11.1(max200 inner iterations), both with initial tolerance cap1e-5,
floor1e-8 and the same physical objective. Iteration budgets differ by native
algorithm and are reported explicitly; this is not an equal-inner-iteration
comparison. Native Clarabel residuals are normalized, whereas native OSQP
residuals are absolute; comparing their printed residual magnitudes directly
would be incorrect. The newest implementation separately measures raw QP
bound violations. Clarabel statuses other than Solved/AlmostSolved cannot be
reinterpreted using the OSQP inexact-candidate rule. Neither solver's status
certifies nonlinear dynamics or physical capture.

The legacy trust-region rule lowers regularization after every accepted step,
even alpha0.001, while retaining the large state/action region. The running
continuation exhibits many such tiny steps. An optional agreement policy
contracts the region and raises regularization after very short steps or poor
predicted-versus-actual merit agreement; accurately predicted full steps can
expand it. A matched forty-iteration Clarabel branch tests this change with the
same source and objective. The published twelve settings remain unchanged.
A curved-plant regression demonstrates contraction after tiny accepted steps,
and both native QP backends match an independent dense linear-horizon oracle.

A separate probe recomputes pure LTV QR tracking gains around the inverse
restored route, with zero added feedback regularization and unchanged controls
and nominal nodes. Its actual episode fails on the rail at5.00 s with zero
hold. This does not rescue the initializer.

A third trajectory repair takes the inverse/restored nodes in physical
coordinates and appends the same explicit five-second virtual capture tail.
Its initial maximum physical gap is0.000974290. A twenty-iteration full-horizon
SCvx screen uses the same successful twelve-link recipe as the direct-source
screen. Outcomes and compute are saved before any subsequent decision.

The reusable entry points now cover physical material transfer, explicit
virtual capture initialization, full-horizon capture synthesis, unchanged-prefix
handoff replays, and physical rebuilding of delivered float32 actions. They are
`scripts/transfer_physical_capture_route.py`,
`scripts/append_virtual_capture_initializer.py`,
`scripts/synthesize_frontier_capture.py`,
`scripts/replay_frontier_handoff.py`, and
`scripts/rebuild_frontier_physical_route.py`. The last two independently
reproduce every physical state of the released twelve-link exact-start route
and its23.50 s hold. A new shared entry point is a reproducibility tool, not a
claim that automatic count advancement or general morphology solving is closed.

All401 tests and both frozen eleven/twelve release verifiers pass as of this
update. Experiments carry frozen source/input snapshots and exact commands.


Thirteen exact-start recovery (2026-10-02): the forty-iteration Clarabel
continuation with nonlinear agreement updates ends at maximum physical
dynamics gap4.50616e-8. Actual replay holds upright7.64 s, then fails after
the14-second LQR handoff. Without changing controls, feedback or physical
prefixes, handoffs9,10 and12 s pass the complete thirty-second episode with
23.50 s hold and maximum cart excursion1.9934265 m;8 s fails. Nine is the
earliest passing time in this predeclared screen. It is selected for an
exact delivered-control physical rebuild and fresh noisy development gates.
Thirteen is not promoted: no reserved validation has been run. The legacy
Clarabel continuation ends with gap0.000182364 and fails at6.06 s; the
inverse/restored twenty-iteration residual-SCvx branch ends with gap
0.000378893 and fails at4.26 s. Every negative and positive source is retained.


Thirteen noisy development recovery (2026-10-02): the selected nine-second
route is rebuilt to actual delivered float32 controls and exactly zero
physical shooting gaps. Every state of the independently replayed full
thirty-second episode is bit-identical to the original successful handoff
replay. The unchanged parked launch(16 s,cart costs10/5,R1000,target-0.05 m)
passes fresh noisy20/20(seed20263200) and disjoint100/100(seed20263300).
Every episode completes1500 steps and holds upright7.50 s; maximum cart
excursion is1.9434625 m. The policy and source are now selected for freezing
before reserved validation, without any reserved-seed tuning.

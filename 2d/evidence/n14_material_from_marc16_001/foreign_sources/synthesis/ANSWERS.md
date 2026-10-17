# Answers to the synthesis questions

These answers distinguish the selected construction, retained measurements and
unresolved causal questions. The [recipes](RECIPES.md) link the runnable stages;
[original-manifest.json](original-manifest.json) binds the exported inputs and
source. Public adapters and projected historical metadata were authored after
qualification. They do not retroactively become part of a controller freeze.

## 1. How the successful references were generated

**Fifteen.** The starting warm data were Corben's public partial-fifteen
`actual_minimax.npz`, SHA256
`6e0d87f4fbc12d1f0f35929b17e8182b767c6f33ee7642a202ff324861c2e22e`.
It was reflected into the canonical `+pi` hanging branch and used to initialize
a separate 650-command constrained repair. The retained repaired reference
passed its analytic planning-map checks, but direct native tracking still failed.

A subsequent native-value controller produced a complete actual exact-hanging
episode with a 24.18-second upright interval. The first **700 delivered commands
and 701 actual states** of that episode became the physical reference used by
the released construction. The selected reference was therefore a recorded
native trajectory, not simply the optimizer's nominal states. Those states,
the earlier repaired reference and the complete source episode are included.

The selected final prototype regenerated feedback/value arrays on that physical
reference, added the fifteen-specific parking design and joint native capture
search, and then packaged a fixed list of 32 native forecast profiles. The
released planner selects the first full valid continuation from the observed
postparking state. All relevant prototype and payload arrays are included.

**Sixteen.** The warm donor was Corben's successful fourteen-controller JSON,
SHA256 `6bc6d9ea44c5fb7eba0b5182aeaaa7cd391721d52301fefc157e5ca99111a34d`.
Its 450-command route was mirrored as needed, lifted by material-coordinate
interpolation of cumulative absolute angles/rates, and used in the original
700-command static-tail repair. That repair was an initializer, not a successful
physical controller.

The successful early-bend construction used the actual native state after 800
parking commands from exact hanging. It optimized a 450-command bridge to a
separately constructed cyclic orbit, then appended 250 fixed orbit commands.
The source, initial guess, raw solver output, assembled reference, original
launch tape and selected orbit are included. `16-launch` independently
reconstructs the actual parked prefix from exact hanging.

The workers expose the exact numerical settings: IPOPT limited-memory Hessian,
MUMPS linear solver, no NLP scaling or bound relaxation, adaptive barrier,
150 iterations and 180 CPU seconds for the sixteen reference/bridge solves.
The base repair uses tolerances `1e-10`/`1e-11`; the delivered-float32 early
bridge uses `1e-9`/`1e-9`. Objectives penalize deviation from the warm controls
and states at `1e-4` and `1e-7`. These are bounded deterministic optimizations
with no random optimizer seed. Initial-state seeds and every actual original
sixteen argv/resource cap are recorded separately. The fifteen repair's exact
earlier argv/revision was not retained; its reconstruction limits are explicit
in [RECIPES.md](RECIPES.md) and [validation](validation/README.md).

An optimizer reference is not a physical success certificate. The selected
sixteen reference's maximum native cold-map gap is approximately `3.06e-6`;
that defect enters its affine feedback construction. The actual successful
native episode and source calibration are separate recordings with separate
native replay checks. The export preserves the reference's original feasibility
flag rather than replacing it with the controller's success.

## 2. The five-hertz bent upright motion

Five hertz came from the registered screen **2.5, 5 and 6.25 Hz**. It was not
identified as a unique physical optimum. The original worker and all three
closed candidate orbits are included.

For each frequency, the initializer computes the complex held-input response
`(exp(i*omega*0.02) I - A)^-1 B` of the upright discrete linearization. Each
link's phase and amplitude come from that response; the spatial bend is not
drawn by assigning identical angles to all links. Force amplitude is limited
by an initial 0.08-radian cumulative-angle target, normalized force cap 0.5
and cart-amplitude cap 0.75 m. Controls are an antisymmetric float32 cosine
sequence. At 5 Hz the period is ten command ticks and the saved reference
contains ten periods. Initial normalized force amplitude is approximately
0.3505258391, or 28.0421 N.

The force sequence is held fixed while Newton iterations refine the cyclic
states. A mean-cart gauge replaces one closure equation for the solve, then
**all original equations**, including the replaced row, are independently
checked against analytic and native transitions. The worker allows eight
Newton iterations, six halvings, residual tolerance `1e-10`, maximum absolute
angle `< 0.12`, cart magnitude `<= 2.5` and normalized force `<= 0.5`.

Among closed candidates, the fixed selection rule minimizes the maximum norm
of the 250-step zero-terminal QR feedback gains, with profile index as a tie
breaker. Five hertz was selected. Its refined maximum absolute angle is about
0.07498 rad and cart magnitude 0.02482 m. Its maximum gain norm is about
`1.0647e8`, versus `2.4977e10` for the matched static finite-250 design.
That roughly 235-fold component comparison is not a whole-controller ablation.

The orbit was constructed **before** the swing bridge. The bridge uses a
quintic position blend with the corresponding velocity derivative, beginning
at route tick 150, and is then optimized to orbit phase zero at tick 450.
The fixed 250-command orbit tail is appended afterward. The orbit and full
swing were not jointly optimized.

## 3. The actual feedback construction

Both designs use native **relative** joint coordinates and rates, including
the cart: fifteen has `[q16, v16]` and sixteen `[q17, v17]`. State units remain
metres, radians and their rates; no learned embedding or whitening is used.
The action is normalized cart force with physical force `80*u`. Translation
is applied only to feedback errors, never to the native world-frame plant.

With `T` the lower-triangular cumulative-angle transform, the running weights
are cart position/rate 0.1, joint-position block `100*T.T*T + I`, joint-rate
block `T.T*T + 0.01*I`, and normalized action cost **1000**. The derivative
provider differentiates the implicit mass equation and propagates its
derivatives through four held-input RK4 steps using the compiled capsule
constants, relative damping and armature. Its source and rebuild command are
included; it is separate from native execution.

The selected trajectory recursion is **float64 square-root QR**. At each
backward step it stacks the input penalty, propagated value root, stage root
and an affine column. The native defect is
`F_native(reference_state, delivered_action) - next_reference_state`;
the propagated affine column is `root @ defect + residual`. The QR result
supplies the gain, affine action offset, new value root and residual. No
additional diagonal regularizer or state scaling is inserted into this QR
construction. Fifteen retains all value roots/residuals for native value
search. Sixteen's simple policy uses the resulting gains and offsets.

**Fifteen's selected `offline_riccati_precision="double"` is correct.** It
refers to this trajectory/value recursion. The separate stationary upright
terminal design was calculated at 100 decimal digits and rounded to float64.
The tested quad recursion was not the selected prototype. Sixteen's upright
component also used 100-digit design, while probe07 used float64 QR feedback
with a zero terminal root over the whole 700-command reference. Live physics,
state, feedback products and delivered actions remain explicitly typed.

`regenerate.py feedback` regenerates all selected payload arrays and compares
them byte for byte. `15-prototype` and `16-probe07` expose the original worker
commands as well. The original probe07 SHA256 is
`d71eec45605803395859c780a965497dcd2fbf78fb8857f908012a4bb5522507`.

## 4. What the finite terminal design does and does not explain

The selected probe07 uses a zero terminal value over **700** route commands,
then literal zero force after the finite route. Later timing selection retains
the first **650** actions/gains/offsets/defects and first 651 states, while
changing parking from 800 to 850 commands. The backward recursion is not run
again at 650. The selected table's SHA256 is
`de5afbb5db053a5094e01bd913bf0f7d2f4b17abd1a2679d3821f341e60ffcdb`.

The finite approach was explored after an accurately designed stationary
upright controller failed native capture even at extremely small perturbations.
Retained matched **local static** comparisons include infinite feedback,
finite-250, finite-300 and receding-100 feedback; none established complete
local capture across the declared cases. Increasing arithmetic precision and
changing the stationary input penalty also did not recover the tested cases.
Consequently zero terminal cost alone is not the demonstrated breakthrough.

There is **no retained clean full-hanging comparison** that holds the successful
early-bend route and all other settings fixed while replacing only its terminal
root with the stationary infinite-horizon root. Nor is the 650-entry prefix a
newly optimized 650-horizon design. The export preserves these limits instead
of presenting a hindsight claim that the terminal choice was necessary or optimal.

## 5. Development sequence and useful before/after evidence

The first full exact-hanging sixteen success was the simple **early-bend bridge
plus finite affine feedback**, with 800 parking ticks and no predictive addon.
It completed 1,500 commands with a 417-tick upright interval. The earlier
bridge's full episode failed; both references/tapes are included. This is a
documented successful construction, but route and its derived gains changed
together, so it does not isolate each coefficient's causal effect.

The unchanged successful probe07 then passed nine of ten noisy diagnostics.
The retained failing launch had a smaller launch-error norm than some successes,
then lost tracking around route ticks 275–300 and saturated. Thus launch-error
norm alone was not a sufficient selector. The exported failing tape permits
more specific analyses.

On that saved launch, zero cart-error offset reproduced the failing suffix
exactly, while registered offsets of magnitude `1e-6` through `1e-3` produced
full passing forecasts. Offsets of magnitude `1e-2` failed. Those were forecast
components, not replacement physical episodes.

A separate matched duration screen tried 17, 18 and 19 seconds parking on the
same ten saved noisy starts plus exact. All 33 full episodes passed. The
predeclared shortest-successful choice selected 17 seconds with the unchanged
650-entry table prefix. Fresh development100 subsequently reached **98/100**,
so longer parking alone still did not qualify the controller.

Finally the bounded observed-state profile selector, zero offset first, passed
new development20+100 and the once-only reserved gates. Two development100
cases needed `+1e-6` m; all held-out cases selected zero. The selected forecasts
matched the actual suffixes exactly. Native forecasts/profile selection improved
the later robustness path; they were not what produced the first nominal pass.
Original qualification and negative-predecessor evidence remain in the
[canonical sixteen release](https://github.com/marctheshark3/pendulum-arena/releases/tag/canonical16-20261016).

For fifteen the retained sequence is repaired nominal reference, first native
value-controlled success, physical reference extraction, improved hanging
contraction, joint native capture search and bounded full-route profile selection.
The receipts preserve failed alternatives. The smaller source export does not
turn comparisons with several changed settings into single-variable ablations.

## 6. Parking construction and selected parameters

**Sixteen:** centered native held-input linearization at exact hanging, state
perturbation `1e-5`, delivered float32 input perturbation and four native RK4
steps per command. The selected `angles10` DARE uses cart-position cost 10,
cart-rate cost 5, baseline diagonal `1e-8`, cumulative-angle cost 10,
cumulative-rate cost 1 and action cost 1000. Its target is cart `-0.05` m,
first relative hinge `pi`, remaining relative coordinates/rates zero. DARE
and gain construction use SciPy float64. No decay scaling is used here.

The five-profile screen used the same ten retained diagnostic initial states.
The incumbent median 16-second state error was about 0.0240; `angles10` reached
`1.6771e-7` with worst cumulative-angle error `8.5325e-8`. These are the measured
screen cases, not a global contraction guarantee. The full native-derivative
and cost-screen workers, selected matrix checkpoint, costs and outcomes are
included. `16-parking` regenerates them.

**Fifteen:** the same cart/cumulative-angle/cumulative-rate weights and action
cost, but using the selected analytic hanging derivative, cart target zero
and exponential design scaling `exp(1.2*0.02)` applied to **both A and B** before
the DARE. The gain is computed from those scaled matrices. Native episode time
and dynamics are not scaled. The selected final controller parks 800 commands.
The physical derivative, decay comparison and actual selected gain are exported;
the short feedback reconstruction redoes this calculation and checks its bytes.

## Learning and seventeen

The retained failures suggest a possible job in **arrival-route proposals** or
ranking difficult launch candidates under a bounded native-validation budget.
They do not establish that latent forecasting would improve the exact native
physics. On sixteen, all held-out episodes already selected the first profile,
so faster ranking has no demonstrated held-out benefit there. Useful learning
would have to reduce measured total search cost or improve physical outcomes.

The ongoing seventeen work has run one small, separately frozen JEPA proposal
probe: eight whole training trajectories, four fresh evaluation trajectories,
36 native state coordinates, immediate delivered actions, a 32-value latent,
64-unit hidden layers and 400 fixed updates. It forecast eight commands and
ranked three route initializers with disabled, shuffled and random controls.
It failed its declared forecast/action-response gates; its learned ranking was
the same as the random model's, and its first choice had about 1.815 times the
native initializer defect of the classical first choice. No subsequent repair
arms or hybrid qualification were activated. This is a narrow negative result,
not a general rejection of a learned/classical method.

That observation supports collecting informative branches around the difficult
arrival/capture states and checking action sensitivity before spending native
repair budgets on a learned proposal. Preserve whole-episode splits, equal
budgets, native validation, classical feedback and failures. A proposed job
remains a hypothesis until it beats its matched controls on physical success
or total computation. The canonical sixteen result itself has no JEPA component.

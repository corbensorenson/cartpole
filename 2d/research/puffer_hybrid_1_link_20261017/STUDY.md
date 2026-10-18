# Qualified one-link classical-swing / learned-capture hybrid

The fixed hybrid passes45/45 original hanging development starts and32/32 new
noisy qualification starts. Every case completes30 seconds, with at least21.58
seconds continuously upright. The first8 seconds use the original classical
swing controller; a PufferLib-trained policy captures and balances for the
remaining22 seconds. No physical state reset or hanging wait is added.

This is a new learned-capture method track at one link. Fourteen prepared links
and ten immediate links remain the established overall frontiers. The result
does not establish pure learned swing-up or higher-link learned control.

## Portable inference and the transfer issue

The selected Puffer checkpoint is unchanged from the preceding balance study:
`b8134743906935239d5ae8366951a1f9ab411f3fc33a09a724413aefad06632a`.
It was saved at8,192,000 controls during the completed9,998,336-control learner.
The original GPU policy passed45/45 upright resets but0/13 hanging resets.

The CPU reader respects16-byte tensor alignment and implements the pinned
encoder, two highway MinGRU layers and mean decoder in float32. On19,500 frozen
GPU observations, the largest normalized-action difference is1.717e-5 and the
largest delivered-force difference0.001030 N. This is numerical agreement, not
bit-identical GPU inference. The CPU implementation is treated as a new policy
and receives its own actual physics tests.

The successful classical swing reaches upright with a negative quaternion scalar
component. The balance training resets used the positive representative. Since
q and−q encode the same orientation, this difference should not change physical
control, but the raw network receives different values.

The observer now chooses the quaternion representative whose first nonzero
component is positive, and canonicalizes signed zero. It does not alter the
simulator quaternion or any physical velocity. All39 antipodal observation
fixtures are byte-identical. Raw and canonical CPU policies both pass13/13
upright starts; all paired complete closed loops match exactly there.

## Local capture before a continuous hanging-start test

Capture tests start from actual recorded one-link states at controls400,425,
450,525 and600. Each has one noiseless and two paired spatially noisy starts.
The original velocities remain, with noise added to all five velocity
coordinates. The incoming force from the recorded predecessor is supplied to
the learned observer. Each local case then runs its own full30-second clock.

| Observation representation | Local capture passes |
|---|---:|
| Raw signed quaternion | 0/15 |
| Canonical quaternion sign | 15/15 |

Every paired initial physical state is byte-identical. The earliest tested
phase400 has about0.42 rad tilt,−1.78 m/s cart X velocity and1.59 rad/s joint
angular motion. It lies outside the0.15 rad upright band. These local cases
demonstrate transfer of capture, not original hanging-start qualification.

## Fixed continuous hybrid and qualification

The full controller executes the frozen classical400-command route, then
initializes the learned recurrent memory at zero. Its first learned input uses
the actual incoming delivered force. The learned clock counts commands since
activation; the actual simulator step count never resets. The controller's
owned geometry copy cannot advance or modify the physical environment.

All thirteen available classical nominal/development400-command prefixes match
the earlier accepted tapes byte for byte. The full hybrid passes13/13 pilots,
then45/45 development starts. The controller source, route, weights and handoff
are frozen before using new seeds20330116..20330147. All32 fresh starts pass;
no parameter or checkpoint is selected using them.

Independent protocol audits reconstruct every learned observation from the full
actual state, every normalized action from a freshly initialized network, all
delivered forces, every world tilt and strict hold streak, the real30-second
clock and the8-second handoff. They verify all1100 learned commands per case.
All32 fresh state/action tapes also cold-replay exactly with substep rail and
warning guards. The complete first closed loop is reexecuted independently.

The original plant remains the unrestricted XY cart and passive ball joint,
with50 Hz control,200 Hz float64 MuJoCo3.3.5 RK4, float32 delivery inside an80 N
disk,3 m square boundary and the original five-second upright requirement.
All five physical velocities remain. The unchanged physical Python runtime is
bound to its previous294-test source fingerprint; new policy evidence is the
explicit inference, capture, continuous-task and audit work above.

## Source, video and accounting

The additive publication package is
`2d/research/puffer_hybrid_1_link_20261017`. It contains an isolated runtime,
the frozen route and weights, strict metrics and a reproduction entry point.
The staged package reproduces the first fresh case byte for byte using only
its own copied source. Deployment needs no GPU or installed PufferLib.
Exact cross-platform BLAS behavior still needs a platform-specific check.

The full first fresh video has1,501 frames at50 fps and shows30 simulated
seconds, with persistent classical/learned role labels and the8-second handoff.
Decoded frames0,400 and1500 were inspected: the full rod/cart and labels are
visible. Video SHA256 is
`886707b348cff5663a0bb05a51bf4e5b7c9afb9df109059f6e68191471f991d4`.

New native costs in this stage:

| Work | Native integrations |
|---|---:|
| CPU/GPU frozen-observation forward audit | 0 |
| Paired CPU upright tests | 312,000 |
| Paired reference capture tests | 182,984 |
| Hanging hybrid pilot | 168,000 |
| Hanging development | 552,000 |
| Independent development audit | 6,000 |
| Fresh qualification | 396,000 |
| Independent fresh audit | 192,000 |
| Isolated publication-source reproduction | 12,000 |
| Rendering | 0 |
| Total | 1,820,984 |

The preceding balance learning study cost44,762,352 integrations; it is not
repeated or double-counted here. Full source, positive and negative tests,
training snapshots and current qualification evidence are retained in the
hash-verified ZIP. New tag: `2d-cartpole-1-link-puffer-hybrid-v1`.
The overall goal remains active: extend learned capture to more links and teach
earlier swing states, without weakening the original full-task gates.

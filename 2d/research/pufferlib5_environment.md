# Native spatial cartpole for PufferLib 5

Target pinned on 2026-10-17: PufferLib 5.0 commit
`ea7f989371270d2948f96728bda8aec5e7634a4c`. Version 4 was a suggestion, not a
constraint; the user explicitly accepted 5 and proposed an environment designed
for it. [Official docs](https://puffer.ai/docs.html).

Design: a native C/C++ MuJoCo environment using PufferLib's `puf_init`,
`puf_reset`, `puf_step`, `puf_log`, `puf_close` and `Agent` buffer interface.
Use one shared immutable model and separate MjData per instance; no Python calls,
allocation, network rendering or model construction inside a normal step.
PufferLib handles batched policy evaluation, pinned contiguous buffers and native
environment threading. [Pinned interface](https://github.com/PufferAI/PufferLib/blob/ea7f989371270d2948f96728bda8aec5e7634a4c/src/pufferenv.h).

Two continuous action heads represent XY forces. Preserve all passive ball-joint
motion, the original model XML, 80 N radial force delivery/float32 commands,
four 5 ms RK4 integrations, per-substep rail/warning checks, 30-second episodes
and 5-second world-tilt hold success. Canonical XML is exported from our Python
model generator before building. Native training randomness has its own stream;
authoritative evaluation injects original Python-generated initial states.
Training resets begin hanging and moving immediately, without preparation.

Observations: normalized full qpos/qvel, previous delivered XY force and elapsed
episode fraction. Keep quaternions and every physical velocity so twist/spin is
not silently discarded. Use float32 observation buffers and float64 physics.
Compile once per link count for fixed observation dimensions, starting at 1,
then 2 and beyond. Training reward may normalize effort, but qualification always
uses the original physical gate, never cumulative reward.

The local RTX 2060 SUPER is compute capability 7.5, 8 GB. Unlike a blanket GPU
incompatibility claim, **5.0 still has a real float32 build path**: `--float`
selects float weights/GEMMs in both build script and trainer. This is the first
path to smoke-test on this GPU; BF16 Tensor Core acceleration is unavailable.
No Torch fallback or CPU training is promised for 5.0. Its `--cpu` is evaluation.
[Pinned build](https://github.com/PufferAI/PufferLib/blob/ea7f989371270d2948f96728bda8aec5e7634a4c/build.sh),
[trainer](https://github.com/PufferAI/PufferLib/blob/ea7f989371270d2948f96728bda8aec5e7634a4c/src/pufferl.cu),
[NVIDIA feature table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

The first implementation uses CPU MuJoCo with CUDA policy training. This removes
Python stepping overhead but does not make the physics GPU-native. A GPU dynamics
implementation is a later separate experiment requiring dynamics, integration,
force, free-joint and outcome parity before claiming the original benchmark.
The website's up-to-60M training-steps/second figure is not our expected throughput.
Measure this plant's wall time, memory, control transitions and actual RK4 steps.

Local prerequisites: native Windows MSVC 2019 and MuJoCo C headers/DLL exist, so
the core and PufferLib adapter can be compiled and checked here. The complete
PufferLib CUDA trainer has Linux/POSIX dependencies. WSL commands exist but no
Linux distribution is installed; full training remains untested until an
isolated Linux/WSL2 build is available. No OS installation or driver change has
been made as part of research.

Prototype acceptance: compare native and authoritative full state, delivered
forces, world tilts, substep margins, warning flags, hold counters and terminal
events; exercise 1, 2 and 8 links, both transverse axes, asymmetric starts,
clipping and complete published 1/2-link trajectories where available. Test
PufferLib auto-reset/buffer handling separately from non-resetting evaluation.
Only then measure serial versus batched native stepping and start bounded
small-policy training. Keep all current release records and reserved seeds.

## Implemented and checked

The native core and actual PufferLib adapter now compile with MSVC against
MuJoCo 3.3.5 and the pinned upstream PufferLib headers. The completed check is
`output/pufferlib5_native_env_001`, charging **160,344** native integrations.
All 42 injected asymmetric transition cases (14 each at 1/2/8 links) have exact
state and delivered-command bytes. Full saved qualified 1/2-link episodes also
match every state/command byte. World tilts, clipping, gates, observation buffers,
terminal clearing, successful-horizon/rail/nonfinite auto-reset and log counters
pass. Serial and four-thread native outcomes match the original Python loop at
all four tested batch sizes. No reserved seeds were consumed.

Eight-link matched CPU stepping, 32 control commands per environment per lane:

| Environments | Original Python controls/s | Native serial controls/s | Native 4-thread controls/s |
|---:|---:|---:|---:|
| 1 | 2,052 | 6,877 | 6,497 |
| 16 | 2,343 | 6,552 | 22,425 |
| 64 | 2,241 | 6,316 | 20,032 |
| 256 | 2,098 | 6,305 | 16,881 |

These are short CPU microbenchmarks, with setup/rendering excluded and no GPU
policy or PPO update. Each control transition contains four physics integrations.
At 64 environments native four-thread stepping is about **8.94x** faster than
the original Python loop for this workload; end-to-end learning benefit remains
unmeasured. Larger batches were slower here, supporting measurement before scale.

An isolated sparse PufferLib checkout is prepared at
`output/pufferlib5_training_003`, pinned to the exact target. Its additive
`build_spatial.sh`, one-link environment/model and initial 64-environment config
are generated by `scripts/prepare_puffer_environment.py`. The stock build script
is unchanged. It is ready for the Linux/WSL toolchain smoke test, which has not
run. This is environment progress, not a trained one-link policy or new record.

## Evaluation capture and initial run preparation

The latest native evaluation-interface run is
`output/pufferlib5_evaluation_io_003`: all 12 fixed diagnostic cases at 1/2/8
links pass, with 20,088 actual native integrations and 20,088 independent cold
replay integrations. Recorded observations/commands/states match exactly;
world tilts, gates, nonfinite rejection, terminal clearing and EOF match too.
Native shutdown accounting agrees with the captured totals at every count.
Synthetic upright cases are interface tests and do not establish swing-up.

The complete 174-frame eight-link hanging diagnostic was rendered and decoded
at 50 fps, with first/middle/final images inspected. It lasts 3.46 simulated
seconds and ends at its actual rail failure. The video says DIAGNOSTIC / NOT A
TRAINED POLICY / NOT A QUALIFIED RESULT. It verifies the export path only.

Earlier capture attempt 001 accidentally imported an unguarded standalone
verifier and ran its old 160,344-integration parity suite, then failed before
the new IO checks. This exceeded its registered budget; it remains charged as
a harness failure. Import is now guarded. Capture run 002 then passed with
40,176 integrations; run 003 added shutdown-cost verification with another
40,176. Thus capture-stage work since the published prototype totals 240,696
native integrations, including the failed harness and independent replay.

The pinned trainer source audit also found a continuous-action memory-layout
issue: `Allocator.total_elems` counts logical elements while allocation inserts
16-byte gaps. Flat weight/gradient/actor lengths, momentum allocation and
checkpoint saving use the shorter logical count, and Muon walks tensor slices
without those gaps. For one link, H64/L2 and two continuous actions, float32
storage needs 102,672 bytes but the original flat length is 102,664 bytes;
the final two recurrent weights are omitted and both recurrent optimizer
slices are shifted. This affects the new Puffer trainer path.
[Pinned allocator and flat views](https://github.com/PufferAI/PufferLib/blob/ea7f989371270d2948f96728bda8aec5e7634a4c/src/pufferl.cu),
[pinned optimizer](https://github.com/PufferAI/PufferLib/blob/ea7f989371270d2948f96728bda8aec5e7634a4c/src/algo.cu).

The additive preparation driver emits corrected copies under `spatial_patches/`
and directs the new build wrapper to them. It uses physical spans for flat
precision slabs and aligns the optimizer walk; upstream tracked files stay
unchanged. Host-compiled original allocation/registration functions reproduce
the issue and verify the correction across 24 cases (1/2/8 links, 4/2-byte
elements, continuous XY and discrete controls). The latest evidence is
`output/pufferlib5_weight_layout_003`; no CUDA or physics was run. Initial
attempt 001 failed on Windows text decoding; 002/003 passed. CPU checkpoint
reader alignment remains unverified, so evaluation will capture actual GPU
policy execution instead.

Training_003 and evaluation_001 are separately prepared one-link checkouts,
respectively with capture compiled out and in. Their integration hashes and
generated shell syntax pass. The bounded launcher's two dry-run plans freeze
4,096 and 999,424 control transitions (16,384 and 3,997,696 native integrations),
seed 73, 64 environments, horizon 64 and no automatic evaluation. Protocol
checks reject partial batches, excess budget, a new unregistered seed, an old
unpatched checkout, Windows execution and modified native input. The corrected
protocol check passed all eight cases; its first harness attempt had an invalid
argument-count assertion and failed with zero native cost.

WSL import remains blocked by firmware virtualization being disabled; the user
will enable ASUS SVM/Windows features and reboot later. No distribution has been
registered and no training has run. After reboot: Linux float32 build, bounded
one-batch GPU smoke test, then initial one-link learning and recorded development
evaluation before moving to two links. Reserved qualification starts remain
untouched. GPU execution, checkpoint quality and end-to-end speed are unproven.

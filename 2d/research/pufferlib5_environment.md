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
`output/pufferlib5_training_001`, pinned to the exact target. Its additive
`build_spatial.sh`, one-link environment/model and initial 64-environment config
are generated by `scripts/prepare_puffer_environment.py`. The stock build script
is unchanged. It is ready for the Linux/WSL toolchain smoke test, which has not
run. This is environment progress, not a trained one-link policy or new record.

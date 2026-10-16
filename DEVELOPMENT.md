# Developing the 1D and 2D cart-pole tracks

[← Project overview](README.md) · [1D results & reproduction](docs/one-dimensional-guide.md) ·
[2D setup & method](2d/README.md)

This repository contains two independently evaluated control benchmarks. Keep
changes, experiments and result claims explicit about which track they affect.

## Choose a track

| | 1D rail / planar chain | 2D plane / spatial chain |
| :--- | :--- | :--- |
| Source | `src/gcartpole/`, root `scripts/` and `configs/` | `2d/src/spatialpole/` |
| Tests | `tests/` | `2d/tests/` |
| Published evidence | `runs/`, `docs/`, `assets/` | `2d/evidence/`, `2d/videos/` |
| Completion contract | [ROADMAP.md](ROADMAP.md) | [Research plan](2d/research/plan.md) |
| Result navigation | [1D guide](docs/one-dimensional-guide.md) | [2D README](2d/README.md), [progress](2d/PROGRESS.md) |

The dimensions describe cart travel. The 1D chain uses planar hinge joints;
the 2D chain has unrestricted spatial ball joints. Force limits, orientation
representations, reset distributions and final cohorts must be interpreted
under the relevant contract.

## Work without interrupting another agent

Use an isolated checkout or worktree for each task. The research checkout may
contain unfinished experiments and private evidence; never reset, stash or
commit that entire tree to prepare a publication.

For a clean documentation or code task, from an existing checkout:

```powershell
git fetch origin
git worktree add -b codex/my-task ../cartpole-my-task origin/master
```

- Keep commits limited to the files owned by the task. Inspect the staged diff.
- Work inside the relevant track. Coordinate changes to the top-level overview,
  shared files and publication metadata when another agent is publishing.
- Fetch again before integrating. Preserve concurrent commits and resolve
  documentation conflicts against the latest evidence.
- Use normal pushes. Never force-push the shared default branch.
- Keep generated outputs and private research out of public commits unless
  deliberately packaged as reviewed evidence.

`master` is the shared integration branch. Use short-lived `codex/` branches;
merge completed work, then delete only branches whose commits are fully
preserved on `master` and which are no longer active. Keep release tags and
frozen source bundles. A local worktree or private research checkout may need
to remain even after its corresponding public release branch is merged.

## Environments and checks

### 1D

Use the environment recorded with the release or experiment you are working
on. The root has several toolchains: historical Mac dependencies in
[requirements-mac.txt](requirements-mac.txt), Crocoddyl/BoxFDDP dependencies in
[requirements-fddp.txt](requirements-fddp.txt), and a separate
[Aligator environment](environment-aligator.yml). Learning integrations have
their own requirement files. One installation command does not cover every
historical experiment.

For release reproduction, use its frozen source bundle and command file;
start with the [fourteen-link guide](docs/one-dimensional-guide.md#reproduce-the-fourteen-link-release).
For a code change, run the affected tests in its matching environment and
replay the relevant frozen reference when numerical behavior could change.

### 2D

Use a separate Python 3.11 environment inside `2d/`, install
`requirements-lock.txt`, then install that package editable. The
[2D setup commands](2d/README.md#run-on-windows) include the test suite,
frozen-controller evaluation and video generation. Set BLAS/OpenMP to one
thread as recorded. Use new output directories so failures remain available.

For changes to dynamics, quaternion coordinates, force delivery, scoring or
feedback, include the relevant interface, derivative, replay and full-episode
checks. Passing near-upright balance tests alone does not validate swing-up.

### Documentation

Check relative links, image paths, heading anchors, result claims and rendered
Markdown. Match counts and metrics to frozen manifests and appendices. Avoid
editing frozen evidence merely to update navigation or presentation.

## Record experiments before running them

Record the question, hypothesis, changed and held parameters, initialization,
seeds, compute ceiling and claim scope. Freeze source, inputs, exact commands
and runtime. Keep optimizer histories, complete physical state/action tapes,
native simulator counts, independent replay and failures where applicable.

Separate development seeds from final qualification. Once a controller and
source are frozen, evaluate fresh reserved cohorts without retuning. Record
interruptions and null results as well as successes. Never overwrite a failed
run with a corrected one or present a retrospective snapshot as preregistered.

For the 1D campaign, consult the [experiment protocol](docs/experiment_protocol.md),
[evidence runbook](docs/evidence_runbook.md), and the applicable local research
recording instructions. For 2D, follow its [research plan](2d/research/plan.md)
and keep [PROGRESS.md](2d/PROGRESS.md) current.

## Publish a qualified result

1. Satisfy the track's complete hanging-start, boundary, full-episode and
   continuous-hold gates. Preserve the distinction between balance-only,
   partial nominal results and qualified noisy-start results.
2. Freeze controller, configuration and source identities. Include the exact
   runtime, commands, seed cohorts, metrics, complete tapes and replay checks.
3. Produce a full held-out episode video from recorded physical states. A
   preview GIF is navigation; retain the uninterrupted full-speed video.
4. Package reviewed public evidence from an isolated checkout. Inspect the
   file list for private data, unrelated experiments and accidental changes
   to previous releases.
5. Update the relevant track's documentation and the top-level status/preview.
   Publish the corresponding GitHub release, full video and downloadable
   evidence. Existing 1D releases use count-specific canonical tags; 2D uses
   `2d-cartpole-N-link-v1` tags.
6. Fetch and integrate concurrent changes before pushing. Verify the published
   files, tag, release assets and links. Remove completed merged branch heads
   only after their history is preserved.

Frozen numerical evidence and source provenance remain immutable. Correct an
error with an explicit amendment or new result rather than changing historical
hashes to match new bytes.

## Historical research notes

Detailed papers and development chronologies remain under [docs/](docs), and
frozen source bundles remain with their releases. Earlier versions of this
development guide are available in
[Git history](https://github.com/corbensorenson/cartpole/blob/3693d3094a88ec128bae09f1547b11fc75ffb26a/DEVELOPMENT.md).
Their active-count statements describe the frontier at that time; use the
[project overview](README.md) for current published status.

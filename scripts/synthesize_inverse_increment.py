#!/usr/bin/env python
"""One reproducible adjacent-count synthesis screen; canonical release gates remain separate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from gcartpole.config import dump_json, load_config, save_config
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp


def exact_candidate_passed(payload, cfg):
    result = payload.get("result", {})
    diagnostics = payload.get("controller", {}).get("final_trajectory_diagnostics", {})
    hold = result.get("max_upright_streak_seconds", -np.inf)
    excursion = result.get("max_cart_excursion", np.inf)
    gap = diagnostics.get("maximum_dynamics_defect", np.inf)
    initial_gap = diagnostics.get("initial_state_gap", np.inf)
    expected_steps = round(cfg["env"]["episode_seconds"] / (cfg["env"]["timestep"]*cfg["env"]["frame_skip"]))
    return bool(result.get("success") and result.get("trajectory_integrity")
                and result.get("termination_reason") == "time_limit"
                and result.get("length") == expected_steps
                and np.all(np.isfinite([hold, excursion, gap, initial_gap]))
                and hold >= cfg["env"]["success_sustain_seconds"]
                and excursion <= cfg["env"]["rail_limit"]
                and diagnostics.get("is_feasible") and max(gap, initial_gap) <= 1e-8)


def validate_increment(source_spline, source_controller, cfg):
    source_count = len(source_spline["search"]["spline_control_points"][0])-1
    controller_count = len(source_controller["selected_state"]["qpos"])-1
    if controller_count != source_count or cfg["env"]["n_links"] != source_count+1:
        raise ValueError("target must increment the same validated source count by one")
    if not exact_candidate_passed(source_controller, source_spline["effective_config"]):
        raise ValueError("source requires feasible dynamics and a successful full exact-start calibration replay")


def fddp_command(python, config, source, output, iterations, seed, *,
                 defer_handoff=True, handoff_lyapunov=1800,
                 terminal_weight=1e-14, initial_regularization=1e-6,
                 rebuild_feedback=False, lqr_decimal_digits=0):
    command = [python, "scripts/search_fddp_capture.py", "--config", str(config),
            "--state-json", str(source), "--state-index", "selected", "--initial-controller", str(source),
            "--iterations", str(iterations), "--solver-verbose", "--continuous-angles",
            "--tracking-gain-scale", "1", "--lqr-scale", "1", "--lqr-control-cost", "1000",
            "--terminal-weight", str(terminal_weight), "--terminal-state-weight", "1000",
            "--initial-regularization", str(initial_regularization),
            "--lqr-decimal-digits", str(lqr_decimal_digits),
            "--terminal-angle-factor", "20", "--terminal-hinge-velocity-factor", "4",
            "--stage-weight", ".01", "--control-cost", ".01",
            "--rail-soft-limit", "2.85", "--rail-weight", "5000000",
            "--handoff-lyapunov", str(handoff_lyapunov), "--handoff-cart-abs", "1.5",
            "--handoff-angle-abs", ".15", "--handoff-cart-velocity-abs", ".5",
            "--handoff-hinge-velocity-rms", ".75",
            "--seed", str(seed), "--out", str(output)]
    if defer_handoff:
        command.append("--defer-handoff-until-horizon")
    if rebuild_feedback:
        command += ["--rebuild-initial-feedback", "--initial-feasible"]
    return command


def restoration_command(python, config, source, output, evaluations, lsmr_iterations=300):
    return [python, "scripts/restore_frontier_trajectory.py", "--config", str(config),
            "--controller", str(source), "--max-evaluations", str(evaluations),
            "--lsmr-max-iterations", str(lsmr_iterations),
            "--defect-weight", "100000", "--terminal-weight", "1000",
            "--reference-weight", "1e-8", "--control-weight", "1e-8", "--out", str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-spline", required=True)
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--horizon-seconds", type=float,
                        help="Optional swing duration; retime the transferred spline before discovery.")
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--inverse-evaluations", type=int, default=100)
    parser.add_argument("--fddp-iterations", type=int, default=100)
    parser.add_argument("--restoration-evaluations", type=int, default=30)
    parser.add_argument("--restoration-lsmr-iterations", type=int, default=300)
    parser.add_argument("--pre-restore", action="store_true",
                        help="Repair the inverse trajectory before FDDP; an experimental recipe variant.")
    parser.add_argument("--state-gated-handoff", action="store_true",
                        help="Permit capture when the state gates pass, before the nominal horizon ends.")
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.)
    parser.add_argument("--no-handoff-recovery", action="store_true",
                        help="Disable the cheap earlier-handoff replay after a feasible route fails full execution.")
    parser.add_argument("--handoff-threshold-grid", nargs="+", type=float,
                        default=[5., 10., 25., 100., 1800.],
                        help="Bounded development-only capture thresholds tried after a feasible candidate fails.")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--terminal-value-coefficients", nargs="*", type=float,
                        default=[.01, .01, 1.],
                        help="Bounded recovery refinements on P after earlier recipes fail; pass no values to disable.")
    parser.add_argument("--lqr-decimal-digits", type=int, default=0,
                        help="Zero retains the standard design; >=30 opts into precision-checked gain/value design.")
    args = parser.parse_args()
    if min(args.inverse_evaluations, args.fddp_iterations, args.restoration_evaluations,
           args.restoration_lsmr_iterations) < 1:
        raise ValueError("positive bounded stage budgets required")
    if not np.isfinite(args.handoff_lyapunov) or args.handoff_lyapunov <= 0:
        raise ValueError("positive finite capture threshold required")
    if any(not np.isfinite(value) or value <= 0 for value in args.handoff_threshold_grid):
        raise ValueError("capture threshold grid requires positive finite values")
    if any(not np.isfinite(value) or value <= 0 for value in args.terminal_value_coefficients):
        raise ValueError("terminal value coefficients must be finite and positive")
    if args.lqr_decimal_digits != 0 and args.lqr_decimal_digits < 30:
        raise ValueError("precision design requires zero or at least thirty digits")
    cfg = load_config(args.config)
    source_spline = json.loads(Path(args.source_spline).read_text())
    source_controller = json.loads(Path(args.source_controller).read_text())
    validate_increment(source_spline, source_controller, cfg)
    horizon = (float(source_spline["controller"]["horizon_seconds"])
               if args.horizon_seconds is None else args.horizon_seconds)
    dt = cfg["env"]["timestep"] * cfg["env"]["frame_skip"]
    if (not np.isfinite(horizon) or not 0 < horizon <= 25.
            or not np.isclose(horizon/dt, round(horizon/dt), rtol=0., atol=1e-10)):
        raise ValueError("swing duration must be positive, at most 25 seconds and aligned to control ticks")
    directory = Path(args.out_directory)
    directory.mkdir(parents=True, exist_ok=False)
    target_config = directory/"target.yaml"
    save_config(cfg, target_config)
    journal = directory/"pipeline.json"
    record = dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(), status="running",
                  n_links=cfg["env"]["n_links"], config=file_metadata(args.config),
                  source_spline=file_metadata(args.source_spline), source_controller=file_metadata(args.source_controller),
                  runtime=runtime_metadata(), parameters=vars(args), stages=[],
                  note="Adjacent-count discovery with exact fixed hanging-start replay only. No noisy held-out gate, release promotion, or automated advancement beyond this count.")
    dump_json(record, journal)
    python = sys.executable
    def refine_command(*arguments, **options):
        return fddp_command(python, *arguments, lqr_decimal_digits=args.lqr_decimal_digits, **options)

    def run_stage(name, command, inputs):
        stage = directory/name
        wrapper = [python, "scripts/run_frontier_experiment.py", "--directory", str(stage)]
        for path in inputs:
            wrapper += ["--input", str(path)]
        wrapper += ["--", *command]
        print(json.dumps(dict(stage=name, state="starting", command=command)), flush=True)
        subprocess.run(wrapper, check=True)
        result_path = stage/"result.json"
        record["stages"].append(dict(name=name, execution=file_metadata(stage/"execution.json"),
                                     result=file_metadata(result_path)))
        dump_json(record, journal)
        return result_path

    try:
        transferred = run_stage("transfer", [python, "scripts/transfer_inverse_spline.py",
            "--config", str(target_config), "--source", args.source_spline,
            "--seconds", str(horizon),
            "--out", str(directory/"transfer/result.json")], [target_config, args.source_spline])
        inverse = run_stage("inverse", [python, "scripts/search_inverse_spline.py", "--config", str(target_config),
            "--seconds", str(horizon), "--initial-spline", str(transferred),
            "--sample-dt", ".02", "--max-evaluations", str(args.inverse_evaluations),
            "--dynamics-residual", "acceleration", "--reference-weight", "1e-12", "--effort-weight", "1e-12",
            "--out", str(directory/"inverse/result.json")], [target_config, transferred])
        warm_start = inverse
        if args.pre_restore:
            warm_start = run_stage("pre_restoration", restoration_command(python, target_config, inverse,
                directory/"pre_restoration/result.json", args.restoration_evaluations,
                args.restoration_lsmr_iterations), [target_config, inverse])
        candidate = run_stage("fddp", refine_command(target_config, warm_start,
            directory/"fddp/result.json", args.fddp_iterations, args.seed,
            defer_handoff=not args.state_gated_handoff,
            handoff_lyapunov=args.handoff_lyapunov), [target_config, warm_start])
        data = json.loads(candidate.read_text())
        if not exact_candidate_passed(data, cfg) and not data["controller"]["final_trajectory_diagnostics"]["is_feasible"]:
            restored = run_stage("restoration", restoration_command(python, target_config, candidate,
                directory/"restoration/result.json", args.restoration_evaluations,
                args.restoration_lsmr_iterations), [target_config, candidate])
            candidate = run_stage("restored_fddp", refine_command(target_config, restored,
                directory/"restored_fddp/result.json", args.fddp_iterations, args.seed,
                defer_handoff=not args.state_gated_handoff,
                handoff_lyapunov=args.handoff_lyapunov), [target_config, restored])
            data = json.loads(candidate.read_text())
        if (not exact_candidate_passed(data, cfg)
                and data["controller"]["final_trajectory_diagnostics"]["is_feasible"]
                and not args.state_gated_handoff and not args.no_handoff_recovery):
            candidate = run_stage("state_capture_replay", refine_command(target_config, candidate,
                directory/"state_capture_replay/result.json", args.fddp_iterations, args.seed,
                defer_handoff=False, handoff_lyapunov=args.handoff_lyapunov)+["--replay-only"],
                [target_config, candidate])
            data = json.loads(candidate.read_text())
        if (not exact_candidate_passed(data, cfg)
                and data["controller"]["final_trajectory_diagnostics"]["is_feasible"]
                and not args.no_handoff_recovery):
            # Keep controls, gains, nominal reference, and optimizer objective
            # frozen. Only the executed gate changes during these cheap replays.
            reference = candidate
            unlatched_minimum = (
                float(data["result"].get("minimum_lyapunov", float("nan")))
                if data["result"].get("first_handoff_step") is None else float("nan")
            )
            for index, threshold in enumerate(dict.fromkeys(args.handoff_threshold_grid)):
                if threshold == args.handoff_lyapunov:
                    continue
                if np.isfinite(unlatched_minimum) and threshold < unlatched_minimum:
                    record.setdefault("capture_recovery_skips", []).append(dict(
                        threshold=threshold, minimum_lyapunov=unlatched_minimum,
                        reason="Gate cannot open anywhere in the unchanged unlatched replay."))
                    dump_json(record, journal)
                    continue
                name = f"state_capture_grid_{index:02d}_replay"
                candidate = run_stage(name, refine_command(target_config, reference,
                    directory/name/"result.json", args.fddp_iterations, args.seed,
                    defer_handoff=False, handoff_lyapunov=args.handoff_lyapunov)
                    + ["--switch-lyapunov", str(threshold), "--replay-only"],
                    [target_config, reference])
                data = json.loads(candidate.read_text())
                if exact_candidate_passed(data, cfg):
                    break
        if not exact_candidate_passed(data, cfg):
            # Restart from the useful repaired inverse curve, not a feasible
            # route that already sacrificed capture. This bounded schedule
            # records the recovery discovered at eleven as count-agnostic
            # policy; it does not imply independent re-synthesis has passed.
            reference = warm_start
            for index, coefficient in enumerate(args.terminal_value_coefficients):
                name = f"capture_value_refinement_{index:02d}"
                candidate = run_stage(name, refine_command(target_config, reference,
                    directory/name/"result.json", args.fddp_iterations, args.seed,
                    defer_handoff=True, handoff_lyapunov=args.handoff_lyapunov,
                    terminal_weight=coefficient*args.handoff_lyapunov,
                    initial_regularization=1e-6 if index == 0 else 10.,
                    rebuild_feedback=index >= 2), [target_config, reference])
                data = json.loads(candidate.read_text())
                reference = candidate
                if exact_candidate_passed(data, cfg):
                    break
        passed = exact_candidate_passed(data, cfg)
        record.update(status="candidate_passed_exact_start" if passed else "candidate_failed_exact_start",
                      candidate=file_metadata(candidate), transfer_spline=file_metadata(inverse),
                      exact_start_passed=passed,
                      result={k:v for k,v in data["result"].items() if k not in ("trajectory", "final_info")},
                      final_diagnostics=data["controller"]["final_trajectory_diagnostics"])
    except BaseException as error:
        record.update(status="execution_failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at"] = utc_timestamp()
        dump_json(record, journal)
    print(json.dumps(dict(n_links=cfg["env"]["n_links"], status=record["status"], candidate=record["candidate"])), flush=True)


if __name__ == "__main__":
    main()

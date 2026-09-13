from __future__ import annotations

from pathlib import Path

from scripts.solve_generalized_morphology import (
    PipelinePaths,
    command_steps,
    parser,
    stage_failure_status,
    validate_args,
)

ROOT = Path(__file__).resolve().parents[1]


def pipeline_args(extra: list[str] | None = None):
    return parser().parse_args(
        [
            "--source-config",
            str(ROOT / "configs/swingup7_uniform.yaml"),
            "--source-controller",
            str(ROOT / "runs/generalized_solver/n2_route.json"),
            "--target-config",
            str(ROOT / "configs/generalized_n2_unequal.yaml"),
            *(extra or []),
        ]
    )


def test_pipeline_infers_controller_count_instead_of_base_config_count() -> None:
    args = pipeline_args()
    source, target, source_links = validate_args(args)
    assert source_links == 2
    assert source.n_links == 2
    assert target.n_links == 2
    assert source.lengths.tolist() == [1.5, 1.5]
    assert target.lengths.tolist() == [1.2, 1.8]


def test_pipeline_builds_complete_target_parameterized_stage_sequence(
    tmp_path: Path,
) -> None:
    args = pipeline_args(["--require-transfer-failure"])
    _source, target, source_links = validate_args(args)
    paths = PipelinePaths(tmp_path, "morphology")
    steps = command_steps(
        args,
        paths,
        source_links=source_links,
        target_links=target.n_links,
        rail_half_length=target.rail_half_length,
    )
    assert [label for label, _command, _output in steps] == [
        "transfer",
        "mirror_transfer",
        "transfer_check",
        "make_fddp_warm",
        "optimize",
        "package",
        "package_mirror",
        "gate",
        "verify",
    ]
    by_label = {label: command for label, command, _output in steps}
    optimize = by_label["optimize"]
    assert optimize[optimize.index("--rail-soft-limit") + 1] == "4.0"
    assert str(paths.optimizer) in optimize
    assert str(paths.mirror) in by_label["package_mirror"]
    assert "--mirror" in by_label["package_mirror"]
    assert "--require-warm-failure" in by_label["verify"]


def test_pipeline_stops_on_unlatched_optimizer_and_failed_gate(tmp_path: Path) -> None:
    optimizer = tmp_path / "optimizer.json"
    optimizer.write_text(
        '{"result":{"success":true,"latched":false}}', encoding="utf-8"
    )
    gate = tmp_path / "gate.json"
    gate.write_text('{"success_rate":0.95}', encoding="utf-8")
    assert stage_failure_status("optimize", optimizer) == "exact_refinement_failed"
    assert stage_failure_status("gate", gate) == "noisy_gate_failed"


def test_pipeline_accepts_only_complete_promotion_stages(tmp_path: Path) -> None:
    optimizer = tmp_path / "optimizer.json"
    optimizer.write_text(
        '{"result":{"success":true,"latched":true}}', encoding="utf-8"
    )
    gate = tmp_path / "gate.json"
    gate.write_text('{"success_rate":1.0}', encoding="utf-8")
    assert stage_failure_status("optimize", optimizer) is None
    assert stage_failure_status("gate", gate) is None

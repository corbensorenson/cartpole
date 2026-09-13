from __future__ import annotations

import json

import numpy as np

from scripts.refine_swingup_endpoint_gauss_newton import (
    blend_controls,
    interpolation_matrix,
    load_controls,
)


def test_interpolation_matrix_preserves_constant_and_endpoints() -> None:
    matrix = interpolation_matrix(knot_count=5, step_count=17)
    assert matrix.shape == (17, 5)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(matrix[0], [1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(matrix[-1], [0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix @ np.ones(5), np.ones(17))


def test_load_controls_accepts_refiner_search_horizon(tmp_path) -> None:
    artifact = tmp_path / "endpoint.json"
    artifact.write_text(
        json.dumps(
            {
                "search": {"horizon_seconds": 0.06},
                "best": {"controls": [-2.0, 0.25, 2.0]},
            }
        ),
        encoding="utf-8",
    )
    controls, seconds = load_controls(artifact, "best")
    np.testing.assert_allclose(controls, [-1.0, 0.25, 1.0])
    assert seconds == 0.06


def test_blend_controls_uses_requested_horizon() -> None:
    blended = blend_controls(
        np.array([0.0, 0.2, 0.4, 0.6]),
        np.array([1.0, 0.8, 0.6, 0.4]),
        alpha=0.25,
        step_count=3,
    )
    np.testing.assert_allclose(blended, [0.25, 0.35, 0.45])

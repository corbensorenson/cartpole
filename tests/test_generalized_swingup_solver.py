import numpy as np

from gcartpole.config import load_config
from gcartpole.generalized_solver import setup_from_config
from scripts.generalized_swingup_solver import (
    coordinate_transform,
    split_continuation_config,
    split_warm_start,
    uniform_config,
)


def test_split_command_builders_make_valid_count_continuation_and_invariant_gain():
    base = load_config("configs/swingup7_uniform.yaml")
    source_cfg = uniform_config(base, 2)
    target_cfg = uniform_config(base, 3)
    continuation, embedding = split_continuation_config(source_cfg, target_cfg)

    source = setup_from_config(source_cfg)
    start = setup_from_config(continuation, progress=0.0)
    end = setup_from_config(continuation, progress=1.0)
    np.testing.assert_array_equal(embedding["split_counts"], [1, 2])
    np.testing.assert_array_equal(embedding["source_joint_locks"], [0.0, 0.0, 1.0])
    assert embedding["compatibility"]["global_dynamic_similarity"]
    assert start.n_links == end.n_links == 3
    np.testing.assert_allclose(start.lengths, [1.5, 0.75, 0.75])
    np.testing.assert_allclose(end.lengths, [1.0, 1.0, 1.0])

    steps = 4
    source_dim = 2 * (source.n_links + 1)
    source_transform = coordinate_transform(source.n_links, None)
    target_transform = coordinate_transform(start.n_links, None)
    payload = {
        "controller": {
            "controls": np.linspace(-0.2, 0.2, steps).tolist(),
            "feedback_gains": np.arange(steps * source_dim, dtype=float)
            .reshape(steps, source_dim)
            .tolist(),
        },
        "search": {
            "nominal_coordinate_states": np.arange(
                (steps + 1) * source_dim, dtype=float
            )
            .reshape(steps + 1, source_dim)
            .tolist(),
        },
    }
    warm = split_warm_start(
        payload,
        source,
        start,
        embedding,
        source_transform,
        target_transform,
    )
    assert warm["embedding"]["feedback_invariance_max_abs_error"] < 1.0e-12
    assert len(warm["controller"]["feedback_gains"][0]) == 8
    assert len(warm["search"]["nominal_coordinate_states"][0]) == 8
    assert warm["selected_state"]["qpos"][3] == 0.0

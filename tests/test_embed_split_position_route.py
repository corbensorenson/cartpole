import numpy as np

from scripts.embed_split_position_route import build_embedding


def test_interior_split_assignments_duplicate_the_selected_source_link():
    np.testing.assert_array_equal(
        build_embedding(10, 5),
        np.array([0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9]),
    )


def test_distal_split_assignments_keep_the_source_order():
    np.testing.assert_array_equal(
        build_embedding(10, 10),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
    )

from __future__ import annotations

from scripts.verify_generalized_uniform_ladder import verify_ladder


def test_published_uniform_ladder_passes_shared_architecture_contract() -> None:
    rows = verify_ladder(minimum_episodes=20)

    assert [row["n_links"] for row in rows] == list(range(1, 8))
    assert all(row["passed"] for row in rows)
    assert sum(row["successes"] for row in rows) == 140
    assert sum(row["prediction_execution_agreements"] for row in rows) == 140

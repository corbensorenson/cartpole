from __future__ import annotations

import numpy as np

from gcartpole.generalized_solver import (
    AdaptiveHomotopy,
    BoundedForceAdapter,
    PhysicalSetup,
    common_count_morphologies,
    dimensionless_setup,
    force_action_scale,
    homotopy_morphology,
    mirror_feedback_route,
    rail_requirement,
    resample_controls,
    split_absolute_coordinate_lift_matrix,
    split_embedding,
    split_joint_profile,
    split_state_lift_matrix,
    split_state_projection,
    state_transfer_matrix,
    transfer_state,
)


def setup(
    lengths=(0.5, 0.5),
    masses=(0.4, 0.6),
    *,
    cart_mass=1.0,
    rail=1.5,
    force=20.0,
    damping=(0.01, 0.02),
    cart_damping=0.03,
    armature=0.001,
    cart_half_length=0.18,
    radius=0.02,
    dt=0.005,
    frame_skip=4,
):
    return PhysicalSetup(
        lengths=np.asarray(lengths),
        masses=np.asarray(masses),
        cart_mass=cart_mass,
        rail_half_length=rail,
        force_limit=force,
        joint_damping=np.asarray(damping),
        cart_damping=cart_damping,
        joint_armature=armature,
        cart_half_length=cart_half_length,
        link_radius=radius,
        timestep=dt,
        frame_skip=frame_skip,
    )


def test_dimensionless_groups_are_invariant_under_dynamic_similarity():
    base = setup()
    length_scale = 3.0
    mass_scale = 2.5
    time_scale = np.sqrt(length_scale)
    scaled = setup(
        lengths=tuple(length_scale * base.lengths),
        masses=tuple(mass_scale * base.masses),
        cart_mass=mass_scale * base.cart_mass,
        rail=length_scale * base.rail_half_length,
        force=mass_scale * base.force_limit,
        damping=tuple(mass_scale * length_scale**2 / time_scale * base.joint_damping),
        cart_damping=mass_scale / time_scale * base.cart_damping,
        armature=mass_scale * length_scale**2 * base.joint_armature,
        cart_half_length=length_scale * base.cart_half_length,
        radius=length_scale * base.link_radius,
        dt=time_scale * base.timestep,
    )
    first = dimensionless_setup(base)
    second = dimensionless_setup(scaled)
    scalar_names = (
        "cart_to_link_mass",
        "force_authority",
        "rail_ratio",
        "usable_rail_ratio",
        "link_radius_ratio",
        "policy_dt_ratio",
        "cart_damping_ratio",
        "joint_armature_ratio",
    )
    for name in scalar_names:
        assert np.isclose(getattr(first, name), getattr(second, name))
    np.testing.assert_allclose(first.length_fractions, second.length_fractions)
    np.testing.assert_allclose(first.mass_fractions, second.mass_fractions)
    np.testing.assert_allclose(first.joint_damping_ratios, second.joint_damping_ratios)


def test_state_transfer_is_identity_for_same_setup():
    physical = setup()
    np.testing.assert_allclose(
        state_transfer_matrix(physical, physical),
        np.eye(2 * (physical.n_links + 1)),
    )


def test_state_transfer_preserves_rigid_chain_orientation_and_scaling():
    source = setup(lengths=(0.2, 0.3, 0.5), masses=(0.2, 0.3, 0.5), damping=(0, 0, 0))
    target = setup(lengths=(0.5, 0.5), masses=(0.5, 0.5), damping=(0, 0), rail=3.0)
    # Relative coordinates [angle, 0, 0] represent a straight chain.
    state = np.array([0.4, 0.7, 0.0, 0.0, 1.2, -0.3, 0.0, 0.0])
    transferred = transfer_state(state, source, target)
    expected_length_scale = target.chain_length / source.chain_length
    np.testing.assert_allclose(transferred[1:3], [0.7, 0.0], atol=1e-12)
    np.testing.assert_allclose(transferred[4:6], [-0.3, 0.0], atol=1e-12)
    assert np.isclose(transferred[0], expected_length_scale * state[0])
    assert np.isclose(transferred[3], np.sqrt(expected_length_scale) * state[4])


def test_force_and_time_resampling_preserve_dimensionless_command():
    source = setup(force=20.0)
    target = setup(force=40.0)
    assert np.isclose(force_action_scale(source, target), 0.5)
    controls = np.full(20, 0.8)
    transferred = resample_controls(controls, source, target)
    np.testing.assert_allclose(transferred, 0.4)
    assert transferred.size == controls.size


def test_rail_requirement_includes_cart_body_and_clearance():
    physical = setup(rail=1.5)
    result = rail_requirement(np.array([-0.8, 0.2, 0.7]), physical, clearance=0.1)
    assert np.isclose(result["required_rail_half_length"], 1.08)
    assert np.isclose(result["required_rail_ratio"], 1.08 / physical.chain_length)
    assert np.isclose(result["margin"], 0.42)


def test_bounded_force_adapter_learns_gain_and_compensates():
    adapter = BoundedForceAdapter(forgetting=1.0, covariance=100.0)
    jacobian = np.array([0.0, 2.0, -1.0])
    true_gain = 0.75
    true_bias = 0.08
    for command in np.tile(np.array([-0.8, -0.3, 0.2, 0.7]), 20):
        predicted = np.array([1.0, 2.0, 3.0])
        equivalent = true_gain * command + true_bias
        observed = predicted + jacobian * (equivalent - command)
        assert adapter.observe(command, predicted, observed, jacobian)
    assert np.isclose(adapter.gain, true_gain, atol=2e-3)
    assert np.isclose(adapter.bias, true_bias, atol=2e-3)
    desired = 0.4
    corrected = adapter.command(desired)
    assert np.isclose(true_gain * corrected + true_bias, desired, atol=2e-3)


def test_bounded_force_adapter_rejects_structural_and_saturated_updates():
    adapter = BoundedForceAdapter(
        forgetting=1.0,
        covariance=100.0,
        structural_fraction_limit=0.25,
    )
    predicted = np.zeros(3)
    jacobian = np.array([1.0, 0.0, 0.0])

    structural = adapter.observe_diagnostic(
        0.2,
        predicted,
        np.array([0.1, 1.0, 0.0]),
        jacobian,
    )
    assert not structural["updated"]
    assert structural["reason"] == "structural_mismatch"
    assert adapter.updates == 0

    saturated = adapter.observe_diagnostic(
        0.9,
        predicted,
        np.array([0.1, 0.0, 0.0]),
        jacobian,
    )
    assert not saturated["updated"]
    assert saturated["reason"] == "command_near_saturation"
    assert adapter.rejections == 2


def test_bounded_force_adapter_correction_never_exceeds_declared_bound():
    adapter = BoundedForceAdapter(correction_bound=0.12)
    adapter.theta[:] = [0.5, 0.25]
    for nominal in np.linspace(-1.0, 1.0, 101):
        command = adapter.command(float(nominal))
        assert -1.0 <= command <= 1.0
        assert abs(command - nominal) <= 0.12 + 1e-12


def test_count_homotopy_conserves_totals_and_uses_positive_ghosts():
    source_l = np.array([0.4, 0.6])
    source_m = np.array([0.3, 0.7])
    target_l = np.array([0.2, 0.3, 0.5])
    target_m = np.array([0.2, 0.2, 0.6])
    sl, sm, tl, tm = common_count_morphologies(source_l, source_m, target_l, target_m)
    assert sl.shape == tl.shape == (3,)
    assert sm.shape == tm.shape == (3,)
    assert np.isclose(sl[-1], 0.02)
    assert np.isclose(sm[-1], 1e-4)
    for progress in (0.0, 0.25, 1.0):
        lengths, masses = homotopy_morphology(
            source_l, source_m, target_l, target_m, progress
        )
        assert np.isclose(np.sum(lengths), 1.0)
        assert np.isclose(np.sum(masses), 1.0)
        assert np.all(lengths > 0.0)
        assert np.all(masses > 0.0)


def test_split_embedding_preserves_totals_and_defers_equal_cost_split_distally():
    source_lengths = np.full(8, 3.0 / 8.0)
    source_masses = np.full(8, 1.0 / 8.0)
    target_lengths = np.full(9, 3.0 / 9.0)
    target_masses = np.full(9, 1.0 / 9.0)

    result = split_embedding(
        source_lengths,
        source_masses,
        target_lengths,
        target_masses,
    )

    np.testing.assert_array_equal(result.split_counts, [1, 1, 1, 1, 1, 1, 1, 2])
    np.testing.assert_array_equal(result.segment_source_links, [0, 1, 2, 3, 4, 5, 6, 7, 7])
    np.testing.assert_array_equal(result.source_joint_locks, [0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert np.isclose(np.sum(result.source_lengths), np.sum(target_lengths))
    assert np.isclose(np.sum(result.source_masses), np.sum(target_masses))
    assert np.all(result.source_lengths > 0.0)
    assert np.all(result.source_masses > 0.0)


def test_split_embedding_optimizes_contiguous_unequal_target_partition():
    result = split_embedding(
        [0.4, 0.6],
        [0.3, 0.7],
        [0.2, 0.3, 0.5],
        [0.2, 0.2, 0.6],
    )

    np.testing.assert_array_equal(result.segment_source_links, [0, 0, 1])
    np.testing.assert_allclose(result.source_lengths, [0.16, 0.24, 0.6])
    np.testing.assert_allclose(result.source_masses, [0.12, 0.18, 0.7])
    np.testing.assert_array_equal(result.source_joint_locks, [0, 1, 0])


def test_split_joint_profile_zeros_only_inserted_joints():
    result = split_joint_profile([0.1, 0.2], [0, 0, 1], internal_value=0.0)
    np.testing.assert_allclose(result, [0.1, 0.0, 0.2])


def test_split_physical_lift_zeros_inserted_joint_and_scales_natural_time():
    lift = split_state_lift_matrix(2, [0, 0, 1], length_scale=4.0)
    state = np.array([0.5, 0.1, 0.2, 1.0, 0.3, 0.4])
    lifted = lift @ state
    np.testing.assert_allclose(
        lifted,
        [2.0, 0.1, 0.0, 0.2, 2.0, 0.15, 0.0, 0.2],
    )


def test_split_coordinate_projection_preserves_feedback_on_locked_manifold():
    assignments = np.array([0, 0, 1])
    lift = split_absolute_coordinate_lift_matrix(2, assignments)
    projection = split_state_projection(lift, [0.16, 0.24, 0.6])
    np.testing.assert_allclose(projection @ lift, np.eye(6), atol=1e-12)

    source_gain = np.array([[0.4, -0.2, 0.6, 0.1, -0.3, 0.7]])
    refined_gain = source_gain @ projection
    source_error = np.array([0.2, 0.1, -0.3, 0.4, -0.5, 0.6])
    assert np.isclose(
        (refined_gain @ (lift @ source_error)).item(),
        (source_gain @ source_error).item(),
    )


def test_adaptive_homotopy_grows_and_bisects_deterministically():
    schedule = AdaptiveHomotopy(step=0.01, maximum_step=0.1, growth=2.0)
    assert np.isclose(schedule.proposal(), 0.01)
    schedule.accept(schedule.proposal())
    assert np.isclose(schedule.progress, 0.01)
    assert np.isclose(schedule.step, 0.02)
    schedule.reject()
    assert np.isclose(schedule.step, 0.01)


def test_mirror_feedback_route_reflects_the_nominal_feedback_law():
    controls = np.asarray([0.2, -0.4])
    states = np.asarray([[1.0, 2.0], [0.5, 1.5], [-0.2, 0.3]])
    gains = np.asarray([[0.7, -0.1], [0.2, 0.9]])
    mirrored_u, mirrored_z, mirrored_k = mirror_feedback_route(controls, states, gains)
    np.testing.assert_allclose(mirrored_u, -controls)
    np.testing.assert_allclose(mirrored_z, -states)
    np.testing.assert_allclose(mirrored_k, gains)
    live = np.asarray([0.8, 2.3])
    original_action = controls[0] + gains[0] @ (live - states[0])
    mirrored_action = mirrored_u[0] + mirrored_k[0] @ (-live - mirrored_z[0])
    assert np.isclose(mirrored_action, -original_action)

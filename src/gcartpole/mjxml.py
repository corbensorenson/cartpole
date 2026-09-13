from __future__ import annotations

from html import escape

import numpy as np

from .morphology import Morphology


def _f(x: float) -> str:
    return f"{float(x):.9g}"


def joint_lock_impedance(strength: float, schedule: str = "linear") -> float:
    """Map a normalized lock strength to MuJoCo constraint impedance.

    ``linear`` preserves the historical diagnostic schedule. ``log_compliance``
    moves uniformly across the four orders of magnitude in ``1 - impedance``
    so early continuation steps do not multiply constraint compliance abruptly.
    """

    strength = max(0.0, min(1.0, float(strength)))
    minimum_impedance = 0.0001
    maximum_impedance = 0.9999
    if schedule == "linear":
        return minimum_impedance + (maximum_impedance - minimum_impedance) * strength
    if schedule == "log_compliance":
        release = 1.0 - strength
        minimum_compliance = 1.0 - maximum_impedance
        maximum_compliance = 1.0 - minimum_impedance
        compliance = float(
            np.exp(
                (1.0 - release) * np.log(minimum_compliance)
                + release * np.log(maximum_compliance)
            )
        )
        return 1.0 - compliance
    raise ValueError(
        "joint_lock_impedance_schedule must be 'linear' or 'log_compliance'"
    )


def _rigid_split_groups(morph: Morphology) -> dict[int, tuple[int, float]]:
    """Map group starts to inclusive ends and their common lock strength."""

    groups: dict[int, tuple[int, float]] = {}
    start = 0
    while start < morph.n_links:
        end = start
        strengths: list[float] = []
        while end + 1 < morph.n_links and float(morph.joint_lock[end + 1]) > 0.0:
            end += 1
            strengths.append(float(morph.joint_lock[end]))
        if end > start:
            groups[start] = (end, min(strengths))
        start = end + 1
    return groups


def generate_nlink_cartpole_xml(
    morph: Morphology,
    *,
    cart_mass: float = 1.0,
    rail_limit: float = 3.0,
    force_limit: float = 80.0,
    timestep: float = 0.005,
    cart_damping: float = 0.02,
    cart_frictionloss: float = 0.0,
    joint_armature: float = 0.0005,
    link_radius: float = 0.025,
    rigid_split_inertia: bool = False,
    joint_lock_impedance_schedule: str = "linear",
) -> str:
    """Generate a planar serial n-link inverted pendulum on a sliding cart.

    Coordinates:
      - cart slides along +x
      - links are upright along +z when all hinge angles are zero
      - hinge axis is +y, so motion is in the x-z plane
    """
    n = morph.n_links
    height = morph.total_length
    cam_y = max(7.0, 2.2 * height)
    cam_z = max(1.2, 0.55 * height)
    rail = float(rail_limit)
    cart_half_z = 0.07
    base_z = cart_half_z
    split_groups = _rigid_split_groups(morph) if rigid_split_inertia else {}
    group_for_link: dict[int, tuple[int, int, float]] = {}
    for group_start, (group_end, strength) in split_groups.items():
        for link in range(group_start, group_end + 1):
            group_for_link[link] = (group_start, group_end, strength)

    lines: list[str] = []
    lines.append(f'<mujoco model="gradient_{n}_link_cartpole">')
    lines.append('  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>')
    lines.append(f'  <option timestep="{_f(timestep)}" gravity="0 0 -9.81" integrator="RK4" iterations="20"/>')
    lines.append('  <visual>')
    lines.append('    <global offwidth="1280" offheight="720"/>')
    lines.append('  </visual>')
    lines.append('  <default>')
    lines.append('    <geom contype="0" conaffinity="0" friction="0 0 0"/>')
    lines.append(f'    <joint armature="{_f(joint_armature)}"/>')
    lines.append('  </default>')
    lines.append('  <worldbody>')
    lines.append('    <light name="key" pos="0 -4 6" dir="0 1 -1" diffuse="0.9 0.9 0.9"/>')
    lines.append(f'    <camera name="side" pos="0 -{_f(cam_y)} {_f(cam_z)}" xyaxes="1 0 0 0 0 1"/>')
    lines.append(f'    <geom name="rail" type="box" pos="0 0 -0.04" size="{_f(rail)} 0.025 0.025" rgba="0.35 0.35 0.35 1"/>')
    lines.append('    <body name="cart" pos="0 0 0">')
    lines.append(f'      <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-{_f(rail)} {_f(rail)}" damping="{_f(cart_damping)}" frictionloss="{_f(cart_frictionloss)}"/>')
    lines.append(f'      <geom name="cart_geom" type="box" size="0.18 0.12 {_f(cart_half_z)}" mass="{_f(cart_mass)}" rgba="0.1 0.25 0.9 1"/>')

    indent = '      '
    parent_pos = base_z
    for i in range(n):
        idx = i + 1
        length = morph.lengths[i]
        mass = float(morph.masses[i])
        damping = morph.damping[i]
        frictionloss = morph.frictionloss[i]
        stiffness = morph.joint_stiffness[i]
        rgba = "0.9 0.25 0.15 1" if i % 2 == 0 else "0.95 0.65 0.10 1"
        lines.append(f'{indent}<body name="link_{idx}" pos="0 0 {_f(parent_pos if i == 0 else morph.lengths[i-1])}">')
        indent += '  '
        spring = "" if stiffness <= 0.0 else f' stiffness="{_f(stiffness)}" springref="0"'
        armature = (
            float(joint_armature) * (1.0 - float(morph.joint_lock[i]))
            if rigid_split_inertia
            else float(joint_armature)
        )
        lines.append(
            f'{indent}<joint name="hinge_{idx}" type="hinge" axis="0 1 0" '
            f'damping="{_f(damping)}" frictionloss="{_f(frictionloss)}" '
            f'armature="{_f(armature)}"{spring}/>'
        )
        standard_mass = mass
        group = group_for_link.get(i)
        if group is not None:
            group_start, group_end, strength = group
            group_mass = float(np.sum(morph.masses[group_start : group_end + 1]))
            numerical_mass = 1.0e-8 * group_mass
            standard_mass = (1.0 - strength) * mass + strength * numerical_mass * (
                mass / group_mass
            )
        lines.append(
            f'{indent}<geom name="link_{idx}_geom" type="capsule" '
            f'fromto="0 0 0 0 0 {_f(length)}" size="{_f(link_radius)}" '
            f'mass="{_f(standard_mass)}" rgba="{escape(rgba)}"/>'
        )
        if i in split_groups:
            group_end, strength = split_groups[i]
            group_mass = float(np.sum(morph.masses[i : group_end + 1]))
            numerical_mass = 1.0e-8 * group_mass
            combined_mass = strength * (group_mass - numerical_mass)
            combined_length = float(np.sum(morph.lengths[i : group_end + 1]))
            lines.append(
                f'{indent}<geom name="link_{idx}_rigid_split_geom" type="capsule" '
                f'fromto="0 0 0 0 0 {_f(combined_length)}" size="{_f(link_radius)}" '
                f'mass="{_f(combined_mass)}" rgba="{escape(rgba)}"/>'
            )
        lines.append(f'{indent}<site name="tip_{idx}" pos="0 0 {_f(length)}" size="0.012" rgba="0 0 0 1"/>')

    # close nested link bodies + cart + worldbody
    for _ in range(n):
        indent = indent[:-2]
        lines.append(f'{indent}</body>')
    lines.append('    </body>')
    lines.append('  </worldbody>')
    locked = [i for i, value in enumerate(morph.joint_lock) if float(value) > 0.0]
    if locked:
        lines.append('  <equality>')
        for i in locked:
            # Continue the constraint through its dimensionless impedance.
            # MuJoCo clamps impedance to [1e-4, 0.9999], so the last positive
            # value is already nearly disabled before the equality disappears
            # at exactly zero.  A constant refsafe-compatible time constant
            # avoids changing both constraint stiffness knobs simultaneously.
            strength = max(0.0, min(1.0, float(morph.joint_lock[i])))
            impedance = joint_lock_impedance(
                strength, joint_lock_impedance_schedule
            )
            timeconst = max(2.0 * float(timestep), 1.0e-4)
            lines.append(
                f'    <joint joint1="hinge_{int(i) + 1}" '
                f'polycoef="0 0 0 0 0" solref="{_f(timeconst)} 1" '
                f'solimp="{_f(impedance)} {_f(impedance)} 0.001 0.5 2"/>'
            )
        lines.append('  </equality>')
    lines.append('  <actuator>')
    lines.append(f'    <motor name="cart_motor" joint="slide" gear="1" ctrllimited="true" ctrlrange="-{_f(force_limit)} {_f(force_limit)}"/>')
    lines.append('  </actuator>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"

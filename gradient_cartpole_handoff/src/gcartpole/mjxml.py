from __future__ import annotations

from html import escape

from .morphology import Morphology


def _f(x: float) -> str:
    return f"{float(x):.9g}"


def generate_nlink_cartpole_xml(
    morph: Morphology,
    *,
    cart_mass: float = 1.0,
    rail_limit: float = 3.0,
    force_limit: float = 80.0,
    timestep: float = 0.005,
    cart_damping: float = 0.02,
    joint_armature: float = 0.0005,
    link_radius: float = 0.025,
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
    lines.append(f'      <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-{_f(rail)} {_f(rail)}" damping="{_f(cart_damping)}"/>')
    lines.append(f'      <geom name="cart_geom" type="box" size="0.18 0.12 {_f(cart_half_z)}" mass="{_f(cart_mass)}" rgba="0.1 0.25 0.9 1"/>')

    indent = '      '
    parent_pos = base_z
    for i in range(n):
        idx = i + 1
        length = morph.lengths[i]
        mass = morph.masses[i]
        damping = morph.damping[i]
        rgba = "0.9 0.25 0.15 1" if i % 2 == 0 else "0.95 0.65 0.10 1"
        lines.append(f'{indent}<body name="link_{idx}" pos="0 0 {_f(parent_pos if i == 0 else morph.lengths[i-1])}">')
        indent += '  '
        lines.append(f'{indent}<joint name="hinge_{idx}" type="hinge" axis="0 1 0" damping="{_f(damping)}"/>')
        lines.append(f'{indent}<geom name="link_{idx}_geom" type="capsule" fromto="0 0 0 0 0 {_f(length)}" size="{_f(link_radius)}" mass="{_f(mass)}" rgba="{escape(rgba)}"/>')
        lines.append(f'{indent}<site name="tip_{idx}" pos="0 0 {_f(length)}" size="0.012" rgba="0 0 0 1"/>')

    # close nested link bodies + cart + worldbody
    for _ in range(n):
        indent = indent[:-2]
        lines.append(f'{indent}</body>')
    lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('  <actuator>')
    lines.append(f'    <motor name="cart_motor" joint="slide" gear="1" ctrllimited="true" ctrlrange="-{_f(force_limit)} {_f(force_limit)}"/>')
    lines.append('  </actuator>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"

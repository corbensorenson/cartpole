"""Offline quarter-turn transport of a tracking law; preparation stays literal."""
import mujoco as mj
import numpy as np


def quarter_turn_frames(env, turns):
    if not isinstance(turns, (int, np.integer)) or turns not in (0, 1, 2, 3):
        raise ValueError('quarter turns must be an integer from zero to three')
    rotations = (np.eye(2), np.array([[0., -1.], [1., 0.]]),
                 -np.eye(2), np.array([[0., 1.], [-1., 0.]]))
    force = rotations[turns]
    tangent = np.eye(2*env.model.nv)
    for start in (0, env.model.nv):
        tangent[start:start+2, start:start+2] = force
    return force, tangent


def rotate_world_state(env, state, turns):
    force, _ = quarter_turn_frames(env, turns)
    out = np.array(state, copy=True)
    if turns == 0:
        return out
    half = np.sqrt(.5)
    quaternion = (None, [half, 0., 0., half], [0., 0., 0., 1.],
                  [half, 0., 0., -half])[turns]
    out[:2] = force@state[:2]
    mj.mju_mulQuat(out[2:6], np.asarray(quaternion), state[2:6])
    nq = env.model.nq
    out[nq:nq+2] = force@state[nq:nq+2]
    return out


def rotate_tracking_payload(env, payload, turns):
    """Rotate the complete finite tracking law and its protected old boundary.

    All local joint velocities, including axial spin, survive. This helper
    edits reference arrays only; it does not forward or modify the environment.
    """
    force, tangent = quarter_turn_frames(env, turns)
    if any(bool(payload.get(k, False)) for k in
           ('momentum_parking', 'protected_native_schedule', 'physical_native_value',
            'world_coordinate_feedback', 'controller_bank')):
        raise ValueError('quarter-turn transport requires the original tracking family')
    out = {k: np.array(v, copy=True) for k, v in payload.items()}
    if turns == 0:
        return out
    for key in ('states',):
        out[key] = np.array([rotate_world_state(env, x, turns) for x in payload[key]])
    for key in ('reference', 'protected_boundary_states'):
        if key in payload:
            out[key] = rotate_world_state(env, payload[key], turns)
    for key in ('actions', 'feed'):
        out[key] = payload[key]@force.T
    for key in ('gains', 'K'):
        out[key] = force@payload[key]@tangent.T
    for key in ('roots', 'value_factor', 'protected_boundary_roots'):
        if key in payload:
            out[key] = payload[key]@tangent.T
    if 'defects' in payload:
        out['defects'] = payload['defects']@tangent.T
    for key in ('P', 'Q'):
        out[key] = tangent@payload[key]@tangent.T
    for key in ('R', 'tracking_R'):
        out[key] = force@payload[key]@force.T
    out['tracking_world_quarter_turns'] = np.array(turns)
    return out

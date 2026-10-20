"""Count-aware geometry for future full-native swing-up synthesis.

The frozen eleven-link implementation is deliberately left unchanged.
Only axial configuration gauge is quotiented; every physical rate remains.
"""
from dataclasses import dataclass
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.aligned_value import aligned_next_value
from spatialpole.control import linearize_many
from spatialpole.quotient import axial_quotient_frame
from spatialpole.world_coordinates import physical_tracking_factor, native_geometry, world_coordinate_jacobian


@dataclass(frozen=True)
class Dimensions:
    links: int
    state: int
    velocities: int
    tangent: int
    quotient: int
    model_native_steps: int

    @classmethod
    def from_config(cls, config):
        if config != Config(n_links=config.n_links):
            raise ValueError('The original benchmark configuration is required')
        nv = 2+3*config.n_links; tangent = 2*nv
        return cls(config.n_links, 4+7*config.n_links, nv, tangent,
                   tangent-config.n_links,
                   config.horizon*(1+4*(tangent+2))*config.frame_skip)


def error(env, reference, frames, state, phase):
    ref, frame, _ = aligned_next_value(env, reference['states'][phase], frames[phase], state)
    return frame@env.difference(state, ref)


def derivative(env, reference, frames, state, phase, eps=1e-5):
    d = Dimensions.from_config(env.config)
    assert frames.shape[1:] == (d.quotient, d.tangent)
    result = np.empty((d.quotient, d.tangent))
    for j in range(d.tangent):
        delta = np.zeros(d.tangent); delta[j] = eps
        result[:, j] = (error(env, reference, frames, env.integrate(state, delta), phase)
                       -error(env, reference, frames, env.integrate(state, -delta), phase))/(2*eps)
    return result


def physical_features(geometry, state, reference, reference_rotation, reference_omega, jacobian=False):
    """Smooth direction/rate residual with model-derived cart-velocity offsets."""
    n, nv, nq = geometry.config.n_links, geometry.model.nv, geometry.model.nq
    rotation, omega = native_geometry(geometry, state)
    angular = np.einsum('nji,nj->ni', reference_rotation, omega-reference_omega)
    angular_weight = np.sqrt([.05, .05, .005])
    residual = np.r_[2.*(state[:2]-reference[:2]),
        np.sqrt(40.)*(rotation[:, :, 2]-reference_rotation[:, :, 2]).ravel(),
        np.sqrt(.05)*(state[nq:nq+2]-reference[nq:nq+2]), (angular*angular_weight).ravel()]
    if not jacobian:
        return residual
    local = world_coordinate_jacobian(geometry, state)
    J = np.zeros((2*nv, 2*nv)); J[:2] = 2.*local[:2]
    for i in range(n):
        polar = local[2+2*i:4+2*i]
        J[2+3*i:5+3*i] = np.sqrt(40.)*(rotation[i, :, 0, None]*polar[1]-rotation[i, :, 1, None]*polar[0])
        start = 4+2*n+3*i
        current = local[start:start+3]
        J[nv+2+3*i:nv+5+3*i] = angular_weight[:, None]*(reference_rotation[i].T@rotation[i]@current)
    J[nv:nv+2] = np.sqrt(.05)*local[2+2*n:4+2*n]
    return residual, J


def defects(geometry, predicted, targets):
    residual = []
    for successor, target in zip(predicted, targets):
        rotation, omega = native_geometry(geometry, target)
        residual.append(physical_features(geometry, successor, target, rotation, omega))
    residual = np.array(residual)
    return .5*float(np.sum(residual*residual)), float(np.max(abs(residual))), residual


def corrector_model(native, geometry, reference, out):
    assert native.config == geometry.config
    d = Dimensions.from_config(native.config); H = native.config.horizon
    assert reference['states'].shape == (H+1, d.state) and reference['actions'].shape == (H, 2)
    native.derivative_workers = 4; before = native.native_steps
    raw, predicted = linearize_many(native, reference['states'][:-1],
        reference['actions'].astype(float), eps=.001, order=4, return_next=True)
    assert native.native_steps-before == d.model_native_steps
    charts = [axial_quotient_frame(geometry, x) for x in reference['states']]
    F = np.array([f['F'] for f in charts]); E = np.array([f['E'] for f in charts])
    factors = np.array([physical_tracking_factor(geometry, x) for x in reference['states']])
    assert all(np.linalg.matrix_rank(L[:, d.velocities:]) == d.velocities for L in factors)
    aa, bb, dd, jacobians = [], [], [], []
    for t, ((A, B), state) in enumerate(zip(raw, predicted)):
        J = derivative(geometry, reference, F, state, t+1)
        assert np.linalg.matrix_rank(J[:, d.velocities:]) == d.velocities
        aa.append(J@A@E[t]); bb.append(J@B)
        dd.append(error(geometry, reference, F, state, t+1)); jacobians.append(J)
    r = dict(A=np.array(aa), B=np.array(bb), d=np.array(dd), factors=factors@E, cart=E[:, :2])
    np.savez_compressed(out/'native_models.npz', reference_states=reference['states'],
        reference_actions=reference['actions'], raw_A=np.array([A for A, B in raw]),
        raw_B=np.array([B for A, B in raw]), predicted=predicted, E=E, F=F,
        successor_error_jacobian=np.array(jacobians), **r)
    return r, F, E, predicted, raw


class CorrectorPolicy:
    def __init__(self, env, reference, frames, coordinates, residual_controls):
        d = Dimensions.from_config(env.config); H = env.config.horizon
        assert reference['states'].shape == (H+1, d.state)
        assert frames.shape == (H+1, d.quotient, d.tangent)
        assert residual_controls.shape == (H, 2)
        self.geometry = SpatialCartpole(env.config)
        self.reference, self.frames, self.coordinates = reference, frames, coordinates
        self.residual_controls, self.dimension = residual_controls, d.quotient

    def __call__(self, state, phase):
        if np.array_equal(state, self.reference['states'][phase]):
            e = np.zeros(self.dimension)
        else:
            e = error(self.geometry, self.reference, self.frames, state, phase)
        c = self.coordinates
        return (self.reference['actions'][phase].astype(float)+c['feed'][phase]
                -c['K'][phase]@(c['S'][phase]@e)+c['Cinv'][phase]@self.residual_controls[phase])

    def close(self):
        assert self.geometry.native_steps == 0
        self.geometry.close()

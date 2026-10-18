"""Float32 CPU deployment of the pinned continuous mean MinGRU policy.

This is a distinct floating-point implementation, not a bit-parity GPU claim.
The native simulator state and every velocity remain untouched.
"""
from pathlib import Path
import numpy as np
from puffer_training_support import float_weight_layout


def sigmoid(x):
    z = np.exp(-np.abs(x))
    denominator = np.float32(1)+z
    return np.where(x >= 0, np.float32(1)/denominator, z/denominator)


class MeanMinGRU:
    def __init__(self, checkpoint, links=1):
        self.links, self.hidden, self.layers = links, 64, 2
        self.layout = float_weight_layout(links)
        raw = Path(checkpoint).read_bytes()
        assert len(raw) == self.layout['bytes']
        arrays = {}
        for tensor in self.layout['tensors']:
            arrays[tensor['name']] = np.frombuffer(raw, '<f4', tensor['elements'], tensor['byte_offset']).copy()
        assert all(np.isfinite(v).all() for v in arrays.values())
        self.encoder = arrays['encoder'].reshape(64, 7+7*links)
        self.decoder = arrays['decoder'].reshape(3, 64)
        self.recurrent = [arrays[f'mingru{i}'].reshape(192, 64) for i in range(2)]
        self.logstd = arrays['logstd']
        self.reset()

    def reset(self):
        self.memory = np.zeros((2, 64), np.float32)

    def __call__(self, observation):
        observation = np.asarray(observation, np.float32)
        assert observation.shape == (7+7*self.links,) and np.isfinite(observation).all()
        x = self.encoder@observation
        for i, weight in enumerate(self.recurrent):
            candidate, gate, highway = np.split(weight@x, 3)
            z, s = sigmoid(gate), sigmoid(highway)
            htilde = np.where(candidate >= 0, candidate+np.float32(.5), sigmoid(candidate))
            difference = htilde-self.memory[i]
            # Same numerically stable algebraic branches as algo.cu::lerp.
            h = np.where(np.abs(z) < np.float32(.5), self.memory[i]+z*difference,
                         htilde-difference*(np.float32(1)-z))
            self.memory[i] = h
            x = s*h+(np.float32(1)-s)*x
        output = self.decoder@x
        assert np.isfinite(output).all()
        return np.clip(output[:2], np.float32(-1e6), np.float32(1e6))


def observation(state, previous, step, links, canonical_sign=False):
    state = np.asarray(state, float)
    nq, nv = 2+4*links, 2+3*links
    assert state.shape == (nq+nv,)
    pose = state[2:nq].copy()
    if canonical_sign:
        for quaternion in pose.reshape(-1, 4):
            nonzero = np.flatnonzero(quaternion)
            assert len(nonzero)
            if quaternion[nonzero[0]] < 0:
                quaternion *= -1
            quaternion[quaternion == 0] = 0.  # canonical signed zero
    return np.r_[state[:2]/3., pose, state[nq:nq+2]/4., state[nq+2:]/15.,
                 np.asarray(previous, np.float32)/np.float32(80), step/1500.].astype(np.float32)


class CPUMeanPolicy:
    def __init__(self, env, checkpoint, canonical_sign=False):
        self.network = MeanMinGRU(checkpoint, env.config.n_links)
        self.previous = np.zeros(2, np.float32)
        self.canonical_sign, self.links = canonical_sign, env.config.n_links
        self.metadata = dict(policy='CPU float32 Puffer mean MinGRU deployment',
                             canonical_quaternion_sign=canonical_sign, park_steps=0, park_seconds=0.,
                             all_physical_velocities_retained=True, predictor_native_steps=0,
                             GPU_bit_identity_claim=False, hardware_realtime_claim=False)

    def __call__(self, state, step):
        obs = observation(state, self.previous, step, self.links, self.canonical_sign)
        return self.network(obs).astype(float)*80.

    def observe_transition(self, state, step, delivered):
        self.previous = delivered.copy()

"""Count-aware independent saved certificates and full-native model audits."""
from dataclasses import asdict
import json
import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from spatialpole.aligned_value import aligned_next_value
from spatialpole.direct_shooting import native_successors
from spatialpole.native_mpc import exact_bytes
from spatialpole.world_coordinates import physical_tracking_factor
from eleven_synthesis_protocol import sha
from eleven_value_sequence import forward
from spatial_synthesis_geometry import error, derivative


def load(path):
    with np.load(path) as z:
        return {name:z[name] for name in z.files}


def cone_error(y,H,nx):
    result=max(0.,float(-y[H*nx:H*nx+4*H].min()))
    for t in range(H):
        start=H*nx+4*H+3*t; v=y[start:start+3]
        result=max(result,float(np.linalg.norm(v[1:])-v[0]))
    return result


def certificate(path,c,initial,reference,actions):
    v=load(path);H,nx=len(actions),len(initial);exact_bytes(v['initial'],initial)
    if 'kind' in v:
        assert str(v['kind'])=='zero_residual';z=v['white_states'];w=np.zeros((H,2))
        residual=z[1:]-np.einsum('tij,tj->ti',c['A'],z[:-1])-c['d']
        assert np.max(abs(residual))<1e-6
    else:
        P=sparse.load_npz(path.with_name(path.stem+'_P.npz'));A=sparse.load_npz(path.with_name(path.stem+'_A.npz'))
        x,y,s,b=v['decision'],v['dual'],v['slack'],v['b'];px,aty=P@x,A.T@y
        primal=float(np.max(abs(A@x+s-b)));dual=float(np.max(abs(px+aty))/max(1.,float(np.max(abs(px))),float(np.max(abs(aty)))))
        objective=.5*float(x@px);gap=abs(float(x@px+b@y))/max(1.,2*objective);comp=abs(float(s@y))/max(1.,2*objective)
        cone=max(float(np.max(abs(s[:H*nx]))),cone_error(s,H,nx),cone_error(y,H,nx))
        assert max(primal,dual,gap,comp,cone)<1e-6
        z=np.vstack([c['S'][0]@initial,x[:H*nx].reshape(H,nx)]);w=x[H*nx:].reshape(H,2)
    errors=np.array([solve_triangular(S,zz) for S,zz in zip(c['S'],z)])
    controls=actions+c['feed']-np.einsum('tij,tj->ti',c['K'],z[:-1])+np.einsum('tij,tj->ti',c['Cinv'],w)
    assert np.linalg.norm(controls,axis=1).max()<=80.+1e-5
    cart=reference[:,:2]+np.einsum('tij,tj->ti',c['cart'],z)
    assert np.max(abs(cart[1:]))<=3.+1e-6
    return controls,errors,w


def models(r,errors,du):
    residual=errors[1:]-np.einsum('tij,tj->ti',r['A'],errors[:-1])-np.einsum('tij,tj->ti',r['B'],du)-r['d']
    result=float(np.max(abs(residual)));assert result<1e-6;return result


def audit(folder, geometry, native, out, index):
    nx = 2*geometry.model.nv-geometry.config.n_links
    nv = geometry.model.nv
    v, m, c, plan, tape = [load(folder/name) for name in
        ('virtual_reference.npz', 'native_models.npz', 'coordinates.npz',
         'model_plan.npz', 'native_forecast.npz')]
    row = json.loads((folder/'summary.json').read_text())
    exact_bytes(v['states'], m['reference_states'])
    exact_bytes(v['actions'], m['reference_actions'])
    assert json.loads(str(tape['config'])) == asdict(geometry.config)
    centers = native_successors(native, v['states'][:-1], v['actions'])
    exact_bytes(centers, m['predicted'])
    frames = m['F']
    defects = np.array([error(geometry, v, frames, x, t+1) for t, x in enumerate(centers)])
    exact_bytes(defects, m['d'])
    mapped_A = m['successor_error_jacobian'] @ m['raw_A'] @ m['E'][:-1]
    mapped_B = m['successor_error_jacobian'] @ m['raw_B']
    np.testing.assert_allclose(mapped_A, m['A'], atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(mapped_B, m['B'], atol=1e-12, rtol=1e-13)
    factors = np.array([physical_tracking_factor(geometry, x) for x in v['states']]) @ m['E']
    exact_bytes(factors, m['factors'])
    assert all(np.linalg.matrix_rank(E[nv:]) == nv for E in m['E'])
    assert all(np.linalg.matrix_rank(J[:, nv:]) == nv for J in m['successor_error_jacobian'])
    controls, errors, w = certificate(folder/'certificate.npz', c, np.zeros(nx),
                                      v['states'], v['actions'].astype(float))
    np.testing.assert_allclose(controls, plan['controls'], atol=2e-12, rtol=1e-13)
    exact_bytes(errors, plan['errors']); exact_bytes(w, plan['w'])
    residual = models(m, errors, controls-v['actions'])
    def cost(e, du):
        f = np.einsum('tij,tj->ti', factors[1:], e[1:])
        return .5*float(np.sum(f*f)+.001*np.sum(du*du))
    score = cost(errors, controls-v['actions'])
    _, zero_errors, zero_du = forward(c, np.zeros(nx), np.zeros((1500, 2)))
    completion = abs(score-cost(zero_errors, zero_du)-.5*np.sum(w*w))/max(1., abs(score))
    assert completion < 1e-6 and abs(score-row['model_cost']) < 1e-9
    exact_bytes(tape['states'][0], v['states'][0])
    streak = best = 0
    for t, action in enumerate(tape['actions']):
        state = tape['states'][t]
        if np.array_equal(state, v['states'][t]):
            e = np.zeros(nx)
        else:
            reference, frame, _ = aligned_next_value(geometry, v['states'][t], m['F'][t], state)
            e = frame @ geometry.difference(state, reference)
        raw = v['actions'][t].astype(float)+c['feed'][t]-c['K'][t]@(c['S'][t]@e)+c['Cinv'][t]@w[t]
        exact_bytes(raw, tape['raw_actions'][t]); exact_bytes(geometry.deliver(raw), action)
        geometry.set_state(tape['states'][t+1]); angles = geometry.tilts()
        np.testing.assert_allclose(angles, tape['tilts'][t], atol=1e-14, rtol=0)
        streak = streak+1 if angles.max() < .15 else 0; best = max(best, streak)
    result = row['result']
    assert result['cold_replay_byte_exact'] and result['inclusive_native_steps'] == 8*len(tape['actions'])
    assert best*.02 == result['best_hold_seconds']
    feasible = bool(len(tape['actions']) == 1500 and best*.02 >= 5. and
                    not result['physics_warning'] and result['plane_margin'] >= 0.)
    assert feasible == result['continuation_feasible'] and not result['benchmark_success']
    checks = []; rng = np.random.default_rng(47191+index)
    for t in (0, 160, 205, 325, 1000):
        Jcheck = derivative(geometry, v, frames, centers[t], t+1, 5e-6)
        jacobian_error = float(np.max(abs(Jcheck-m['successor_error_jacobian'][t])))
        assert jacobian_error < 1e-6
        direction = rng.normal(size=nx); direction /= np.linalg.norm(direction)
        dx = m['E'][t] @ direction; u = v['actions'][t].astype(float)
        du = -u/max(1., float(np.linalg.norm(u))); eps = 1e-4
        xs = np.array([geometry.integrate(v['states'][t], k*eps*dx) for k in range(1, 5)])
        us = np.array([u+k*eps*du for k in range(1, 5)])
        assert np.linalg.norm(us, axis=1).max() <= 80.
        successors = native_successors(native, xs, us)
        values = np.array([error(geometry, v, frames, x, t+1) for x in successors])
        measured = (-25.*defects[t]+48.*values[0]-36.*values[1]+16.*values[2]-3.*values[3])/(12.*eps)
        expected = m['A'][t]@direction+m['B'][t]@du
        discrepancy = float(np.max(abs(measured-expected))); assert discrepancy < 1e-5
        checks.append(dict(phase=t, directional_error=discrepancy, jacobian_error=jacobian_error))
        np.savez_compressed(out/f'model{index}_probe{t}.npz', states=xs, actions=us,
                            successors=successors, measured=measured, expected=expected)
    print(json.dumps(dict(folder=str(folder), passed=True, callbacks=len(tape['actions']),
                          full_native_success=feasible)), flush=True)
    return dict(folder=str(folder), passed=True, callbacks=len(tape['actions']),
        full_native_success=feasible, best_hold_seconds=best*.02, checks=checks,
        mapped_residual=residual, completion_error=completion, model_cost=score,
        maximum_explicit_defect=float(np.max(abs(defects))),
        source_summary_sha256=sha(folder/'summary.json'))


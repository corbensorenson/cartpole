import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.quotient import axial_quotient_frame,quotient_tracking_values
from spatialpole.conditioning import axial_gauge_state,axial_gauge_route
from spatialpole.physical import directions_and_rates
from spatialpole.aligned_value import aligned_next_value


@pytest.mark.parametrize('n',[1,3,14])
def test_analytic_axial_alignment_derivative_and_physical_generators(n):
    env = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(117+n)
    nx = 2*env.model.nv
    reference = env.integrate(env.upright_state(),rng.normal(size=nx)*.3)
    before = env.state()
    f = axial_quotient_frame(env,reference)
    np.testing.assert_array_equal(env.state(),before)
    np.testing.assert_allclose(f['C']@f['D'],np.eye(n),atol=2e-15)
    np.testing.assert_allclose(f['G']@f['G'],f['G'],atol=3e-15)
    np.testing.assert_allclose(f['F']@f['E'],np.eye(nx-n),atol=3e-15)
    np.testing.assert_allclose(f['E']@f['F'],f['G'],atol=4e-15)
    # Every original velocity tangent, including physical twist, survives G.
    np.testing.assert_array_equal(f['G'][:,env.model.nv:],np.eye(nx)[:,env.model.nv:])
    eps = 1e-6
    scratch = SpatialCartpole(env.config)
    for _ in range(4):
        direction = rng.normal(size=nx)
        values,angles = [],[]
        for sign in (-1.,1.):
            state = env.integrate(reference,sign*eps*direction)
            ref,_,angle = aligned_next_value(scratch,reference,np.eye(nx),state)
            values.append(env.difference(state,ref))
            angles.append(angle)
        np.testing.assert_allclose((values[1]-values[0])/(2*eps),f['G']@direction,atol=2e-8,rtol=2e-8)
        np.testing.assert_allclose((angles[1]-angles[0])/(2*eps),f['C']@direction,atol=2e-8,rtol=2e-8)
    a = rng.normal(size=n)
    minus,plus = [axial_gauge_state(env,reference,sign*eps*a) for sign in (-1.,1.)]
    observed = (env.difference(plus,reference)-env.difference(minus,reference))/(2*eps)
    np.testing.assert_allclose(observed,f['D']@a,atol=5e-9,rtol=5e-9)
    changed = axial_gauge_state(env,reference,a)
    for original,transformed in zip(directions_and_rates(env,reference),directions_and_rates(env,changed)):
        np.testing.assert_allclose(original,transformed,atol=3e-14)
    # Full native action flow also commutes with the frame relabeling.
    force = np.array([3.,-2.],np.float32)
    native = scratch.transition(reference,force)
    transformed = scratch.transition(changed,force)
    np.testing.assert_allclose(transformed,axial_gauge_state(env,native,a),atol=5e-13,rtol=5e-13)
    scratch.close()
    env.close()


@pytest.mark.parametrize('n',[1,3,14])
def test_planar_quotient_keeps_both_physical_branches_and_all_velocities(n):
    env = SpatialCartpole(Config(n_links=n))
    state = env.upright_state()
    for i in range(n):
        angle = .17*np.sin(i+1)
        state[2+4*i:6+4*i] = [np.cos(angle/2),0.,np.sin(angle/2),0.]
        state[env.model.nq+3+3*i] = .3*np.cos(i)
    f = axial_quotient_frame(env,state,planar=True)
    assert f['E'].shape==(2*env.model.nv,2*env.model.nv-n)
    assert sum(f['signs']>0)==2*n+2 and sum(f['signs']<0)==3*n+2
    np.testing.assert_allclose(f['F']@f['E'],np.eye(2*env.model.nv-n),atol=3e-15)
    np.testing.assert_allclose(f['C']@f['E'],0.,atol=3e-15)
    nv=env.model.nv
    assert np.linalg.matrix_rank(f['F'][:,nv:])==nv
    env.close()


@pytest.mark.parametrize('planar',[True,False])
def test_quotient_affine_qr_matches_independent_full_horizon_condensation(planar):
    env = SpatialCartpole(Config(n_links=1))
    rng = np.random.default_rng(735)
    nx,H = 2*env.model.nv,4
    states = [env.upright_state() for _ in range(H+1)]
    for i,state in enumerate(states):
        state[2:6] = [np.cos(.05*i),0.,np.sin(.05*i),0.]
        state[env.model.nq+3] = .3*i
    if not planar:
        states=[env.integrate(s,rng.normal(size=nx)*.2) for s in states]
    frames = [axial_quotient_frame(env,s,planar) for s in states]
    # Reflection-consistent synthetic FULL models with nonzero affine defects.
    from spatialpole.symmetry import reflection_signs
    parity = reflection_signs(env)
    models=[]
    for _ in range(H):
        A,B = rng.normal(size=(nx,nx))*.2,rng.normal(size=(nx,2))
        models.append((np.where(parity[:,None]==parity[None,:],A,0.),
                       np.where(parity[:,None]==np.array([1.,-1.])[None,:],B,0.)) if planar else (A,B))
    d = rng.normal(size=(H,nx))*.1
    if planar:
        d[:,parity<0] = 0.
    L,R = np.diag(rng.uniform(.2,2,nx)),np.diag([.3,.7])
    result = quotient_tracking_values(models,d,[L]*H,R,L,frames)
    ny = nx-1
    y = rng.normal(size=ny)
    initial = y.copy()
    influence,bias = np.zeros((ny,H*2)),y.copy()
    M,b = [],[]
    for t,(A,B) in enumerate(result['reduced_models']):
        factor = L@frames[t]['E']
        M.append(factor@influence); b.append(factor@bias)
        influence,bias = A@influence,A@bias+result['reduced_defects'][t]
        influence[:,2*t:2*t+2] += B
    M.extend([L@frames[-1]['E']@influence,np.kron(np.eye(H),np.linalg.cholesky(R).T)])
    b.extend([L@frames[-1]['E']@bias,np.zeros(H*2)])
    optimum = np.linalg.lstsq(np.vstack(M),-np.concatenate(b),rcond=None)[0].reshape(H,2)
    actual=[]
    for t,(A,B) in enumerate(result['reduced_models']):
        force = result['feed'][t]-result['reduced_gains'][t]@y
        actual.append(force)
        error = frames[t]['E']@y+frames[t]['D']@rng.normal(size=1)
        np.testing.assert_allclose(result['feed'][t]-result['gains'][t]@error,force,atol=2e-14)
        np.testing.assert_allclose(result['roots'][t]@error+result['residuals'][t],
            result['reduced_roots'][t]@y+result['residuals'][t],atol=2e-14)
        y = A@y+B@force+result['reduced_defects'][t]
    np.testing.assert_allclose(actual,optimum,atol=3e-13,rtol=3e-13)
    assert np.linalg.norm(result['feed'])>1e-3
    env.close()

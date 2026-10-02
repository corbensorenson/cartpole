"""Independent native-mechanics and transition checks for the MP model derivation."""
import copy
import mujoco
import numpy as np
import pytest
from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.upright_mechanics import upright_rk4_matrices
from scripts.make_lqr_checkpoint import finite_difference_dynamics


@pytest.mark.parametrize('count',[1,7,14,20])
def test_derived_mechanics_matches_native_force_and_transition(count):
    cfg=copy.deepcopy(load_config('configs/swingup13_uniform.yaml'));cfg['env']['n_links']=count
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
    try:
        a,b,diag=upright_rk4_matrices(env,decimal_digits=50)
        gravity=np.array(diag['gravity_hessian'],dtype=float)
        native=np.zeros_like(gravity)
        h=1e-6
        data=mujoco.MjData(env.model)
        for j in range(count+1):
            data.qpos[:]=0;data.qpos[j]=h;mujoco.mj_forward(env.model,data)
            plus=data.qfrc_bias.copy()
            data.qpos[j]=-h;mujoco.mj_forward(env.model,data)
            native[:,j]=(plus-data.qfrc_bias)/(2*h)
        np.testing.assert_allclose(gravity,native,rtol=2e-12,atol=1e-13)
        fd_a,fd_b=finite_difference_dynamics(cfg,1.,1e-8)
        np.testing.assert_allclose(np.array(a.tolist(),dtype=float),fd_a,rtol=2e-10,atol=2e-11)
        np.testing.assert_allclose(np.array(b.tolist(),dtype=float),fd_b,rtol=2e-10,atol=2e-11)
        assert diag['maximum_mass_difference_from_native']<1e-13
    finally:
        env.close()


def test_reject_non_rk4_and_spring_models():
    cfg=load_config('configs/swingup13_uniform.yaml')
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
    try:
        env.model.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
        with pytest.raises(ValueError,match='passive'):
            upright_rk4_matrices(env)
        env.model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
        env.model.jnt_stiffness[1]=1.
        with pytest.raises(ValueError,match='passive'):
            upright_rk4_matrices(env)
    finally:
        env.close()

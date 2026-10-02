import mujoco
import numpy as np
import pytest

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.upright_linearization import planar_upright_rk4
from scripts.make_lqr_checkpoint import finite_difference_dynamics


@pytest.mark.parametrize('n', [1, 3, 12])
def test_structured_mass_and_gravity_match_independent_mujoco_quantities(n):
    cfg=load_config('configs/swingup12_uniform.yaml');cfg['env']['n_links']=n
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
    try:
        _,_,info=planar_upright_rk4(env.model,env.force_limit,env.frame_skip,decimal_digits=50)
        data=mujoco.MjData(env.model);mujoco.mj_forward(env.model,data)
        mass=np.empty((n+1,n+1));mujoco.mj_fullM(env.model,mass,data.qM)
        np.testing.assert_allclose(np.asarray(info['mass_matrix'].tolist(),float),mass,rtol=3e-14,atol=3e-15)
        gravity=np.zeros_like(mass);eps=1e-5
        for j in range(n+1):
            data.qpos[:]=0;data.qvel[:]=0;data.qacc[:]=0;data.qpos[j]=eps
            mujoco.mj_inverse(env.model,data);upper=data.qfrc_inverse.copy()
            data.qpos[j]=-eps;mujoco.mj_inverse(env.model,data)
            gravity[:,j]=-(upper-data.qfrc_inverse)/(2*eps)
        np.testing.assert_allclose(np.asarray(info['gravity_matrix'].tolist(),float),gravity,rtol=3e-11,atol=1e-10)
    finally:env.close()


@pytest.mark.parametrize('frame_skip', [1,2,4])
def test_structured_rk4_map_matches_actual_physics_derivative(frame_skip):
    cfg=load_config('configs/swingup12_uniform.yaml');cfg['env']['n_links']=3;cfg['env']['frame_skip']=frame_skip
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
    try:
        a,b,_=planar_upright_rk4(env.model,env.force_limit,frame_skip,decimal_digits=50)
        numerical_a,numerical_b=finite_difference_dynamics(cfg,1.,1e-6)
        np.testing.assert_allclose(np.asarray(a.tolist(),float),numerical_a,rtol=2e-10,atol=2e-11)
        np.testing.assert_allclose(np.asarray(b.tolist(),float),numerical_b,rtol=2e-10,atol=2e-11)
    finally:env.close()


def test_structured_model_rejects_other_axes_and_non_rk4_physics():
    cfg=load_config('configs/swingup12_uniform.yaml');cfg['env']['n_links']=2
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
    try:
        env.model.jnt_axis[1]=[0,0,1]
        with pytest.raises(ValueError,match='axes'):planar_upright_rk4(env.model,env.force_limit,4)
        env.model.jnt_axis[1]=[0,1,0];env.model.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
        with pytest.raises(ValueError,match='RK4'):planar_upright_rk4(env.model,env.force_limit,4)
    finally:env.close()
